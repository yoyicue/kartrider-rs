//! Clean-room parser for KartRider **kart vehicle** animation clips (`fXX.1s`,
//! MAGIC_KART_ANIM): a 55-channel rigid-body format. Per channel, position /
//! combined-PRS / euler-rotation curves (Cubic/Step) are sampled to per-frame
//! keyframes; rotation is cubic-Hermite resampled onto the position keyframe
//! times. Ported from the validated C# `KartAnimParser` (the project's own
//! reimplementation, not the GPL KartLibrary). See docs/kartrider-1s-format.md.

pub type Vec3 = [f32; 3];
/// A sampled per-frame curve: `(time_ms, value)` keyframes.
type Vec3Track = Vec<(i32, Vec3)>;

const MAGIC_KART_ANIM: &[u8] = &[0xAA, 0x47, 0xCB, 0x04, 0x8A, 0x1E];
const MAGIC_ANIM: &[u8] = &[0xAA, 0x47, 0x2A, 0x05, 0x4B, 0x22];
const MAGIC_UNK67: &[u8] = &[0xAA, 0x47, 0x67, 0x05, 0x0D, 0x25];

#[derive(Default)]
pub struct KartAnimClip {
    pub name: String,
    pub num_frames: i32,
    pub duration_ms: i32,
    pub channels: Vec<KartAnimChannel>,
}

#[derive(Default, Clone)]
pub struct KartAnimChannel {
    pub node_index: i32,
    pub anim_id: i32,
    pub keyframes: Vec<KartKeyframe>,
}

#[derive(Clone, Copy)]
pub struct KartKeyframe {
    pub time_ms: i32,
    pub pos: Vec3,
    pub rot_euler: Vec3,
    pub scale: Vec3,
}

impl KartAnimChannel {
    pub fn is_animated(&self) -> bool {
        if self.keyframes.len() <= 1 {
            return false;
        }
        let k0 = &self.keyframes[0];
        self.keyframes
            .iter()
            .any(|k| k.pos != k0.pos || k.rot_euler != k0.rot_euler || k.scale != k0.scale)
    }
}

pub fn is_kart_anim(d: &[u8]) -> bool {
    d.len() >= 6 && &d[..6] == MAGIC_KART_ANIM
}

// ---- low-level reads -------------------------------------------------------
fn u32a(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn i32a(d: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn f32a(d: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn vec3a(d: &[u8], o: usize) -> Vec3 {
    [f32a(d, o), f32a(d, o + 4), f32a(d, o + 8)]
}
fn is_time(raw: u32) -> bool {
    ((raw >> 23) & 0xFF) == 0 && raw != 0 && raw != 0x80000000
}

#[derive(Clone, Copy, PartialEq)]
enum BType {
    Anim,
    Unk67,
    Hdr,
    Aa27,
}

pub fn parse(d: &[u8], name: &str) -> Option<KartAnimClip> {
    if !is_kart_anim(d) || d.len() < 20 {
        return None;
    }
    let blocks = find_blocks(d);
    let channels = group_channels(&blocks);
    let mut clip = KartAnimClip {
        name: name.to_string(),
        ..Default::default()
    };

    let mut max_kf = 0;
    for ch in &channels {
        let anim_id = if ch.anim_off + 10 <= d.len() {
            u32a(d, ch.anim_off + 6) as i32
        } else {
            -1
        };
        let mut c = KartAnimChannel {
            node_index: clip.channels.len() as i32,
            anim_id,
            keyframes: Vec::new(),
        };
        build_keyframes(d, ch, 0, &mut c);
        max_kf = max_kf.max(c.keyframes.len());
        clip.channels.push(c);
    }
    clip.num_frames = max_kf as i32;

    // Pass 2: regrow short channels using the clip frame count as a hint.
    if clip.num_frames > 1 {
        for i in 0..channels.len() {
            if clip.channels[i].keyframes.len() < clip.num_frames as usize {
                let mut tmp = KartAnimChannel {
                    node_index: clip.channels[i].node_index,
                    anim_id: clip.channels[i].anim_id,
                    keyframes: Vec::new(),
                };
                build_keyframes(d, &channels[i], clip.num_frames, &mut tmp);
                if tmp.keyframes.len() > clip.channels[i].keyframes.len() {
                    clip.channels[i].keyframes = tmp.keyframes;
                }
            }
        }
    }

    clip.duration_ms = 0;
    for c in &clip.channels {
        for kf in &c.keyframes {
            clip.duration_ms = clip.duration_ms.max(kf.time_ms);
        }
    }
    Some(clip)
}

// ---- block discovery -------------------------------------------------------
struct Chan {
    anim_off: usize,
    anim_aa27: Vec<(usize, usize)>, // (off, size)
}

fn find_blocks(d: &[u8]) -> Vec<(BType, usize, usize)> {
    let len = d.len();
    let mut raw: Vec<(BType, usize)> = Vec::new();
    let mut i = 0usize;
    while i + 1 < len {
        if i + 6 <= len {
            let s = &d[i..i + 6];
            if s == MAGIC_ANIM {
                raw.push((BType::Anim, i));
                i += 6;
                continue;
            }
            if s == MAGIC_UNK67 {
                raw.push((BType::Unk67, i));
                i += 6;
                continue;
            }
            if i > 0 && s == MAGIC_KART_ANIM {
                raw.push((BType::Hdr, i));
                i += 6;
                continue;
            }
        }
        if d[i] == 0xAA && i + 1 < len && d[i + 1] == 0x27 {
            let overlap = i >= 4
                && ((d[i - 4] == 0x2A && d[i - 3] == 0x05 && d[i - 2] == 0x4B && d[i - 1] == 0x22)
                    || (d[i - 4] == 0x67
                        && d[i - 3] == 0x05
                        && d[i - 2] == 0x0D
                        && d[i - 1] == 0x25)
                    || (d[i - 4] == 0xCB
                        && d[i - 3] == 0x04
                        && d[i - 2] == 0x8A
                        && d[i - 1] == 0x1E));
            if !overlap {
                raw.push((BType::Aa27, i));
            }
        }
        i += 1;
    }
    raw.sort_by_key(|x| x.1);
    let mut outp = Vec::with_capacity(raw.len());
    for k in 0..raw.len() {
        let end = if k + 1 < raw.len() {
            raw[k + 1].1
        } else {
            d.len()
        };
        outp.push((raw[k].0, raw[k].1, end - raw[k].1));
    }
    outp
}

fn group_channels(blocks: &[(BType, usize, usize)]) -> Vec<Chan> {
    let mut channels: Vec<Chan> = Vec::new();
    let mut cur: Option<Chan> = None;
    let mut has_key = false;
    for &(t, off, size) in blocks {
        match t {
            BType::Hdr => {}
            BType::Anim => {
                if let Some(c) = cur.take() {
                    channels.push(c);
                }
                cur = Some(Chan {
                    anim_off: off,
                    anim_aa27: Vec::new(),
                });
                has_key = false;
            }
            BType::Unk67 => {
                if cur.is_some() {
                    has_key = true;
                }
            }
            BType::Aa27 => {
                if let Some(c) = cur.as_mut() {
                    if !has_key {
                        c.anim_aa27.push((off, size));
                    }
                }
            }
        }
    }
    if let Some(c) = cur {
        channels.push(c);
    }
    channels
}

// ---- per-channel sampling --------------------------------------------------
fn build_keyframes(d: &[u8], ch: &Chan, hint: i32, outc: &mut KartAnimChannel) {
    let g = parse_aa27_group(d, &ch.anim_aa27, hint);
    let n = g.pos.len().max(g.rot.len()).max(g.scale.len()).max(1);
    for fi in 0..n {
        let t = if fi < g.times.len() {
            g.times[fi]
        } else {
            fi as i32
        };
        outc.keyframes.push(KartKeyframe {
            time_ms: t,
            pos: pick(&g.pos, fi, [0.0; 3]),
            rot_euler: pick(&g.rot, fi, [0.0; 3]),
            scale: pick(&g.scale, fi, [1.0; 3]),
        });
    }
}

fn pick(l: &[Vec3], i: usize, def: Vec3) -> Vec3 {
    if l.is_empty() {
        def
    } else if i < l.len() {
        l[i]
    } else {
        l[l.len() - 1]
    }
}

#[derive(Default)]
struct Group {
    pos: Vec<Vec3>,
    rot: Vec<Vec3>,
    scale: Vec<Vec3>,
    times: Vec<i32>,
}

fn parse_aa27_group(d: &[u8], aa27s: &[(usize, usize)], hint: i32) -> Group {
    let mut g = Group::default();
    let mut pos_kf: Vec<(i32, Vec3)> = Vec::new();
    let mut scale_kf: Vec<(i32, Vec3)> = Vec::new();
    let mut rot_combined: Vec<(i32, Vec3)> = Vec::new();
    let mut rot_payload: Option<(usize, usize)> = None;

    // parsed: (ct, n, innerStart, innerEnd, payStart, payEnd)
    let mut parsed: Vec<(u32, u32, usize, usize, usize, usize)> = Vec::new();
    for &(off, size) in aa27s {
        let pay_start = off + 4;
        let pay_end = off + size;
        if pay_end < pay_start + 8 {
            continue;
        }
        let ct = u32a(d, pay_start);
        let n = u32a(d, pay_start + 4);
        parsed.push((ct, n, pay_start + 12, pay_end, pay_start, pay_end));
    }

    // Pass A: combined PRS
    for b in &parsed {
        if b.0 != 0 || b.1 == 0 {
            continue;
        }
        let (s, r, p) = read_combined(d, b.2, b.3, b.1 as usize);
        if scale_kf.is_empty() {
            scale_kf = s;
            rot_combined = r;
            pos_kf = p;
        } else {
            if !s.is_empty() {
                scale_kf = s;
            }
            if !r.is_empty() {
                rot_combined = r;
            }
            if !p.is_empty() {
                pos_kf = p;
            }
        }
    }
    // Pass B: dedicated position / rotation
    for b in &parsed {
        if b.0 == 1 && b.1 > 0 {
            pos_kf = read_channel(d, b.2, b.3, b.1 as usize);
        } else if b.0 == 4 {
            rot_payload = Some((b.4 + 8, b.5));
        }
    }

    let mut rot_values: Option<Vec<Vec3>> = None;
    let mut rot_times_from_data = false;
    if let Some(rp) = rot_payload {
        let rtimes: Vec<i32> = if !pos_kf.is_empty() {
            pos_kf.iter().map(|k| k.0).collect()
        } else if !scale_kf.is_empty() {
            scale_kf.iter().map(|k| k.0).collect()
        } else {
            let mut rt = extract_rot_times(d, rp.0, rp.1);
            rot_times_from_data = !rt.is_empty();
            if rt.is_empty() {
                let cnt = if hint > 1 { hint } else { 1 };
                rt = (0..cnt).collect();
            }
            rt
        };
        let rot = read_rot_keyframes(d, rp.0, rp.1, &rtimes);
        if !rot.is_empty() {
            rot_values = Some(rot);
        }
    }

    g.pos = pos_kf.iter().map(|k| k.1).collect();
    g.scale = scale_kf.iter().map(|k| k.1).collect();
    g.rot = rot_values.unwrap_or_else(|| rot_combined.iter().map(|k| k.1).collect());

    let pos_times: Vec<i32> = pos_kf.iter().map(|k| k.0).collect();
    let scale_times: Vec<i32> = scale_kf.iter().map(|k| k.0).collect();
    if pos_times.len() >= scale_times.len() {
        g.times = if !pos_times.is_empty() {
            pos_times
        } else if rot_times_from_data {
            if let Some(rp2) = rot_payload {
                extract_rot_times(d, rp2.0, rp2.1)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
    } else {
        g.times = scale_times;
    }
    g
}

fn read_channel(d: &[u8], mut off: usize, end: usize, n_kf: usize) -> Vec3Track {
    let mut keys = Vec::new();
    let mut t = 0i32;
    for ki in 0..n_kf {
        if off + 12 > end {
            break;
        }
        keys.push((t, vec3a(d, off)));
        off += 12;
        if off + 4 <= end && ki < n_kf - 1 {
            let raw = u32a(d, off);
            if is_time(raw) {
                t = raw as i32;
                off += 4;
            }
        }
    }
    keys
}

fn read_combined(
    d: &[u8],
    mut off: usize,
    end: usize,
    n_kf: usize,
) -> (Vec3Track, Vec3Track, Vec3Track) {
    let (mut s, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
    let mut t = 0i32;
    for _ in 0..n_kf {
        if off + 36 > end {
            break;
        }
        s.push((t, vec3a(d, off)));
        r.push((t, vec3a(d, off + 12)));
        p.push((t, vec3a(d, off + 24)));
        off += 36;
        if off + 4 <= end {
            let raw = u32a(d, off);
            if is_time(raw) {
                t = raw as i32;
                off += 4;
            }
        }
    }
    (s, r, p)
}

// ---- rotation axes (cubic Hermite) -----------------------------------------
type AxisKey = (i32, f64, f64, f64); // (t, value, leftSlope, rightSlope)

fn read_axis(d: &[u8], mut off: usize, end: usize) -> (Vec<AxisKey>, usize) {
    let mut keys: Vec<AxisKey> = Vec::new();
    if off + 8 > end {
        return (keys, off);
    }
    let kf_type = i32a(d, off);
    let count = i32a(d, off + 4);
    off += 8;
    let per = match kf_type {
        0 => 16,
        1 => 8,
        2 => 20,
        3 => 8,
        _ => return (keys, off),
    };
    if kf_type == 2 {
        let mut raws: Vec<(i32, f64, f64, f64, f64)> = Vec::new();
        for _ in 0..count {
            if off + 20 > end {
                break;
            }
            raws.push((
                i32a(d, off),
                f32a(d, off + 4) as f64,
                f32a(d, off + 8) as f64,
                f32a(d, off + 12) as f64,
                f32a(d, off + 16) as f64,
            ));
            off += 20;
        }
        let nn = raws.len();
        for i in 0..nn {
            let (t, v, _x1, x2, x3) = raws[i];
            let (mut ls, mut rs) = (0.0, 0.0);
            if nn > 1 {
                if i == 0 || i == nn - 1 {
                    let dd = if i == 0 {
                        raws[1].1 - v
                    } else {
                        v - raws[i - 1].1
                    };
                    let x1 = raws[i].2;
                    ls =
                        ((1.0 + x2) * (1.0 - x3) + (1.0 - x2) * (1.0 + x3)) * (1.0 - x1) * dd * 0.5;
                    rs =
                        ((1.0 - x2) * (1.0 - x3) + (1.0 + x2) * (1.0 + x3)) * (1.0 - x1) * dd * 0.5;
                } else {
                    let d01 = v - raws[i - 1].1;
                    let d12 = raws[i + 1].1 - v;
                    let dur01 = (t - raws[i - 1].0) as f64;
                    let dur12 = (raws[i + 1].0 - t) as f64;
                    let dur02 = dur01 + dur12;
                    let x1 = raws[i].2;
                    if dur02 > 0.0 {
                        ls = ((1.0 + x2) * (1.0 - x3) * d12 + (1.0 - x2) * (1.0 + x3) * d01)
                            * (1.0 - x1)
                            * (dur01 / dur02);
                        rs = ((1.0 - x2) * (1.0 - x3) * d12 + (1.0 + x2) * (1.0 + x3) * d01)
                            * (1.0 - x1)
                            * (dur12 / dur02);
                    }
                }
            }
            keys.push((t, v, ls, rs));
        }
        return (keys, off);
    }
    for _ in 0..count {
        if off + per > end {
            break;
        }
        let t = i32a(d, off);
        let v = f32a(d, off + 4) as f64;
        let (mut ls, mut rs) = (0.0, 0.0);
        if kf_type == 0 {
            ls = f32a(d, off + 8) as f64;
            rs = f32a(d, off + 12) as f64;
        }
        off += per;
        keys.push((t, v, ls, rs));
    }
    (keys, off)
}

fn eval_axis(kfs: &[AxisKey], time: f64) -> f64 {
    if kfs.is_empty() {
        return 0.0;
    }
    if kfs.len() == 1 || time <= kfs[0].0 as f64 {
        return kfs[0].1;
    }
    if time >= kfs[kfs.len() - 1].0 as f64 {
        return kfs[kfs.len() - 1].1;
    }
    let (mut lo, mut hi) = (0usize, kfs.len() - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        let mt = kfs[mid].0 as f64;
        if time < mt {
            hi = mid;
        } else if mt < time {
            lo = mid;
        } else {
            lo = mid;
            break;
        }
    }
    let cur = kfs[lo];
    let nxt = kfs[lo + 1];
    let dur = (nxt.0 - cur.0) as f64;
    if dur <= 0.0 {
        return cur.1;
    }
    let s = (time - cur.0 as f64) / dur;
    let delta = nxt.1 - cur.1;
    let a = (cur.3 + nxt.2) - 2.0 * delta;
    let b = 3.0 * delta - nxt.2 - 2.0 * cur.3;
    let c = cur.3;
    ((a * s + b) * s + c) * s + cur.1
}

fn read_axes(
    d: &[u8],
    rot_start: usize,
    rot_end: usize,
) -> (Vec<AxisKey>, Vec<AxisKey>, Vec<AxisKey>) {
    let off = rot_start + 20;
    let (x, o1) = read_axis(d, off, rot_end);
    let (y, o2) = read_axis(d, o1, rot_end);
    let (z, _) = read_axis(d, o2, rot_end);
    (x, y, z)
}

fn extract_rot_times(d: &[u8], rot_start: usize, rot_end: usize) -> Vec<i32> {
    if rot_end < rot_start + 28 {
        return Vec::new();
    }
    let (x, y, z) = read_axes(d, rot_start, rot_end);
    let mut set = std::collections::BTreeSet::new();
    for k in x.iter().chain(&y).chain(&z) {
        set.insert(k.0);
    }
    set.into_iter().collect()
}

fn read_rot_keyframes(d: &[u8], rot_start: usize, rot_end: usize, times: &[i32]) -> Vec<Vec3> {
    if rot_end < rot_start + 28 {
        return Vec::new();
    }
    let (x, y, z) = read_axes(d, rot_start, rot_end);
    if x.is_empty() && y.is_empty() && z.is_empty() {
        return Vec::new();
    }
    times
        .iter()
        .map(|&t| {
            [
                eval_axis(&x, t as f64) as f32,
                eval_axis(&y, t as f64) as f32,
                eval_axis(&z, t as f64) as f32,
            ]
        })
        .collect()
}

/// Euler (radians, XYZ) → quaternion (xyzw) with order q = qZ·qY·qX.
pub fn euler_to_quat(e: Vec3) -> [f32; 4] {
    let h = |a: f32| a * 0.5;
    let (sx, cx) = h(e[0]).sin_cos();
    let (sy, cy) = h(e[1]).sin_cos();
    let (sz, cz) = h(e[2]).sin_cos();
    // qz * qy * qx (each about its axis)
    // qx=(sx,0,0,cx) qy=(0,sy,0,cy) qz=(0,0,sz,cz)
    // qy*qx:
    let yx = [cy * sx, sy * cx, -sy * sx, cy * cx];
    // qz * (yx):  qz=(0,0,sz,cz)
    let x = cz * yx[0] - sz * yx[1];
    let y = cz * yx[1] + sz * yx[0];
    let z = sz * yx[3] + cz * yx[2];
    let w = cz * yx[3] - sz * yx[2];
    let l = (x * x + y * y + z * z + w * w).sqrt();
    if l > 1e-12 {
        [x / l, y / l, z / l, w / l]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}
