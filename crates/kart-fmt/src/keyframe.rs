//! Shared float keyframe curve — a clean-room port of the C# `FloatKeyframeData`
//! (`GetValue` + per-type `CalculateKeyFrame`). Used by the ThreeAxis euler
//! rotation in **both** the KartObject PRS controllers (`kart_object`) and the
//! boss skeletal anim clips (`anim_clip`). The kart rigid-body anim (`kart_anim`)
//! keeps its own variant — it additionally recovers CubicAlt tangent slopes and
//! is validated bit-exact against C#, so it is intentionally not merged here.

/// One rotation axis as a float keyframe list: `(time, value, leftSlope, rightSlope)`.
#[derive(Clone)]
pub struct FloatAxis {
    pub kf_type: i32,
    pub keys: Vec<(i32, f64, f64, f64)>,
}

fn i32a(d: &[u8], p: &mut usize) -> i32 {
    let v = i32::from_le_bytes(d[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn f32a(d: &[u8], p: &mut usize) -> f32 {
    let v = f32::from_le_bytes(d[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

/// Read one float axis from `d` at `*p` (advancing it): `i32 type, i32 count`,
/// then per keyframe by type — Cubic(0) 16B with left/right slopes, Linear(1) 8B,
/// CubicAlt(2) 20B (x1/x2/x3 skipped), NoEasing(3) 8B.
pub fn read_float_axis(d: &[u8], p: &mut usize) -> FloatAxis {
    let kf_type = i32a(d, p);
    let count = i32a(d, p);
    let mut keys = Vec::with_capacity(count.max(0) as usize);
    for _ in 0..count {
        let t = i32a(d, p);
        let v = f32a(d, p) as f64;
        let (ls, rs) = match kf_type {
            0 => (f32a(d, p) as f64, f32a(d, p) as f64), // Cubic: leftSlope/rightSlope
            2 => {
                *p += 12; // CubicAlt x1/x2/x3 (slope recovery not needed here)
                (0.0, 0.0)
            }
            _ => (0.0, 0.0), // Linear / NoEasing
        };
        keys.push((t, v, ls, rs));
    }
    FloatAxis { kf_type, keys }
}

/// Evaluate the axis at `time`, replicating C# `FloatKeyframeData.GetValue`: the
/// binary-search-with-quirk segment lookup, then per-type `CalculateKeyFrame`
/// (Linear lerp / NoEasing step / Cubic & CubicAlt cubic-Hermite).
pub fn eval_float_axis(ax: &FloatAxis, time: f64) -> f64 {
    let n = ax.keys.len();
    if n == 0 {
        return 0.0;
    }
    let (mut start, mut end) = (0i32, n as i32 - 1);
    while (start - end).abs() > 1 {
        let mid = (start + end) >> 1;
        let mt = ax.keys[mid as usize].0;
        if (time as i64) < mt as i64 {
            end = mid;
        } else if mid < mt {
            start = mid;
        } else {
            while start + 1 < end && ax.keys[(start + 1) as usize].0 == ax.keys[start as usize].0 {
                start += 1;
            }
            break;
        }
    }
    let (s, e) = (start as usize, end as usize);
    if (ax.keys[e].0 as f64) < time {
        return ax.keys[e].1;
    }
    if (ax.keys[s].0 as f64) > time {
        return ax.keys[s].1;
    }
    let cur = ax.keys[s];
    let next = if s + 1 >= n {
        None
    } else {
        Some(ax.keys[s + 1])
    };
    let dur = (next.map(|k| k.0).unwrap_or(cur.0) - cur.0) as f64;
    let t = (time - cur.0 as f64) / if dur == 0.0 { 1.0 } else { dur };
    match ax.kf_type {
        1 => cur.1 * (1.0 - t) + next.map(|k| k.1).unwrap_or(cur.1) * t, // Linear
        3 => cur.1,                                                      // NoEasing (step)
        _ => {
            let nx = next.unwrap_or(cur); // Cubic / CubicAlt cubic-Hermite
            let delta = nx.1 - cur.1;
            let a = cur.3 + nx.2 - 2.0 * delta;
            let b = 3.0 * delta - nx.2 - 2.0 * cur.3;
            let c = cur.3;
            ((a * t + b) * t + c) * t + cur.1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ax(kf_type: i32, keys: &[(i32, f64, f64, f64)]) -> FloatAxis {
        FloatAxis {
            kf_type,
            keys: keys.to_vec(),
        }
    }
    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn linear_lerp_and_clamp() {
        let a = ax(1, &[(0, 0.0, 0.0, 0.0), (100, 10.0, 0.0, 0.0)]);
        approx(eval_float_axis(&a, -5.0), 0.0); // clamp before
        approx(eval_float_axis(&a, 0.0), 0.0);
        approx(eval_float_axis(&a, 50.0), 5.0); // midpoint lerp
        approx(eval_float_axis(&a, 100.0), 10.0);
        approx(eval_float_axis(&a, 200.0), 10.0); // clamp after
    }

    #[test]
    fn noeasing_is_step() {
        let a = ax(3, &[(0, 0.0, 0.0, 0.0), (100, 10.0, 0.0, 0.0)]);
        approx(eval_float_axis(&a, 50.0), 0.0); // holds the segment's left value
                                                // At exactly the last key, C# still interpolates the final segment, so a
                                                // step curve returns the *left* value; only strictly past the end clamps.
        approx(eval_float_axis(&a, 100.0), 0.0);
        approx(eval_float_axis(&a, 150.0), 10.0);
    }

    #[test]
    fn cubic_hermite_with_slopes() {
        // tangents force a non-linear curve: zero-slope cubic is the smoothstep
        // shape, and at the midpoint of a 0→10 segment that still passes 5.0,
        // but a left/right slope bends it away from the linear value.
        let flat = ax(0, &[(0, 0.0, 0.0, 0.0), (100, 10.0, 0.0, 0.0)]);
        approx(eval_float_axis(&flat, 50.0), 5.0);
        let sloped = ax(0, &[(0, 0.0, 0.0, 10.0), (100, 10.0, 0.0, 0.0)]);
        // a=rs+next.ls-2δ=10+0-20=-10, b=3δ-next.ls-2rs=30-0-20=10, c=rs=10
        // f(0.5)=((-10*.5+10)*.5+10)*.5+0 = ((5)*.5+10)*.5 = (2.5+10)*.5 = 6.25
        approx(eval_float_axis(&sloped, 50.0), 6.25);
        approx(eval_float_axis(&sloped, 0.0), 0.0);
        approx(eval_float_axis(&sloped, 100.0), 10.0);
    }

    #[test]
    fn empty_and_single() {
        approx(eval_float_axis(&ax(0, &[]), 5.0), 0.0);
        approx(eval_float_axis(&ax(0, &[(10, 3.0, 0.0, 0.0)]), 999.0), 3.0);
    }
}
