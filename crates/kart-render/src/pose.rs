//! CPU skinning: deform a character's body mesh by a skeletal animation pose.
//!
//! The body mesh's vertices are authored in bind-pose model space; each vertex
//! references up to two bones (`SkinWeight`). For a frame at time `t`:
//!   skin[i] = posed_world(i, t) · inverse(bind_world(i))
//!   p' = Σ wⱼ · skin[idxⱼ] · p
//! At the bind pose (or for un-animated bones) skin[i] = I, so vertices are
//! unchanged. Bind/posed world transforms come from the bone table's local
//! matrices composed up the parent hierarchy; the bind inverse is derived from
//! those same locals (matching how the C# path lets glTF auto-derive it), not
//! from the stored `inv_bind`.

use glam::{Mat3, Mat4, Quat, Vec3};
use kart_fmt::anim_clip::AnimClip;
use kart_fmt::raw_model::{RawBone, RawModel};

/// 3x4 row-major `[R|t]` (column-vector form, result = R·v + t) → glam Mat4.
fn bone_matrix(r: &[f32; 12]) -> Mat4 {
    Mat4::from_cols_array(&[
        r[0], r[4], r[8], 0.0, //
        r[1], r[5], r[9], 0.0, //
        r[2], r[6], r[10], 0.0, //
        r[3], r[7], r[11], 1.0, //
    ])
}

/// Compose each bone's world matrix from its `local` up the parent chain.
/// Parents always precede children, so one in-order pass suffices.
fn bind_worlds(bones: &[RawBone]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; bones.len()];
    for (i, b) in bones.iter().enumerate() {
        let local = bone_matrix(&b.local);
        world[i] = if b.parent >= 0 {
            world[b.parent as usize] * local
        } else {
            local
        };
    }
    world
}

/// Linear-interpolate a keyframe list at `t_ms` (clamped to the ends).
fn sample_vec3(keys: &[(i32, [f32; 3])], t_ms: i32) -> Option<Vec3> {
    if keys.is_empty() {
        return None;
    }
    if t_ms <= keys[0].0 {
        return Some(keys[0].1.into());
    }
    let last = keys.len() - 1;
    if t_ms >= keys[last].0 {
        return Some(keys[last].1.into());
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t_ms >= t0 && t_ms <= t1 {
            let f = (t_ms - t0) as f32 / (t1 - t0).max(1) as f32;
            return Some(Vec3::from(a).lerp(b.into(), f));
        }
    }
    Some(keys[last].1.into())
}

/// Slerp a quaternion keyframe list (stored xyzw) at `t_ms`.
fn sample_quat(keys: &[(i32, [f32; 4])], t_ms: i32) -> Option<Quat> {
    if keys.is_empty() {
        return None;
    }
    let q = |k: &(i32, [f32; 4])| Quat::from_xyzw(k.1[0], k.1[1], k.1[2], k.1[3]).normalize();
    if t_ms <= keys[0].0 {
        return Some(q(&keys[0]));
    }
    let last = keys.len() - 1;
    if t_ms >= keys[last].0 {
        return Some(q(&keys[last]));
    }
    for w in keys.windows(2) {
        if t_ms >= w[0].0 && t_ms <= w[1].0 {
            let f = (t_ms - w[0].0) as f32 / (w[1].0 - w[0].0).max(1) as f32;
            return Some(q(&w[0]).slerp(q(&w[1]), f));
        }
    }
    Some(q(&keys[last]))
}

/// Per-bone posed local: a bone with an animation track uses T·R·S sampled from
/// its keyframes (matching the glTF TRS convention); otherwise the bind local.
fn posed_locals(model: &RawModel, clip: &AnimClip, t_ms: i32) -> Vec<Mat4> {
    let mut locals: Vec<Mat4> = model.bones.iter().map(|b| bone_matrix(&b.local)).collect();
    for track in &clip.tracks {
        let i = track.bone_index;
        if i < 0 || i as usize >= locals.len() {
            continue;
        }
        let bind = locals[i as usize];
        // Fall back to the bind components for any channel the track omits.
        let (s, _, t) = bind.to_scale_rotation_translation();
        let pos = sample_vec3(&track.pos, t_ms).unwrap_or(t);
        let rot =
            sample_quat(&track.rot, t_ms).unwrap_or_else(|| bind.to_scale_rotation_translation().1);
        let scale = sample_vec3(&track.scale, t_ms).unwrap_or(s);
        locals[i as usize] = Mat4::from_scale_rotation_translation(scale, rot, pos);
    }
    locals
}

/// Skinning matrices for a frame: `posed_world · inverse(bind_world)` per bone.
fn skin_matrices(model: &RawModel, clip: &AnimClip, t_ms: i32) -> Vec<Mat4> {
    let bind = bind_worlds(&model.bones);
    let locals = posed_locals(model, clip, t_ms);
    let mut posed = vec![Mat4::IDENTITY; model.bones.len()];
    for (i, b) in model.bones.iter().enumerate() {
        posed[i] = if b.parent >= 0 {
            posed[b.parent as usize] * locals[i]
        } else {
            locals[i]
        };
    }
    bind.iter()
        .zip(&posed)
        .map(|(bw, pw)| *pw * bw.inverse())
        .collect()
}

/// Return a copy of `model` with its skinned body node's vertices and normals
/// deformed to the animation pose at `time_ms`. Other nodes are untouched.
/// Falls back to the original model if there is no skinned body or no bones.
pub fn pose_model(model: &RawModel, clip: &AnimClip, time_ms: i32) -> RawModel {
    if model.bones.is_empty() {
        return model.clone();
    }
    let skin = skin_matrices(model, clip, time_ms);
    let mut out = model.clone();
    let nbones = skin.len();
    for node in &mut out.nodes {
        if !node.is_body || node.bone_data.is_empty() {
            continue;
        }
        let weights = node.skin_weights();
        if weights.len() != node.vertices.len() {
            continue;
        }
        for (vi, sw) in weights.iter().enumerate() {
            let p = Vec3::from(node.vertices[vi]);
            let n = Vec3::from(node.normals[vi]);
            // Normalize the (up to two) blend weights.
            let i0 = sw.index0.clamp(0, nbones as i32 - 1) as usize;
            let (mut w0, mut w1) = (sw.weight0, sw.weight1);
            let m0 = skin[i0];
            let mat = if sw.index1 >= 0 && w1 > 0.0 {
                let s = (w0 + w1).max(1e-6);
                w0 /= s;
                w1 /= s;
                let i1 = sw.index1.clamp(0, nbones as i32 - 1) as usize;
                blend(m0, skin[i1], w0, w1)
            } else {
                m0
            };
            node.vertices[vi] = mat.transform_point3(p).into();
            let nm = Mat3::from_mat4(mat) * n;
            node.normals[vi] = nm.normalize_or_zero().into();
        }
        break; // single body node
    }
    out
}

fn blend(a: Mat4, b: Mat4, wa: f32, wb: f32) -> Mat4 {
    Mat4::from_cols(
        a.x_axis * wa + b.x_axis * wb,
        a.y_axis * wa + b.y_axis * wb,
        a.z_axis * wa + b.z_axis * wb,
        a.w_axis * wa + b.w_axis * wb,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kart_fmt::anim_clip::AnimClip;

    // A non-trivial 2-bone chain: child offset + rotated parent. With no
    // animation tracks the posed world equals the bind world, so every skin
    // matrix must collapse to identity — vertices come out unchanged.
    #[test]
    fn empty_clip_is_identity_skin() {
        let bones = vec![
            RawBone {
                index: 0,
                parent: -1,
                inv_bind: [0.0; 12],
                local: [0.0, -1.0, 0.0, 5.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, -3.0],
            },
            RawBone {
                index: 1,
                parent: 0,
                inv_bind: [0.0; 12],
                local: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 7.0, 0.0, 0.0, 1.0, 0.0],
            },
        ];
        let model = RawModel {
            bones,
            ..Default::default()
        };
        let skin = skin_matrices(&model, &AnimClip::default(), 0);
        for m in skin {
            let d = m - Mat4::IDENTITY;
            let max = d.to_cols_array().iter().fold(0f32, |a, b| a.max(b.abs()));
            assert!(max < 1e-5, "skin not identity: {max}");
        }
    }
}
