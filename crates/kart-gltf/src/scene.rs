//! Scene conversion: KartRider parsed assets → a `Gltf` builder, mirroring the
//! C# `SceneConverter` so the exported `.glb` matches. E1 covers raw-magic
//! models (character / kart) as static geometry; skins + animation arrive in
//! later milestones.

use crate::{Alpha, AnimChannel, Gltf, MaterialDef, NodeDef, Primitive};
use glam::Mat4;
use kart_fmt::anim_clip::AnimClip;
use kart_fmt::kart_anim::{self, KartAnimClip};
use kart_fmt::kart_object::{AlphaInfo, Track};
use kart_fmt::raw_model::{RawBone, RawModel, RawNode};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Z-up(LH) → Y-up(RH) axis swap (and unit scale), as a glTF column-major
/// matrix. Matches the C# root `swap` (determinant −1 ⇒ winding is reversed
/// when emitting triangles).
fn swap_matrix(scale: f32) -> [f32; 16] {
    [
        scale, 0.0, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// A raw node's local transform (engine row-vector `S·R·T`) as a glTF
/// column-major matrix: column i = scaleᵢ · rotation-row-i, translation last col.
fn node_matrix(n: &RawNode) -> [f32; 16] {
    let r = n
        .rotation
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let [sx, sy, sz] = n.scale;
    let p = n.position;
    [
        sx * r[0],
        sx * r[1],
        sx * r[2],
        0.0,
        sy * r[3],
        sy * r[4],
        sy * r[5],
        0.0,
        sz * r[6],
        sz * r[7],
        sz * r[8],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    ]
}

/// glTF column-major bone local matrix from the bone table's 3x4 row-major
/// `[R|t]` (same convention as the C# `BoneMatrix`).
fn bone_matrix(r: &[f32; 12]) -> [f32; 16] {
    [
        r[0], r[4], r[8], 0.0, r[1], r[5], r[9], 0.0, r[2], r[6], r[10], 0.0, r[3], r[7], r[11],
        1.0,
    ]
}

fn mat(a: &[f32; 16]) -> Mat4 {
    Mat4::from_cols_array(a)
}

/// Convert a raw model to a glTF builder. `tex_dir` holds the sibling
/// `<meshIndex>.png` textures. When the model has a bone table, the body mesh is
/// exported as a skinned mesh (joints + inverse-bind) and `anims` (character
/// skeletal clips) become glTF TRS animations on the joints.
pub fn convert_raw(
    model: &RawModel,
    name: &str,
    tex_dir: &Path,
    anims: &[AnimClip],
    kart_anims: &[KartAnimClip],
) -> Gltf {
    let mut g = Gltf::new();
    let mut mat_cache: HashMap<i32, usize> = HashMap::new();
    let swap = swap_matrix(1.0);

    if model.bones.is_empty() {
        // no skeleton → every node is a rigid mesh (E1 path). Record anim entity
        // ids → glTF node so kart rigid-body clips can target the right nodes.
        let mut anim_map: HashMap<i32, usize> = HashMap::new();
        let top: Vec<usize> = model
            .nodes
            .iter()
            .map(|n| walk_raw_anim(&mut g, n, tex_dir, &mut mat_cache, &mut anim_map))
            .collect();
        let root = g.add_node(NodeDef {
            name: Some(name.to_string()),
            matrix: Some(swap),
            children: top,
            ..Default::default()
        });
        g.set_scene_roots(&[root]);
        add_kart_animations(&mut g, kart_anims, &anim_map);
        return g;
    }

    // --- joint skeleton: one node per bone, wired parent→child afterwards ---
    let bones = &model.bones;
    let max_idx = bones.iter().map(|b| b.index).max().unwrap_or(0);
    let mut joint_node = vec![usize::MAX; max_idx + 1];
    for b in bones {
        let n = g.add_node(NodeDef {
            name: Some(format!("bone{}", b.index)),
            matrix: Some(bone_matrix(&b.local)),
            ..Default::default()
        });
        joint_node[b.index] = n;
    }
    for b in bones {
        let kids: Vec<usize> = bones
            .iter()
            .filter(|c| c.parent == b.index as i32)
            .map(|c| joint_node[c.index])
            .collect();
        g.set_children(joint_node[b.index], &kids);
    }
    let top_joints: Vec<usize> = bones
        .iter()
        .filter(|b| b.parent < 0)
        .map(|b| joint_node[b.index])
        .collect();

    // bind-pose joint world transforms (under the swap root) → inverse-bind
    let world_bind = bind_worlds(bones, &mat(&swap));
    let joints_array: Vec<usize> = (0..=max_idx)
        .map(|i| {
            if joint_node[i] != usize::MAX {
                joint_node[i]
            } else {
                usize::MAX
            }
        })
        .collect();
    let inverse_bind: Vec<[f32; 16]> = (0..=max_idx)
        .map(|i| world_bind[i].inverse().to_cols_array())
        .collect();

    // --- meshes: skinned body + rigid extras ---
    // Rigid (non-body) meshes hang under the swap root; the skinned body is a
    // *separate* scene root at identity with the swap baked into its vertices
    // (skinning cancels the swap-included joint/inverse-bind at the bind pose) —
    // exactly the C# AddSkinnedMesh-at-root placement, so it isn't swapped twice.
    let mut root_children = top_joints.clone();
    let mut body_node = None;
    for n in &model.nodes {
        if n.is_body && !n.bone_data.is_empty() {
            let bake = mat(&swap) * mat(&node_matrix(n));
            if let Some(mesh) = build_skinned_body(&mut g, n, tex_dir, &mut mat_cache, &bake) {
                body_node = Some(g.add_node(NodeDef {
                    name: Some(n.name.clone()),
                    mesh: Some(mesh),
                    ..Default::default()
                }));
            }
        } else {
            root_children.push(walk_raw(&mut g, n, tex_dir, &mut mat_cache));
        }
    }

    let root = g.add_node(NodeDef {
        name: Some(name.to_string()),
        matrix: Some(swap),
        children: root_children,
        ..Default::default()
    });
    let skin = g.add_skin(
        &joints_array
            .iter()
            .map(|&j| if j == usize::MAX { root } else { j })
            .collect::<Vec<_>>(),
        &inverse_bind,
        Some(root),
    );
    let mut roots = vec![root];
    if let Some(bn) = body_node {
        g.set_node_skin(bn, skin);
        roots.push(bn);
    }
    g.set_scene_roots(&roots);

    // --- animations: one glTF animation per clip, TRS tracks on joint nodes ---
    for clip in anims {
        let mut channels = Vec::new();
        for track in &clip.tracks {
            let bi = track.bone_index;
            if bi < 0 || bi as usize > max_idx || joint_node[bi as usize] == usize::MAX {
                continue;
            }
            let node = joint_node[bi as usize];
            if !track.pos.is_empty() {
                channels.push(AnimChannel {
                    node,
                    path: "translation",
                    times: track.pos.iter().map(|(t, _)| *t as f32 / 1000.0).collect(),
                    values: track.pos.iter().flat_map(|(_, v)| *v).collect(),
                });
            }
            if !track.rot.is_empty() {
                channels.push(AnimChannel {
                    node,
                    path: "rotation",
                    times: track.rot.iter().map(|(t, _)| *t as f32 / 1000.0).collect(),
                    values: track
                        .rot
                        .iter()
                        .flat_map(|(_, q)| normalize_quat(*q))
                        .collect(),
                });
            }
            if !track.scale.is_empty() {
                channels.push(AnimChannel {
                    node,
                    path: "scale",
                    times: track
                        .scale
                        .iter()
                        .map(|(t, _)| *t as f32 / 1000.0)
                        .collect(),
                    values: track.scale.iter().flat_map(|(_, v)| *v).collect(),
                });
            }
        }
        g.add_animation(&clip.name, channels);
    }
    g
}

/// Compose each bone's bind world matrix from `swap · (local up the parent chain)`.
fn bind_worlds(bones: &[RawBone], swap: &Mat4) -> Vec<Mat4> {
    let max_idx = bones.iter().map(|b| b.index).max().unwrap_or(0);
    let mut world = vec![*swap; max_idx + 1];
    for b in bones {
        let local = mat(&bone_matrix(&b.local));
        world[b.index] = if b.parent >= 0 {
            world[b.parent as usize] * local
        } else {
            *swap * local
        };
    }
    world
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 1e-12 {
        [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Body mesh exported skinned: vertices baked by `bake` (= swap · body local),
/// with JOINTS_0/WEIGHTS_0 from the per-vertex skin records. Winding reversed
/// (bake has det < 0).
fn build_skinned_body(
    g: &mut Gltf,
    n: &RawNode,
    tex_dir: &Path,
    cache: &mut HashMap<i32, usize>,
    bake: &Mat4,
) -> Option<usize> {
    if n.vertices.is_empty() || n.vertex_indices.is_empty() {
        return None;
    }
    let weights = n.skin_weights();
    if weights.len() != n.vertices.len() {
        return None;
    }
    let has_n = n.normals.len() == n.vertices.len();
    let mut prim = Primitive {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        joints: Some(Vec::new()),
        weights: Some(Vec::new()),
        material: Some(resolve_raw_material(g, n.mesh_index, tex_dir, cache)),
    };
    let nmat = glam::Mat3::from_mat4(*bake);
    let vi = &n.vertex_indices;
    let ui = &n.uv_indices;
    for f in 0..vi.len() / 3 {
        let idx = [
            vi[f * 3] as usize,
            vi[f * 3 + 1] as usize,
            vi[f * 3 + 2] as usize,
        ];
        if idx.iter().any(|&v| v >= n.vertices.len()) {
            continue;
        }
        let baked: Vec<glam::Vec3> = idx
            .iter()
            .map(|&v| bake.transform_point3(glam::Vec3::from(n.vertices[v])))
            .collect();
        let baked_face = glam::Vec3::from(face_normal(
            baked[0].into(),
            baked[2].into(),
            baked[1].into(),
        ));
        for &k in &[0usize, 2, 1] {
            let v = idx[k];
            let p = baked[k];
            let nn = if has_n {
                let sn = (nmat * glam::Vec3::from(n.normals[v])).normalize_or_zero();
                if sn == glam::Vec3::ZERO {
                    baked_face
                } else {
                    sn
                }
            } else {
                baked_face
            };
            let base = prim.positions.len() as u32;
            prim.positions.push(p.into());
            prim.normals.push(nn.into());
            prim.uvs.push(uv_at(n, ui, f * 3 + k));
            let sw = weights[v];
            let (i0, i1) = (sw.index0.max(0) as u16, sw.index1.max(0) as u16);
            let (mut w0, mut w1) = (sw.weight0, sw.weight1);
            if sw.index1 >= 0 && w1 > 0.0 {
                let s = (w0 + w1).max(1e-6);
                w0 /= s;
                w1 /= s;
                prim.joints.as_mut().unwrap().push([i0, i1, 0, 0]);
                prim.weights.as_mut().unwrap().push([w0, w1, 0.0, 0.0]);
            } else {
                prim.joints.as_mut().unwrap().push([i0, 0, 0, 0]);
                prim.weights.as_mut().unwrap().push([1.0, 0.0, 0.0, 0.0]);
            }
            prim.indices.push(base);
        }
    }
    if prim.indices.is_empty() {
        return None;
    }
    Some(g.add_mesh(if n.name.is_empty() { "body" } else { &n.name }, vec![prim]))
}

fn walk_raw(
    g: &mut Gltf,
    n: &RawNode,
    tex_dir: &Path,
    mat_cache: &mut HashMap<i32, usize>,
) -> usize {
    let children: Vec<usize> = n
        .children
        .iter()
        .map(|c| walk_raw(g, c, tex_dir, mat_cache))
        .collect();
    let mesh = build_raw_mesh(g, n, tex_dir, mat_cache);
    let name = if n.name.is_empty() {
        format!("node{}", n.node_id)
    } else {
        n.name.clone()
    };
    g.add_node(NodeDef {
        name: Some(name),
        matrix: Some(sanitize(node_matrix(n))),
        mesh,
        children,
        ..Default::default()
    })
}

/// Like `walk_raw`, but records each node's anim entity ids → glTF node index.
fn walk_raw_anim(
    g: &mut Gltf,
    n: &RawNode,
    tex_dir: &Path,
    mat_cache: &mut HashMap<i32, usize>,
    anim_map: &mut HashMap<i32, usize>,
) -> usize {
    let children: Vec<usize> = n
        .children
        .iter()
        .map(|c| walk_raw_anim(g, c, tex_dir, mat_cache, anim_map))
        .collect();
    let mesh = build_raw_mesh(g, n, tex_dir, mat_cache);
    let name = if n.name.is_empty() {
        format!("node{}", n.node_id)
    } else {
        n.name.clone()
    };
    let node = g.add_node(NodeDef {
        name: Some(name),
        matrix: Some(sanitize(node_matrix(n))),
        mesh,
        children,
        ..Default::default()
    });
    for &id in &n.anim_entity_ids {
        anim_map.insert(id, node);
    }
    node
}

/// Apply kart rigid-body clips: each animated channel maps to a node via its
/// anim_id, and becomes glTF TRS tracks (euler → quaternion for rotation).
fn add_kart_animations(g: &mut Gltf, clips: &[KartAnimClip], anim_map: &HashMap<i32, usize>) {
    for clip in clips {
        let mut channels = Vec::new();
        for ch in &clip.channels {
            if !ch.is_animated() {
                continue;
            }
            let Some(&node) = anim_map.get(&ch.anim_id) else {
                continue;
            };
            let times: Vec<f32> = ch
                .keyframes
                .iter()
                .map(|k| k.time_ms as f32 / 1000.0)
                .collect();
            channels.push(AnimChannel {
                node,
                path: "translation",
                times: times.clone(),
                values: ch.keyframes.iter().flat_map(|k| k.pos).collect(),
            });
            channels.push(AnimChannel {
                node,
                path: "scale",
                times: times.clone(),
                values: ch.keyframes.iter().flat_map(|k| k.scale).collect(),
            });
            channels.push(AnimChannel {
                node,
                path: "rotation",
                times,
                values: ch
                    .keyframes
                    .iter()
                    .flat_map(|k| kart_anim::euler_to_quat(k.rot_euler))
                    .collect(),
            });
        }
        g.add_animation(&clip.name, channels);
    }
}

fn build_raw_mesh(
    g: &mut Gltf,
    n: &RawNode,
    tex_dir: &Path,
    mat_cache: &mut HashMap<i32, usize>,
) -> Option<usize> {
    if !n.is_mesh || n.vertices.is_empty() || n.vertex_indices.is_empty() {
        return None;
    }
    let has_n = n.normals.len() == n.vertices.len();
    let mut prim = Primitive {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        joints: None,
        weights: None,
        material: Some(resolve_raw_material(g, n.mesh_index, tex_dir, mat_cache)),
    };
    let vi = &n.vertex_indices;
    let ui = &n.uv_indices;
    let nf = vi.len() / 3;
    for f in 0..nf {
        let (a, b, c) = (
            vi[f * 3] as usize,
            vi[f * 3 + 1] as usize,
            vi[f * 3 + 2] as usize,
        );
        if a >= n.vertices.len() || b >= n.vertices.len() || c >= n.vertices.len() {
            continue;
        }
        let p = [n.vertices[a], n.vertices[b], n.vertices[c]];
        let nrm = {
            // Source winding is left-handed (.1s is Z-up LH); the right-handed
            // cross product must use the emitted a,c,b order or every fallback
            // normal points into the surface (observed: factory_I02 floor lit
            // only by the hemisphere ground color).
            let fnrm = face_normal(p[0], p[2], p[1]);
            if has_n {
                [
                    source_or_face(n.normals[a], fnrm),
                    source_or_face(n.normals[b], fnrm),
                    source_or_face(n.normals[c], fnrm),
                ]
            } else {
                [fnrm; 3]
            }
        };
        let uv = [
            uv_at(n, ui, f * 3),
            uv_at(n, ui, f * 3 + 1),
            uv_at(n, ui, f * 3 + 2),
        ];
        // natural order a,b,c reversed to a,c,b to compensate det(−1) swap root
        for &k in &[0usize, 2, 1] {
            let base = prim.positions.len() as u32;
            prim.positions.push(p[k]);
            prim.normals.push(nrm[k]);
            prim.uvs.push(uv[k]);
            prim.indices.push(base);
        }
    }
    if prim.indices.is_empty() {
        return None;
    }
    Some(g.add_mesh(if n.name.is_empty() { "mesh" } else { &n.name }, vec![prim]))
}

fn uv_at(n: &RawNode, ui: &[i32], k: usize) -> [f32; 2] {
    if k >= ui.len() {
        return [0.0, 0.0];
    }
    let i = ui[k];
    if i < 0 || i as usize >= n.uvs.len() {
        return [0.0, 0.0];
    }
    n.uvs[i as usize] // already v-flipped by the parser
}

fn resolve_raw_material(
    g: &mut Gltf,
    mesh_index: i32,
    tex_dir: &Path,
    cache: &mut HashMap<i32, usize>,
) -> usize {
    if let Some(&m) = cache.get(&mesh_index) {
        return m;
    }
    let image = if mesh_index >= 0 {
        std::fs::read(tex_dir.join(format!("{mesh_index}.png")))
            .ok()
            .map(|png| g.add_png(&png))
    } else {
        None
    };
    let m = g.add_material(MaterialDef {
        name: format!("mat{mesh_index}"),
        base_color_image: image,
        base_color: [1.0, 1.0, 1.0, 1.0],
        double_sided: true, // C# default (Cull off)
        alpha: Alpha::Opaque,
        unlit: false,
        extras: None,
    });
    cache.insert(mesh_index, m);
    m
}

/// Mirror of the C# `Sanitize`: glTF node matrices must be finite and cleanly
/// decomposable into translation·rotation·scale. Engine nodes often carry
/// *sheared* bases (mutually non-orthogonal columns) that System.Numerics
/// `Matrix4x4.Decompose` rejects — C# then keeps translation only. We reproduce
/// that: non-finite → identity; non-orthonormal/degenerate basis → translation
/// only; otherwise the matrix is kept.
fn sanitize(m: [f32; 16]) -> [f32; 16] {
    if !m.iter().all(|v| v.is_finite()) {
        return identity4();
    }
    // Replicates System.Numerics Matrix4x4.Decompose's success test: normalize
    // the three basis columns, then require the determinant of that normalized
    // basis to be ≈ ±1 ((det−1)² ≤ DecomposeEpsilon after a handedness flip).
    // Sheared (non-orthonormal) bases fail and fall back to translation-only.
    let cols = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let len = |c: &[f32; 3]| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let l = [len(&cols[0]), len(&cols[1]), len(&cols[2])];
    let decomposable = if l.iter().any(|&x| x <= 1e-7) {
        false
    } else {
        let n: Vec<[f32; 3]> = (0..3)
            .map(|i| [cols[i][0] / l[i], cols[i][1] / l[i], cols[i][2] / l[i]])
            .collect();
        // det of the normalized basis (n0 · (n1 × n2))
        let cross = [
            n[1][1] * n[2][2] - n[1][2] * n[2][1],
            n[1][2] * n[2][0] - n[1][0] * n[2][2],
            n[1][0] * n[2][1] - n[1][1] * n[2][0],
        ];
        let mut det = n[0][0] * cross[0] + n[0][1] * cross[1] + n[0][2] * cross[2];
        if det < 0.0 {
            det = -det; // .NET flips one axis for a left-handed basis
        }
        let d = det - 1.0;
        d * d <= 1e-4
    };
    if decomposable {
        m
    } else {
        let t = [m[12], m[13], m[14]];
        let t = if t.iter().all(|v| v.is_finite()) {
            t
        } else {
            [0.0; 3]
        };
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t[0], t[1], t[2], 1.0,
        ]
    }
}

fn identity4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

// ===== track (KartObject) export =====================================

/// A track node's local transform (engine `S·R·T`, transform columns = the 3
/// stored vectors) as a glTF column-major matrix.
fn track_node_matrix(t: &[f32; 9], pos: [f32; 3], scale: [f32; 3]) -> [f32; 16] {
    let [sx, sy, sz] = scale;
    [
        sx * t[0],
        sx * t[3],
        sx * t[6],
        0.0,
        sy * t[1],
        sy * t[4],
        sy * t[7],
        0.0,
        sz * t[2],
        sz * t[5],
        sz * t[8],
        0.0,
        pos[0],
        pos[1],
        pos[2],
        1.0,
    ]
}

/// Shared state for a track conversion (avoids threading many params).
struct TrackCtx<'a> {
    g: Gltf,
    track: &'a Track,
    tex_dirs: &'a [PathBuf],
    tex_png: HashMap<String, Option<usize>>,
    mat_cache: HashMap<String, usize>,
    node_map: HashMap<usize, usize>, // track node idx → glTF node idx
    anims: bool,
}

/// Export a parsed KartObject track/stage to glTF: scene tree under the swap
/// root, per-node meshes (reversed winding) and materials (DDS/PNG texture
/// resolved by searching `tex_dirs` in order, alpha mask/blend, double-sided by
/// default — matching C#'s repeatable `--tex-dir`). With `anims`, the per-node
/// PRSTontroller keyframes become a single "track" animation and TexProperty UV
/// curves go into `material.extras.kartUvAnim`.
pub fn convert_track(track: &Track, name: &str, tex_dirs: &[PathBuf], anims: bool) -> Gltf {
    let mut c = TrackCtx {
        g: Gltf::new(),
        track,
        tex_dirs,
        tex_png: HashMap::new(),
        mat_cache: HashMap::new(),
        node_map: HashMap::new(),
        anims,
    };

    // The scene-root Relement's own transform is REPLACED by the swap (not
    // composed) — exactly as C# Convert puts the scene on the swap rootNode.
    let (mesh, children) = match track.root {
        Some(r) => {
            let kids: Vec<usize> = track.nodes[r]
                .children
                .iter()
                .map(|&ch| walk_track(&mut c, ch))
                .collect();
            (build_track_mesh(&mut c, &track.nodes[r]), kids)
        }
        None => (None, Vec::new()),
    };
    let root = c.g.add_node(NodeDef {
        name: Some(name.to_string()),
        matrix: Some(swap_matrix(1.0)),
        mesh,
        children,
        ..Default::default()
    });
    if let Some(r) = track.root {
        c.node_map.insert(r, root);
    }
    c.g.set_scene_roots(&[root]);

    if anims {
        add_track_prs(&mut c);
    }
    c.g
}

/// An event-subtree root's authored local transform as a glb-space (post
/// swap) column-major matrix — the placement matrix for instancing the
/// normalized subtree (`convert_track_events`) back at its authored spot.
/// Composed as `S · track_node_matrix(...)` exactly like the scene export's
/// root composition, so the instance lands where the track scene would have
/// put it.
pub fn track_event_placement_glb(
    transform: &[f32; 9],
    position: [f32; 3],
    scale: [f32; 3],
) -> [f32; 16] {
    let m = mat(&swap_matrix(1.0)) * mat(&track_node_matrix(transform, position, scale));
    m.to_cols_array()
}

/// Export ONLY the To* event subtrees of a track (E2/E3 of
/// docs/kart-fmt-to-events-handoff-2026-07-30.md): one top-level node per
/// Relement-family event root, with the root's own transform NORMALIZED to
/// identity — the glb is an asset library; placements live in
/// track_events.json (`track_event_placement_glb`) and the page instances
/// each subtree at its matrix. Node names are the authored '#mo_*' names so
/// placements can find their subtree. Materials/UV curves go through the
/// same resolve_track_material path as the scene export.
pub fn convert_track_events(track: &Track, name: &str, tex_dirs: &[PathBuf], anims: bool) -> Gltf {
    let mut c = TrackCtx {
        g: Gltf::new(),
        track,
        tex_dirs,
        tex_png: HashMap::new(),
        mat_cache: HashMap::new(),
        node_map: HashMap::new(),
        anims,
    };

    let mut top = Vec::new();
    for root in track.runtime_event_roots() {
        let Some(i) = root.node else { continue };
        let n = &track.nodes[i];
        let kids: Vec<usize> = n.children.clone();
        let children: Vec<usize> = kids.iter().map(|&ch| walk_track(&mut c, ch)).collect();
        let n = &track.nodes[i];
        let mesh = build_track_mesh(&mut c, n);
        let node = c.g.add_node(NodeDef {
            name: Some(if n.name.is_empty() {
                format!("event{}", root.index)
            } else {
                n.name.clone()
            }),
            matrix: Some(identity4()), // normalized; placement comes from the JSON
            mesh,
            children,
            ..Default::default()
        });
        c.node_map.insert(i, node);
        top.push(node);
    }
    let root = c.g.add_node(NodeDef {
        name: Some(name.to_string()),
        matrix: Some(swap_matrix(1.0)),
        children: top,
        ..Default::default()
    });
    c.g.set_scene_roots(&[root]);

    if anims {
        add_track_prs(&mut c);
    }
    c.g
}

/// Emit the single "track" animation from every node's PRSTontroller keyframes.
fn add_track_prs(c: &mut TrackCtx) {
    let mut channels = Vec::new();
    let finite3 = |v: &[f32; 3]| v.iter().all(|x| x.is_finite());
    let finite4 = |v: &[f32; 4]| v.iter().all(|x| x.is_finite());
    for (&ti, &node) in &c.node_map {
        let Some(prs) = &c.track.nodes[ti].prs else {
            continue;
        };
        if !prs.has_anim() {
            continue;
        }
        let t = |k: i32| k as f32 / 1000.0;
        let pos: Vec<_> = prs.pos.iter().filter(|(_, v)| finite3(v)).collect();
        if !pos.is_empty() {
            channels.push(AnimChannel {
                node,
                path: "translation",
                times: pos.iter().map(|(k, _)| t(*k)).collect(),
                values: pos.iter().flat_map(|(_, v)| *v).collect(),
            });
        }
        let scale: Vec<_> = prs.scale.iter().filter(|(_, v)| finite3(v)).collect();
        if !scale.is_empty() {
            channels.push(AnimChannel {
                node,
                path: "scale",
                times: scale.iter().map(|(k, _)| t(*k)).collect(),
                values: scale.iter().flat_map(|(_, v)| *v).collect(),
            });
        }
        let rot: Vec<_> = prs.rot.iter().filter(|(_, q)| finite4(q)).collect();
        if !rot.is_empty() {
            channels.push(AnimChannel {
                node,
                path: "rotation",
                times: rot.iter().map(|(k, _)| t(*k)).collect(),
                values: rot.iter().flat_map(|(_, q)| *q).collect(),
            });
        }
    }
    c.g.add_animation("track", channels);
}

fn walk_track(c: &mut TrackCtx, i: usize) -> usize {
    let n = &c.track.nodes[i];
    let kids: Vec<usize> = n.children.clone();
    let children: Vec<usize> = kids.iter().map(|&ch| walk_track(c, ch)).collect();
    let n = &c.track.nodes[i];
    let mesh = build_track_mesh(c, n);
    let n = &c.track.nodes[i];
    let name = if n.name.is_empty() {
        "node".to_string()
    } else {
        n.name.clone()
    };
    let mat = sanitize(track_node_matrix(&n.transform, n.position, n.scale));
    let node = c.g.add_node(NodeDef {
        name: Some(name),
        matrix: Some(mat),
        mesh,
        children,
        ..Default::default()
    });
    c.node_map.insert(i, node);
    node
}

fn build_track_mesh(c: &mut TrackCtx, n: &kart_fmt::kart_object::Node) -> Option<usize> {
    let geom = n.geom.as_ref()?;
    if geom.vertices.is_empty() || geom.indices.is_empty() {
        return None;
    }
    let (material, discards_all) =
        resolve_track_material(c, n.tex.as_deref(), n.alpha, &n.uv_curves);
    if discards_all {
        // Original game discards every texel of this surface (MASK with
        // alpha_ref 0 over a fully transparent texture, e.g. the authored
        // "transparency" marker), so it is invisible; emitting it would
        // produce an opaque white shell because glTF alphaTest is
        // `alpha >= cutoff` and cannot express the discard-all case.
        return None;
    }
    let has_n = geom.normals.len() == geom.vertices.len();
    // expand index buffer to triangles (strip → list), then reverse winding
    let tris: Vec<[u32; 3]> = if geom.strip {
        (0..geom.indices.len().saturating_sub(2))
            .filter_map(|k| {
                let (a, b, c) = (geom.indices[k], geom.indices[k + 1], geom.indices[k + 2]);
                if a == b || b == c || a == c {
                    None
                } else if k % 2 == 0 {
                    Some([a, b, c])
                } else {
                    Some([b, a, c])
                }
            })
            .collect()
    } else {
        geom.indices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    };
    let mut prim = Primitive {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        joints: None,
        weights: None,
        material: Some(material),
    };
    for tri in &tris {
        if tri.iter().any(|&v| v as usize >= geom.vertices.len()) {
            continue;
        }
        let p = [
            geom.vertices[tri[0] as usize],
            geom.vertices[tri[1] as usize],
            geom.vertices[tri[2] as usize],
        ];
        let nrm = {
            // LH source winding: cross in the emitted a,c,b order (see
            // build_raw_mesh).
            let fnrm = face_normal(p[0], p[2], p[1]);
            if has_n {
                [
                    source_or_face(geom.normals[tri[0] as usize], fnrm),
                    source_or_face(geom.normals[tri[1] as usize], fnrm),
                    source_or_face(geom.normals[tri[2] as usize], fnrm),
                ]
            } else {
                [fnrm; 3]
            }
        };
        let uvc = [
            geom.uvs.get(tri[0] as usize).copied().unwrap_or([0.0, 0.0]),
            geom.uvs.get(tri[1] as usize).copied().unwrap_or([0.0, 0.0]),
            geom.uvs.get(tri[2] as usize).copied().unwrap_or([0.0, 0.0]),
        ];
        // SharpGLTF's MeshBuilder drops a triangle when two corners share the
        // same POSITION (regardless of normal/uv) — empirically verified. A
        // small-but-valid effect triangle (distinct positions) is kept.
        if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
            continue;
        }
        // natural a,b,c reversed to a,c,b (det −1 swap root)
        for &k in &[0usize, 2, 1] {
            let base = prim.positions.len() as u32;
            prim.positions.push(p[k]);
            prim.normals.push(nrm[k]);
            prim.uvs.push(uvc[k]);
            prim.indices.push(base);
        }
    }
    if prim.indices.is_empty() {
        return None;
    }
    let label = if n.name.is_empty() {
        "mesh".to_string()
    } else {
        n.name.clone()
    };
    Some(c.g.add_mesh(&label, vec![prim]))
}

fn resolve_track_material(
    c: &mut TrackCtx,
    tex: Option<&str>,
    alpha: Option<AlphaInfo>,
    uv_curves: &[(&'static str, kart_fmt::kart_object::FloatCurve)],
) -> (usize, bool) {
    let alpha_key = match alpha {
        None => "o".to_string(),
        Some(a) if a.use_alpha_test => format!("m{}", a.alpha_ref),
        Some(a) if a.use_blend => "b".to_string(),
        Some(_) => "o".to_string(),
    };
    let key = format!("{}|{}", tex.unwrap_or("none"), alpha_key);
    let discards_all = matches!(alpha, Some(a) if a.use_alpha_test && a.alpha_ref == 0)
        && tex
            .map(|name| texture_fully_transparent(name, c.tex_dirs))
            .unwrap_or(false);
    if let Some(&m) = c.mat_cache.get(&key) {
        return (m, discards_all);
    }
    let image = match tex {
        Some(name) => {
            if let Some(&cached) = c.tex_png.get(name) {
                cached
            } else {
                let img = load_track_texture(&mut c.g, name, c.tex_dirs);
                c.tex_png.insert(name.to_string(), img);
                img
            }
        }
        None => None,
    };
    let alpha_mode = match alpha {
        Some(a) if a.use_alpha_test => Alpha::Mask(a.alpha_ref as f32 / 255.0),
        Some(a) if a.use_blend => Alpha::Blend,
        _ => Alpha::Opaque,
    };
    // UV-scroll/scale controller curves preserved as material extras (core glTF
    // cannot animate a texture transform), matching C# material.extras.kartUvAnim.
    // Each curve carries its controller kf_type (1 linear / 3 step / else
    // cubic-Hermite) and per-key tangents — blink signs are step curves and
    // linear-sampling them would smear the beat.
    let extras = if c.anims && !uv_curves.is_empty() {
        let mut obj = serde_json::Map::new();
        for (param, curve) in uv_curves {
            let arr: Vec<serde_json::Value> = curve
                .keys
                .iter()
                .map(|(t, v, tin, tout)| serde_json::json!([t, v, tin, tout]))
                .collect();
            obj.insert(
                param.to_string(),
                serde_json::json!({ "kf_type": curve.kf_type, "keys": arr }),
            );
        }
        Some(serde_json::json!({ "kartUvAnim": obj }))
    } else {
        None
    };
    let m = c.g.add_material(MaterialDef {
        name: tex
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("mat_{key}")),
        base_color_image: image,
        base_color: [1.0, 1.0, 1.0, 1.0],
        double_sided: true, // C# default (Cull off → everything double-sided)
        alpha: alpha_mode,
        unlit: false,
        extras,
    });
    c.mat_cache.insert(key, m);
    (m, discards_all)
}

/// True when the named texture resolves (PNG or DDS, searched in `tex_dirs`
/// order) and every pixel has alpha 0. Used to detect the original game's
/// discard-all alpha-test surfaces (MASK with alpha_ref 0 keeps only
/// alpha > 0 texels, so a fully transparent texture renders nothing).
fn texture_fully_transparent(name: &str, tex_dirs: &[PathBuf]) -> bool {
    for dir in tex_dirs {
        if let Ok(png) = std::fs::read(dir.join(format!("{name}.png"))) {
            return png_alpha_fully_transparent(&png).unwrap_or(false);
        }
        if let Ok(dds) = std::fs::read(dir.join(format!("{name}.dds"))) {
            if let Some(img) = kart_fmt::dds::decode(&dds) {
                return img.pixels.chunks_exact(4).all(|px| px[3] == 0);
            }
            return false;
        }
    }
    false
}

fn png_alpha_fully_transparent(bytes: &[u8]) -> Option<bool> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let data = &buf[..info.buffer_size()];
    let all_zero = match info.color_type {
        png::ColorType::Rgba => data.chunks_exact(4).all(|px| px[3] == 0),
        png::ColorType::GrayscaleAlpha => data.chunks_exact(2).all(|px| px[1] == 0),
        // No alpha channel (or palette without tRNS handling): the original
        // alpha test keeps every texel, so this is never discard-all.
        _ => false,
    };
    Some(all_zero)
}

/// Resolve a track texture name to an embedded PNG image index by searching each
/// of `tex_dirs` in order (first hit wins): prefer an existing `<name>.png`, else
/// decode `<name>.dds` and re-encode to PNG (glTF accepts only PNG/JPG). Per-name
/// search across dirs is what lets a track pull most textures from its own theme
/// and a few from a `texTheme` override theme.
fn load_track_texture(g: &mut Gltf, name: &str, tex_dirs: &[PathBuf]) -> Option<usize> {
    for dir in tex_dirs {
        if let Ok(png) = std::fs::read(dir.join(format!("{name}.png"))) {
            return Some(g.add_png(&png));
        }
        if let Ok(dds) = std::fs::read(dir.join(format!("{name}.dds"))) {
            if let Some(img) = kart_fmt::dds::decode(&dds) {
                if let Some(png) = encode_png(img.width, img.height, &img.pixels) {
                    return Some(g.add_png(&png));
                }
            }
        }
    }
    None
}

fn encode_png(w: u32, h: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().ok()?;
        writer.write_image_data(rgba).ok()?;
    }
    Some(out)
}

/// Authored .1s meshes can carry all-zero normals (observed on factory_I02
/// track nodes); a zero normal makes N·L lighting black out the mesh, so fall
/// back to the face normal exactly as if the source had no normals at all.
fn source_or_face(n: [f32; 3], face: [f32; 3]) -> [f32; 3] {
    if n[0] * n[0] + n[1] * n[1] + n[2] * n[2] < 1e-12 {
        face
    } else {
        n
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1e-6 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}
