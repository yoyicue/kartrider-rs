//! USD ASCII (`.usda`) exporter for KartRider assets — the oracle-local
//! `karttools` output format, written here directly from the open USD spec
//! (the oracle's own exporter is closed `.pyc`). Emits static geometry +
//! per-node transforms + textured `UsdPreviewSurface` materials for raw-magic
//! models (character / kart / pet / stuff) and KartObject tracks. The parsed
//! geometry is shared with the glTF path, so vertex/triangle counts match.

use kart_fmt::kart_object::{self, Track};
use kart_fmt::raw_model::{RawModel, RawNode};
use std::fmt::Write;
use std::path::Path;

/// Row-vector (USD/GfMatrix) Z-up→Y-up swap, matching the glTF axis swap.
const SWAP_ROWS: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn header(default_prim: &str) -> String {
    format!(
        "#usda 1.0\n(\n    defaultPrim = \"{default_prim}\"\n    metersPerUnit = 1\n    upAxis = \"Y\"\n)\n\n"
    )
}

fn fmt_matrix(rows: &[[f64; 4]; 4]) -> String {
    let r = |v: &[f64; 4]| format!("({}, {}, {}, {})", v[0], v[1], v[2], v[3]);
    format!(
        "( {}, {}, {}, {} )",
        r(&rows[0]),
        r(&rows[1]),
        r(&rows[2]),
        r(&rows[3])
    )
}

fn usd_name(raw: &str, fallback: &str) -> String {
    let mut s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        s = fallback.to_string();
    }
    if s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
        s.insert(0, '_');
    }
    s
}

// ---- a generic mesh accumulated for emission --------------------------------

struct UsdMesh {
    name: String,
    matrix: [[f64; 4]; 4],
    points: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    face_counts: usize, // number of triangles (faceVertexCounts all = 3)
    material: Option<usize>,
}

struct UsdMaterial {
    name: String,
    texture: Option<String>, // relative file path
}

fn write_mesh(out: &mut String, m: &UsdMesh, mat_names: &[UsdMaterial]) {
    let pad = "        ";
    let _ = writeln!(out, "    def Mesh \"{}\"\n    {{", m.name);
    // points
    let pts: Vec<String> = m
        .points
        .iter()
        .map(|p| format!("({}, {}, {})", p[0], p[1], p[2]))
        .collect();
    let _ = writeln!(out, "{pad}point3f[] points = [{}]", pts.join(", "));
    // faces (all triangles, sequential faceVarying indices)
    let n = m.face_counts;
    let counts = vec!["3"; n].join(", ");
    let idx: Vec<String> = (0..n * 3).map(|i| i.to_string()).collect();
    let _ = writeln!(out, "{pad}int[] faceVertexCounts = [{counts}]");
    let _ = writeln!(out, "{pad}int[] faceVertexIndices = [{}]", idx.join(", "));
    // normals (faceVarying)
    let nrm: Vec<String> = m
        .normals
        .iter()
        .map(|p| format!("({}, {}, {})", p[0], p[1], p[2]))
        .collect();
    let _ = writeln!(
        out,
        "{pad}normal3f[] normals = [{}] (interpolation = \"faceVarying\")",
        nrm.join(", ")
    );
    // uvs (faceVarying st)
    let st: Vec<String> = m
        .uvs
        .iter()
        .map(|p| format!("({}, {})", p[0], p[1]))
        .collect();
    let _ = writeln!(
        out,
        "{pad}texCoord2f[] primvars:st = [{}] (interpolation = \"faceVarying\")",
        st.join(", ")
    );
    // material binding
    if let Some(mi) = m.material {
        let _ = writeln!(
            out,
            "{pad}rel material:binding = </{}/Materials/{}>",
            ROOT, mat_names[mi].name
        );
    }
    // transform
    let _ = writeln!(
        out,
        "{pad}matrix4d xformOp:transform = {}",
        fmt_matrix(&m.matrix)
    );
    let _ = writeln!(
        out,
        "{pad}uniform token[] xformOpOrder = [\"xformOp:transform\"]"
    );
    let _ = writeln!(out, "    }}");
}

const ROOT: &str = "Root";

fn write_materials(out: &mut String, mats: &[UsdMaterial]) {
    if mats.is_empty() {
        return;
    }
    let _ = writeln!(out, "    def Scope \"Materials\"\n    {{");
    for m in mats {
        let _ = writeln!(out, "        def Material \"{}\"\n        {{", m.name);
        let _ = writeln!(out, "            token outputs:surface.connect = </{ROOT}/Materials/{}/Surface.outputs:surface>", m.name);
        let _ = writeln!(out, "            def Shader \"Surface\"\n            {{");
        let _ = writeln!(
            out,
            "                uniform token info:id = \"UsdPreviewSurface\""
        );
        if m.texture.is_some() {
            let _ = writeln!(out, "                color3f inputs:diffuseColor.connect = </{ROOT}/Materials/{}/Tex.outputs:rgb>", m.name);
        } else {
            let _ = writeln!(
                out,
                "                color3f inputs:diffuseColor = (0.8, 0.8, 0.8)"
            );
        }
        let _ = writeln!(out, "                token outputs:surface");
        let _ = writeln!(out, "            }}");
        if let Some(tex) = &m.texture {
            let _ = writeln!(out, "            def Shader \"Tex\"\n            {{");
            let _ = writeln!(
                out,
                "                uniform token info:id = \"UsdUVTexture\""
            );
            let _ = writeln!(out, "                asset inputs:file = @{tex}@");
            let _ = writeln!(out, "                float2 inputs:st.connect = </{ROOT}/Materials/{}/uv.outputs:result>", m.name);
            let _ = writeln!(out, "                float3 outputs:rgb");
            let _ = writeln!(out, "            }}");
            let _ = writeln!(out, "            def Shader \"uv\"\n            {{");
            let _ = writeln!(
                out,
                "                uniform token info:id = \"UsdPrimvarReader_float2\""
            );
            let _ = writeln!(out, "                token inputs:varname = \"st\"");
            let _ = writeln!(out, "                float2 outputs:result");
            let _ = writeln!(out, "            }}");
        }
        let _ = writeln!(out, "        }}");
    }
    let _ = writeln!(out, "    }}");
}

fn assemble(
    name: &str,
    meshes: &[UsdMesh],
    mats: &[UsdMaterial],
    skel_json: Option<String>,
) -> String {
    let mut out = header(ROOT);
    let custom = match &skel_json {
        Some(j) => format!(
            " (\n    customData = {{\n        string kart_skelData = {}\n    }}\n)",
            usd_escape(j)
        ),
        None => String::new(),
    };
    let _ = writeln!(
        out,
        "def Xform \"{ROOT}\" (kind = \"component\"){custom}\n{{"
    );
    let _ = writeln!(
        out,
        "    matrix4d xformOp:transform = {}",
        fmt_matrix(&SWAP_ROWS)
    );
    let _ = writeln!(
        out,
        "    uniform token[] xformOpOrder = [\"xformOp:transform\"]"
    );
    let _ = writeln!(out, "    string userProperties:asset = \"{name}\"");
    for m in meshes {
        write_mesh(&mut out, m, mats);
    }
    write_materials(&mut out, mats);
    let _ = writeln!(out, "}}");
    out
}

// ---- raw model --------------------------------------------------------------

/// Raw-node local transform as a USD row-vector matrix (S·R·T; column form would
/// transpose for glTF, but USD/GfMatrix uses the engine's row convention as-is).
fn raw_matrix(n: &RawNode) -> [[f64; 4]; 4] {
    let r = n
        .rotation
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let [sx, sy, sz] = [n.scale[0] as f64, n.scale[1] as f64, n.scale[2] as f64];
    let r = r.map(|x| x as f64);
    let p = n.position.map(|x| x as f64);
    [
        [sx * r[0], sx * r[1], sx * r[2], 0.0],
        [sy * r[3], sy * r[4], sy * r[5], 0.0],
        [sz * r[6], sz * r[7], sz * r[8], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

/// Export a raw model to USDA. `tex_dir` resolves `<meshIndex>.png` textures.
pub fn raw_model_usda(model: &RawModel, name: &str, tex_dir: &Path) -> String {
    let mut meshes = Vec::new();
    let mut mats: Vec<UsdMaterial> = Vec::new();
    let mut mat_idx: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut counter = 0usize;
    for n in &model.nodes {
        emit_raw_node(
            n,
            identity4(),
            tex_dir,
            &mut meshes,
            &mut mats,
            &mut mat_idx,
            &mut counter,
        );
    }
    // Character skeleton + skin weights, embedded as JSON customData (the way the
    // oracle's usda_model serialises kart_skelData for the C4D rig importer).
    let skel = if !model.bones.is_empty() {
        Some(skel_json(model))
    } else {
        None
    };
    assemble(name, &meshes, &mats, skel)
}

/// Serialise the bone table (parent, inverse-bind 3x4, local 3x4) and the body
/// mesh's per-vertex skin records as a compact JSON string.
fn skel_json(model: &RawModel) -> String {
    let arr = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut bones = Vec::new();
    for b in &model.bones {
        bones.push(format!(
            "{{\"index\":{},\"parent\":{},\"inv_bind\":[{}],\"local\":[{}]}}",
            b.index,
            b.parent,
            arr(&b.inv_bind),
            arr(&b.local)
        ));
    }
    let mut skin = Vec::new();
    if let Some(body) = model
        .nodes
        .iter()
        .find(|n| n.is_body && !n.bone_data.is_empty())
    {
        for w in body.skin_weights() {
            skin.push(format!(
                "[{},{},{},{}]",
                w.index0, w.index1, w.weight0, w.weight1
            ));
        }
    }
    format!(
        "{{\"bones\":[{}],\"skin\":[{}]}}",
        bones.join(","),
        skin.join(",")
    )
}

/// Quote a string as a USD string literal (escape backslash and double-quote).
fn usd_escape(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn emit_raw_node(
    n: &RawNode,
    parent: [[f64; 4]; 4],
    tex_dir: &Path,
    meshes: &mut Vec<UsdMesh>,
    mats: &mut Vec<UsdMaterial>,
    mat_idx: &mut std::collections::HashMap<i32, usize>,
    counter: &mut usize,
) {
    let world = mat_mul(&parent, &raw_matrix(n));
    let idx = *counter;
    *counter += 1;
    if let Some(m) = raw_node_mesh(n, idx, &world, tex_dir, mats, mat_idx) {
        meshes.push(m);
    }
    for c in &n.children {
        emit_raw_node(c, world, tex_dir, meshes, mats, mat_idx, counter);
    }
}

fn raw_node_mesh(
    n: &RawNode,
    idx: usize,
    world: &[[f64; 4]; 4],
    tex_dir: &Path,
    mats: &mut Vec<UsdMaterial>,
    mat_idx: &mut std::collections::HashMap<i32, usize>,
) -> Option<UsdMesh> {
    if !n.is_mesh || n.vertices.is_empty() || n.vertex_indices.is_empty() {
        return None;
    }
    let has_n = n.normals.len() == n.vertices.len();
    let (mut points, mut normals, mut uvs) = (Vec::new(), Vec::new(), Vec::new());
    let vi = &n.vertex_indices;
    let ui = &n.uv_indices;
    for f in 0..vi.len() / 3 {
        let tri = [
            vi[f * 3] as usize,
            vi[f * 3 + 1] as usize,
            vi[f * 3 + 2] as usize,
        ];
        if tri.iter().any(|&v| v >= n.vertices.len()) {
            continue;
        }
        let p = [n.vertices[tri[0]], n.vertices[tri[1]], n.vertices[tri[2]]];
        let nr = if has_n {
            [n.normals[tri[0]], n.normals[tri[1]], n.normals[tri[2]]]
        } else {
            [face_normal(p[0], p[1], p[2]); 3]
        };
        let uv = [
            uv_at(n, ui, f * 3),
            uv_at(n, ui, f * 3 + 2),
            uv_at(n, ui, f * 3 + 1),
        ];
        // reversed winding (a,c,b) to compensate the det(−1) swap root
        for (k, &j) in [0usize, 2, 1].iter().enumerate() {
            points.push(p[j]);
            normals.push(nr[j]);
            uvs.push(uv[k]);
        }
    }
    if points.is_empty() {
        return None;
    }
    let material = Some(raw_material(n.mesh_index, tex_dir, mats, mat_idx));
    Some(UsdMesh {
        name: usd_name(&n.name, &format!("mesh{idx}")),
        matrix: *world,
        face_counts: points.len() / 3,
        points,
        normals,
        uvs,
        material,
    })
}

fn raw_material(
    mesh_index: i32,
    tex_dir: &Path,
    mats: &mut Vec<UsdMaterial>,
    cache: &mut std::collections::HashMap<i32, usize>,
) -> usize {
    if let Some(&m) = cache.get(&mesh_index) {
        return m;
    }
    let texture = if mesh_index >= 0 && tex_dir.join(format!("{mesh_index}.png")).exists() {
        Some(format!("{mesh_index}.png"))
    } else {
        None
    };
    mats.push(UsdMaterial {
        name: format!("mat{}", if mesh_index < 0 { 0 } else { mesh_index }),
        texture,
    });
    let i = mats.len() - 1;
    cache.insert(mesh_index, i);
    i
}

// ---- track ------------------------------------------------------------------

fn track_matrix(t: &[f32; 9], pos: [f32; 3], scale: [f32; 3]) -> [[f64; 4]; 4] {
    let t = t.map(|x| x as f64);
    let [sx, sy, sz] = [scale[0] as f64, scale[1] as f64, scale[2] as f64];
    let p = pos.map(|x| x as f64);
    [
        [sx * t[0], sx * t[3], sx * t[6], 0.0],
        [sy * t[1], sy * t[4], sy * t[7], 0.0],
        [sz * t[2], sz * t[5], sz * t[8], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

/// Export a KartObject track to USDA (geometry + transforms + texture refs).
pub fn track_usda(track: &Track, name: &str, tex_dir: &Path) -> String {
    let mut meshes = Vec::new();
    let mut mats: Vec<UsdMaterial> = Vec::new();
    let mut mat_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if let Some(r) = track.root {
        // scene-root transform is replaced by the swap (handled by the Root Xform)
        let identity = track_matrix(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0; 3],
            [1.0; 3],
        );
        emit_track_node(
            track,
            r,
            identity,
            true,
            tex_dir,
            &mut meshes,
            &mut mats,
            &mut mat_idx,
        );
    }
    assemble(name, &meshes, &mats, None)
}

#[allow(clippy::too_many_arguments)]
fn emit_track_node(
    track: &Track,
    i: usize,
    parent: [[f64; 4]; 4],
    is_root: bool,
    tex_dir: &Path,
    meshes: &mut Vec<UsdMesh>,
    mats: &mut Vec<UsdMaterial>,
    mat_idx: &mut std::collections::HashMap<String, usize>,
) {
    let n = &track.nodes[i];
    let local = if is_root {
        identity4()
    } else {
        track_matrix(&n.transform, n.position, n.scale)
    };
    let world = mat_mul(&parent, &local);
    if let Some(geom) = &n.geom {
        if let Some(m) = track_node_mesh(n, geom, &world, tex_dir, mats, mat_idx) {
            meshes.push(m);
        }
    }
    for &c in &n.children {
        emit_track_node(track, c, world, false, tex_dir, meshes, mats, mat_idx);
    }
}

fn track_node_mesh(
    n: &kart_object::Node,
    geom: &kart_object::Geom,
    world: &[[f64; 4]; 4],
    tex_dir: &Path,
    mats: &mut Vec<UsdMaterial>,
    mat_idx: &mut std::collections::HashMap<String, usize>,
) -> Option<UsdMesh> {
    if geom.vertices.is_empty() || geom.indices.is_empty() {
        return None;
    }
    let has_n = geom.normals.len() == geom.vertices.len();
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
    let (mut points, mut normals, mut uvs) = (Vec::new(), Vec::new(), Vec::new());
    for tri in &tris {
        if tri.iter().any(|&v| v as usize >= geom.vertices.len()) {
            continue;
        }
        let p = [
            geom.vertices[tri[0] as usize],
            geom.vertices[tri[1] as usize],
            geom.vertices[tri[2] as usize],
        ];
        if is_degenerate(p[0], p[1], p[2]) {
            continue;
        }
        let nr = if has_n {
            [
                geom.normals[tri[0] as usize],
                geom.normals[tri[1] as usize],
                geom.normals[tri[2] as usize],
            ]
        } else {
            [face_normal(p[0], p[1], p[2]); 3]
        };
        for &j in &[0usize, 2, 1] {
            points.push(p[j]);
            normals.push(nr[j]);
            uvs.push(geom.uvs.get(tri[j] as usize).copied().unwrap_or([0.0, 0.0]));
        }
    }
    if points.is_empty() {
        return None;
    }
    let material = n
        .tex
        .as_deref()
        .map(|tex| track_material(tex, tex_dir, mats, mat_idx));
    Some(UsdMesh {
        name: usd_name(&n.name, "mesh"),
        matrix: *world,
        face_counts: points.len() / 3,
        points,
        normals,
        uvs,
        material,
    })
}

fn track_material(
    tex: &str,
    tex_dir: &Path,
    mats: &mut Vec<UsdMaterial>,
    cache: &mut std::collections::HashMap<String, usize>,
) -> usize {
    if let Some(&m) = cache.get(tex) {
        return m;
    }
    let texture = if tex_dir.join(format!("{tex}.png")).exists() {
        Some(format!("{tex}.png"))
    } else if tex_dir.join(format!("{tex}.dds")).exists() {
        Some(format!("{tex}.dds"))
    } else {
        None
    };
    mats.push(UsdMaterial {
        name: usd_name(tex, "mat"),
        texture,
    });
    let i = mats.len() - 1;
    cache.insert(tex.to_string(), i);
    i
}

// ---- math helpers -----------------------------------------------------------

fn identity4() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Row-vector matrix product (a·b), USD/GfMatrix convention.
fn mat_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut o = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn uv_at(n: &RawNode, ui: &[i32], k: usize) -> [f32; 2] {
    if k >= ui.len() {
        return [0.0, 0.0];
    }
    let i = ui[k];
    if i < 0 || i as usize >= n.uvs.len() {
        return [0.0, 0.0];
    }
    n.uvs[i as usize]
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

fn is_degenerate(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cr = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] < 1e-12
}
