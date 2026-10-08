//! Structural tests for the hand-written GLB/glTF writer: container framing,
//! accessor/bufferView consistency, JSON schema shape, and determinism.
//! Exercises only the public `Gltf` builder API (scene.rs's `convert_raw` /
//! `convert_track` need `kart_fmt` domain fixtures and are out of scope here).

use kart_gltf::{Alpha, AnimChannel, Gltf, MaterialDef, NodeDef, Primitive};
use serde_json::Value;

const GLTF_MAGIC: u32 = 0x46546C67; // "glTF"
const JSON_CHUNK_TYPE: u32 = 0x4E4F534A; // "JSON"
const BIN_CHUNK_TYPE: u32 = 0x004E4942; // "BIN\0"

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Parsed GLB container split into its two chunks, with the header fields
/// re-derived for cross-checking against `bytes.len()`.
struct Glb {
    bytes: Vec<u8>,
    version: u32,
    total_length: u32,
    json_chunk_type: u32,
    json: Value,
    bin: Vec<u8>,
}

fn parse_glb(bytes: Vec<u8>) -> Glb {
    assert_eq!(read_u32(&bytes, 0), GLTF_MAGIC, "magic");
    let version = read_u32(&bytes, 4);
    let total_length = read_u32(&bytes, 8);

    let json_len = read_u32(&bytes, 12) as usize;
    let json_chunk_type = read_u32(&bytes, 16);
    let json_start = 20;
    let json_bytes = &bytes[json_start..json_start + json_len];
    let json: Value = serde_json::from_slice(json_bytes).expect("valid JSON chunk");

    let bin_start = json_start + json_len;
    let bin_len = read_u32(&bytes, bin_start) as usize;
    let bin_chunk_type = read_u32(&bytes, bin_start + 4);
    assert_eq!(bin_chunk_type, BIN_CHUNK_TYPE, "bin chunk type");
    let bin = bytes[bin_start + 8..bin_start + 8 + bin_len].to_vec();

    Glb {
        bytes,
        version,
        total_length,
        json_chunk_type,
        json,
        bin,
    }
}

/// A minimal 2-triangle quad primitive (4 verts, 6 indices) with material `mat`.
fn quad_primitive(material: Option<usize>) -> Primitive {
    Primitive {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        joints: None,
        weights: None,
        material,
    }
}

fn build_quad_gltf() -> Gltf {
    let mut g = Gltf::new();
    let mat = g.add_material(MaterialDef {
        name: "quad_mat".to_string(),
        base_color_image: None,
        base_color: [1.0, 1.0, 1.0, 1.0],
        double_sided: true,
        alpha: Alpha::Opaque,
        unlit: false,
        extras: None,
    });
    let mesh = g.add_mesh("quad", vec![quad_primitive(Some(mat))]);
    let node = g.add_node(NodeDef {
        name: Some("quad_node".to_string()),
        mesh: Some(mesh),
        ..Default::default()
    });
    g.set_scene_roots(&[node]);
    g
}

// ---- 1. GLB container structure ---------------------------------------

#[test]
fn glb_header_fields_are_correct() {
    let bytes = build_quad_gltf().into_glb();
    let glb = parse_glb(bytes);
    assert_eq!(glb.version, 2);
    assert_eq!(glb.json_chunk_type, JSON_CHUNK_TYPE);
    assert_eq!(
        glb.total_length as usize,
        glb.bytes.len(),
        "total_length header must match actual byte count"
    );
}

#[test]
fn json_chunk_is_4_byte_aligned_and_space_padded() {
    let bytes = build_quad_gltf().into_glb();
    let json_len = read_u32(&bytes, 12) as usize;
    assert_eq!(json_len % 4, 0, "JSON chunk length must be 4-byte aligned");
    // The last byte(s) of padding (if any) must be ASCII space, not NUL, per spec.
    let json_start = 20;
    let json_bytes = &bytes[json_start..json_start + json_len];
    // Find where the real JSON content ends (last non-space byte should be '}').
    let last_non_space = json_bytes.iter().rposition(|&b| b != b' ').unwrap();
    assert_eq!(json_bytes[last_non_space], b'}');
    for &b in &json_bytes[last_non_space + 1..] {
        assert_eq!(b, b' ', "JSON padding must be spaces");
    }
}

#[test]
fn bin_chunk_is_4_byte_aligned_and_zero_padded() {
    let bytes = build_quad_gltf().into_glb();
    let json_len = read_u32(&bytes, 12) as usize;
    let bin_chunk_start = 20 + json_len;
    let bin_len = read_u32(&bytes, bin_chunk_start) as usize;
    assert_eq!(bin_len % 4, 0, "BIN chunk length must be 4-byte aligned");

    let glb = parse_glb(bytes);
    // Every bufferView must fit inside the (aligned) bin; any padding tail
    // beyond the last view's data should be zero.
    let buffer_views = glb.json["bufferViews"].as_array().cloned().unwrap_or_default();
    let last_end = buffer_views
        .iter()
        .map(|v| {
            v["byteOffset"].as_u64().unwrap_or(0) + v["byteLength"].as_u64().unwrap_or(0)
        })
        .max()
        .unwrap_or(0) as usize;
    for &b in &glb.bin[last_end..] {
        assert_eq!(b, 0, "BIN padding must be zero bytes");
    }
}

#[test]
fn empty_gltf_still_produces_a_valid_container() {
    let bytes = Gltf::new().into_glb();
    let glb = parse_glb(bytes);
    assert_eq!(glb.json["asset"]["version"], "2.0");
    assert!(glb.bin.is_empty());
}

// ---- 2. accessor / bufferView consistency ------------------------------

#[test]
fn quad_accessors_match_written_data() {
    let bytes = build_quad_gltf().into_glb();
    let glb = parse_glb(bytes);
    let accessors = glb.json["accessors"].as_array().unwrap();
    let buffer_views = glb.json["bufferViews"].as_array().unwrap();

    // Attributes/indices order from Gltf::add_mesh: POSITION, NORMAL, TEXCOORD_0, indices.
    let position = &accessors[0];
    assert_eq!(position["componentType"], 5126); // FLOAT
    assert_eq!(position["type"], "VEC3");
    assert_eq!(position["count"], 4);
    assert!(position["min"].is_array(), "POSITION accessor needs min");
    assert!(position["max"].is_array(), "POSITION accessor needs max");
    assert_eq!(position["min"], serde_json::json!([0.0, 0.0, 0.0]));
    assert_eq!(position["max"], serde_json::json!([1.0, 1.0, 0.0]));

    let normal = &accessors[1];
    assert_eq!(normal["componentType"], 5126);
    assert_eq!(normal["type"], "VEC3");
    assert_eq!(normal["count"], 4);

    let texcoord = &accessors[2];
    assert_eq!(texcoord["componentType"], 5126);
    assert_eq!(texcoord["type"], "VEC2");
    assert_eq!(texcoord["count"], 4);

    // 6 indices, max value 3 -> fits u8 (componentType 5121).
    let indices = &accessors[3];
    assert_eq!(indices["componentType"], 5121);
    assert_eq!(indices["type"], "SCALAR");
    assert_eq!(indices["count"], 6);

    // Every bufferView: offset 4-aligned, offset+length within the bin buffer.
    let bin_len = glb.bin.len() as u64;
    for view in buffer_views {
        let offset = view["byteOffset"].as_u64().unwrap();
        let length = view["byteLength"].as_u64().unwrap();
        assert_eq!(offset % 4, 0, "bufferView offset must be 4-byte aligned");
        assert!(
            offset + length <= bin_len,
            "bufferView must not exceed buffer length"
        );
    }

    // POSITION/NORMAL/TEXCOORD0 views carry vertex-buffer sizes consistent with
    // the accessor: byteLength == count * component_size.
    let pos_view = &buffer_views[position["bufferView"].as_u64().unwrap() as usize];
    assert_eq!(pos_view["byteLength"], 4 * 3 * 4); // 4 verts * vec3 * f32
    let idx_view = &buffer_views[indices["bufferView"].as_u64().unwrap() as usize];
    assert_eq!(idx_view["byteLength"], 6); // 6 * u8
}

#[test]
fn no_bufferview_overlap() {
    let bytes = build_quad_gltf().into_glb();
    let glb = parse_glb(bytes);
    let mut views: Vec<(u64, u64)> = glb.json["bufferViews"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let off = v["byteOffset"].as_u64().unwrap();
            let len = v["byteLength"].as_u64().unwrap();
            (off, off + len)
        })
        .collect();
    views.sort();
    for pair in views.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0,
            "bufferViews must not overlap: {:?}",
            pair
        );
    }
}

#[test]
fn indices_pick_smallest_legal_component_type() {
    // 300 distinct vertices -> max index 299 requires u16 (component 5123), not u8.
    let mut g = Gltf::new();
    let n = 300usize;
    let positions: Vec<[f32; 3]> = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
    let indices: Vec<u32> = (0..n as u32).collect();
    let prim = Primitive {
        positions,
        normals: vec![],
        uvs: vec![],
        indices,
        joints: None,
        weights: None,
        material: None,
    };
    g.add_mesh("big", vec![prim]);
    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let accessors = glb.json["accessors"].as_array().unwrap();
    // POSITION is accessors[0], indices accessors[1] (no normals/uvs supplied).
    let indices_accessor = &accessors[1];
    assert_eq!(indices_accessor["componentType"], 5123); // UNSIGNED_SHORT
    assert_eq!(indices_accessor["count"], n as u64);
}

// ---- 3. JSON validity / required fields --------------------------------

#[test]
fn required_top_level_fields_present() {
    let bytes = build_quad_gltf().into_glb();
    let glb = parse_glb(bytes);
    assert_eq!(glb.json["asset"]["version"], "2.0");
    assert!(glb.json["asset"]["generator"].is_string());
    assert_eq!(glb.json["scene"], 0);
    let scenes = glb.json["scenes"].as_array().unwrap();
    assert_eq!(scenes.len(), 1);
    assert_eq!(scenes[0]["nodes"], serde_json::json!([0]));

    let nodes = glb.json["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["mesh"], 0);

    let meshes = glb.json["meshes"].as_array().unwrap();
    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0]["primitives"].as_array().unwrap().len(), 1);

    let buffers = glb.json["buffers"].as_array().unwrap();
    assert_eq!(buffers.len(), 1);
    assert_eq!(buffers[0]["byteLength"], glb.bin.len() as u64);
}

#[test]
fn scene_roots_empty_omits_scene_pointer() {
    // A Gltf with no scene_roots set must not emit a "scene" index (glTF spec:
    // scene must reference a valid index, so it should be omitted rather than 0).
    let bytes = Gltf::new().into_glb();
    let glb = parse_glb(bytes);
    assert!(glb.json.get("scene").is_none() || glb.json["scene"].is_null());
}

// ---- 4. optional feature paths -----------------------------------------

#[test]
fn skinned_mesh_has_joints_and_weights_accessors() {
    let mut g = Gltf::new();
    let prim = Primitive {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![],
        uvs: vec![],
        indices: vec![0, 1, 2],
        joints: Some(vec![[0, 1, 0, 0]; 3]),
        weights: Some(vec![[0.5, 0.5, 0.0, 0.0]; 3]),
        material: None,
    };
    let mesh = g.add_mesh("skinned", vec![prim]);
    let joint_node = g.add_node(NodeDef {
        name: Some("bone0".to_string()),
        ..Default::default()
    });
    let mesh_node = g.add_node(NodeDef {
        mesh: Some(mesh),
        ..Default::default()
    });
    let skin = g.add_skin(
        &[joint_node],
        &[[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]],
        Some(joint_node),
    );
    g.set_node_skin(mesh_node, skin);
    g.set_scene_roots(&[joint_node, mesh_node]);

    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let mesh_json = &glb.json["meshes"][0]["primitives"][0];
    assert!(mesh_json["attributes"]["JOINTS_0"].is_number());
    assert!(mesh_json["attributes"]["WEIGHTS_0"].is_number());

    let joints_acc_idx = mesh_json["attributes"]["JOINTS_0"].as_u64().unwrap() as usize;
    let joints_acc = &glb.json["accessors"][joints_acc_idx];
    assert_eq!(joints_acc["componentType"], 5123); // UNSIGNED_SHORT
    assert_eq!(joints_acc["type"], "VEC4");

    let weights_acc_idx = mesh_json["attributes"]["WEIGHTS_0"].as_u64().unwrap() as usize;
    let weights_acc = &glb.json["accessors"][weights_acc_idx];
    assert_eq!(weights_acc["componentType"], 5126); // FLOAT
    assert_eq!(weights_acc["type"], "VEC4");

    let skins = glb.json["skins"].as_array().unwrap();
    assert_eq!(skins.len(), 1);
    assert_eq!(skins[0]["joints"], serde_json::json!([0]));
    assert_eq!(glb.json["nodes"][1]["skin"], 0);
}

#[test]
fn alpha_mask_material_sets_mode_and_cutoff() {
    let mut g = Gltf::new();
    g.add_material(MaterialDef {
        name: "mask_mat".to_string(),
        base_color_image: None,
        base_color: [1.0, 1.0, 1.0, 1.0],
        double_sided: false,
        alpha: Alpha::Mask(0.5),
        unlit: false,
        extras: None,
    });
    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let mat = &glb.json["materials"][0];
    assert_eq!(mat["alphaMode"], "MASK");
    assert_eq!(mat["alphaCutoff"], 0.5);
    assert!(mat.get("doubleSided").is_none() || mat["doubleSided"] == false);
}

#[test]
fn alpha_blend_material_sets_mode_without_cutoff() {
    let mut g = Gltf::new();
    g.add_material(MaterialDef {
        name: "blend_mat".to_string(),
        base_color_image: None,
        base_color: [1.0, 1.0, 1.0, 0.5],
        double_sided: true,
        alpha: Alpha::Blend,
        unlit: false,
        extras: None,
    });
    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let mat = &glb.json["materials"][0];
    assert_eq!(mat["alphaMode"], "BLEND");
    assert!(mat.get("alphaCutoff").is_none());
    assert_eq!(mat["doubleSided"], true);
}

#[test]
fn opaque_material_omits_alpha_mode() {
    let mut g = Gltf::new();
    g.add_material(MaterialDef {
        name: "opaque_mat".to_string(),
        base_color_image: None,
        base_color: [1.0, 1.0, 1.0, 1.0],
        double_sided: false,
        alpha: Alpha::Opaque,
        unlit: false,
        extras: None,
    });
    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let mat = &glb.json["materials"][0];
    assert!(mat.get("alphaMode").is_none());
    assert!(mat.get("alphaCutoff").is_none());
}

#[test]
fn trs_animation_produces_sampler_and_channel_per_path() {
    let mut g = Gltf::new();
    let node = g.add_node(NodeDef {
        translation: Some([0.0, 0.0, 0.0]),
        ..Default::default()
    });
    g.set_scene_roots(&[node]);
    g.add_animation(
        "clip",
        vec![
            AnimChannel {
                node,
                path: "translation",
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            },
            AnimChannel {
                node,
                path: "rotation",
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            AnimChannel {
                node,
                path: "scale",
                times: vec![0.0, 1.0],
                values: vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            },
        ],
    );

    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let anims = glb.json["animations"].as_array().unwrap();
    assert_eq!(anims.len(), 1);
    let anim = &anims[0];
    assert_eq!(anim["name"], "clip");
    let channels = anim["channels"].as_array().unwrap();
    assert_eq!(channels.len(), 3);
    let paths: Vec<&str> = channels
        .iter()
        .map(|c| c["target"]["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"translation"));
    assert!(paths.contains(&"rotation"));
    assert!(paths.contains(&"scale"));
    assert_eq!(anim["samplers"].as_array().unwrap().len(), 3);

    // Rotation output accessor must be VEC4, translation/scale VEC3.
    for ch in channels {
        let sampler_idx = ch["sampler"].as_u64().unwrap() as usize;
        let sampler = &anim["samplers"][sampler_idx];
        let output_idx = sampler["output"].as_u64().unwrap() as usize;
        let acc = &glb.json["accessors"][output_idx];
        let path = ch["target"]["path"].as_str().unwrap();
        if path == "rotation" {
            assert_eq!(acc["type"], "VEC4");
        } else {
            assert_eq!(acc["type"], "VEC3");
        }
    }
}

#[test]
fn animation_with_empty_channels_is_dropped() {
    let mut g = Gltf::new();
    let node = g.add_node(NodeDef::default());
    g.set_scene_roots(&[node]);
    g.add_animation(
        "empty",
        vec![AnimChannel {
            node,
            path: "translation",
            times: vec![],
            values: vec![],
        }],
    );
    let bytes = g.into_glb();
    let glb = parse_glb(bytes);
    let anims = glb.json["animations"].as_array().cloned().unwrap_or_default();
    assert!(anims.is_empty(), "animation with all-empty channels must be dropped");
}

// ---- 5. round-trip determinism -----------------------------------------

#[test]
fn same_input_produces_byte_identical_glb() {
    let a = build_quad_gltf().into_glb();
    let b = build_quad_gltf().into_glb();
    assert_eq!(a, b, "exporting the same scene twice must be byte-for-byte deterministic");
}

#[test]
fn same_input_produces_byte_identical_glb_with_skin_and_animation() {
    fn build() -> Vec<u8> {
        let mut g = Gltf::new();
        let prim = Primitive {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            joints: Some(vec![[0, 0, 0, 0]; 3]),
            weights: Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]),
            material: None,
        };
        let mesh = g.add_mesh("m", vec![prim]);
        let joint = g.add_node(NodeDef::default());
        let mesh_node = g.add_node(NodeDef {
            mesh: Some(mesh),
            ..Default::default()
        });
        let skin = g.add_skin(
            &[joint],
            &[[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]],
            Some(joint),
        );
        g.set_node_skin(mesh_node, skin);
        g.set_scene_roots(&[joint, mesh_node]);
        g.add_animation(
            "clip",
            vec![AnimChannel {
                node: joint,
                path: "translation",
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }],
        );
        g.into_glb()
    }
    assert_eq!(build(), build());
}
