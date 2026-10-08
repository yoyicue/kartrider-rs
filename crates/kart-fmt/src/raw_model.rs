//! Clean-room parser for KartRider's raw-magic `.1s` MODEL format (character and
//! kart vehicle models). A fixed-layout node tree, NOT the KartObject protocol.
//! Layout reverse-engineered as format facts; see docs/kartrider-1s-format.md.
//! Ported from the validated C# RawModelParser (M2 = geometry; skin in M5).

pub type Vec3 = [f32; 3];
pub type Vec2 = [f32; 2];

const MAGIC_KART: &[u8] = &[0xAA, 0x47, 0x49, 0x02, 0x8C, 0x07];
const MAGIC_CHAR_A: &[u8] = &[0xAA, 0x47, 0x44, 0x04, 0x6F, 0x18];
const MAGIC_CHAR_B: &[u8] = &[0xAA, 0x47, 0x12, 0x02, 0x6E, 0x07];
const MAGIC_DUMMY: &[u8] = &[0xAA, 0x47, 0x3C, 0x03, 0x07, 0x0E];
const MAGIC_MESH: &[u8] = &[0xAA, 0x47, 0x46, 0x04, 0x2A, 0x19];
const MAGIC_BODY_MESH: &[u8] = &[0xAA, 0x47, 0x23, 0x05, 0x33, 0x23];
const MAGIC_VERTEX: u16 = 0x27AA;

#[derive(Default, Clone)]
pub struct RawModel {
    pub asset_type: &'static str,
    pub nodes: Vec<RawNode>,
    pub bones: Vec<RawBone>,
}

/// Per-vertex skin binding: up to two bones with normalized blend weights.
/// `index1 == -1` means single-bone (rigid) binding.
#[derive(Clone, Copy)]
pub struct SkinWeight {
    pub index0: i32,
    pub index1: i32,
    pub weight0: f32,
    pub weight1: f32,
}

#[derive(Default, Clone)]
pub struct RawNode {
    pub name: String,
    pub node_id: i16,
    pub children: Vec<RawNode>,
    pub is_mesh: bool,
    pub is_body: bool,
    pub mesh_index: i32,
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub vertex_indices: Vec<i32>,
    pub uv_indices: Vec<i32>,
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Option<[f32; 9]>,
    pub bone_data: Vec<[u8; 12]>, // raw per-vertex skin (decoded in M5)
    /// Node ids a kart anim channel can bind to (self, plus a collapsed dummy's id).
    pub anim_entity_ids: Vec<i32>,
}

#[derive(Clone)]
pub struct RawBone {
    pub index: usize,
    pub parent: i32,
    pub inv_bind: [f32; 12],
    pub local: [f32; 12],
}

pub fn is_raw_model(d: &[u8]) -> bool {
    d.len() >= 6 && (&d[..6] == MAGIC_KART || &d[..6] == MAGIC_CHAR_A || &d[..6] == MAGIC_CHAR_B)
}

pub fn parse(d: &[u8]) -> RawModel {
    let mut r = Reader::new(d);
    let is_kart = &d[..6] == MAGIC_KART;
    let mut model = RawModel {
        asset_type: if is_kart { "kart" } else { "character" },
        ..Default::default()
    };
    r.pos = 12; // skip 6-byte magic + 6-byte unknown tail
    let num_objs = r.i32();
    for _ in 0..num_objs {
        let flag = r.read(6);
        if flag == MAGIC_MESH {
            model.nodes.push(read_mesh(&mut r, false, is_kart));
        } else if flag == MAGIC_DUMMY {
            model.nodes.push(read_dummy(&mut r, is_kart));
        } else if !is_kart && flag == MAGIC_BODY_MESH {
            let body = read_mesh(&mut r, true, false);
            model.nodes.push(body);
            model.bones = read_bone_table(&mut r);
            // skin decode deferred to M5
        }
        // unknown 6-byte markers are consumed and skipped
    }
    model
}

fn read_mesh(r: &mut Reader, is_body: bool, is_kart: bool) -> RawNode {
    let mut node = RawNode {
        is_mesh: true,
        is_body,
        scale: [1.0; 3],
        ..Default::default()
    };
    read_name_prefix(r, &mut node, is_kart);
    node.anim_entity_ids.push(node.node_id as i32);
    read_mesh_header(r, &mut node); // 138 bytes
    read_mesh_geometry(r, &mut node, is_body);
    node
}

fn read_dummy(r: &mut Reader, is_kart: bool) -> RawNode {
    let mut node = RawNode {
        scale: [1.0; 3],
        ..Default::default()
    };
    read_name_prefix(r, &mut node, is_kart);
    let dummy_id = node.node_id as i32;
    node.anim_entity_ids.push(dummy_id);
    let mut collapsed = false;
    let child_count = r.i32();
    for _ in 0..child_count {
        let flag = r.read(6);
        if flag == MAGIC_MESH {
            let mut child = read_mesh(r, false, is_kart);
            if child_count == 1 {
                // collapse a single-mesh wrapper into the mesh: keep the mesh's
                // transform, inherit the dummy's name/children, own both ids.
                if child.name.is_empty() {
                    child.name = std::mem::take(&mut node.name);
                }
                child.children = std::mem::take(&mut node.children);
                child.anim_entity_ids = vec![dummy_id, child.node_id as i32];
                node = child;
                collapsed = true;
            } else {
                node.children.push(child);
            }
        } else if flag == MAGIC_DUMMY {
            node.children.push(read_dummy(r, is_kart));
        }
    }
    // trailing 130-byte transform; when collapsed it must NOT overwrite the mesh's.
    let mut throwaway = RawNode::default();
    let target = if collapsed { &mut throwaway } else { &mut node };
    if is_kart {
        read_dummy_transform_kart(r, target);
    } else {
        read_dummy_transform_char(r, target);
    }
    node
}

fn read_name_prefix(r: &mut Reader, node: &mut RawNode, is_kart: bool) {
    node.node_id = r.i16();
    let name_count = r.i32();
    let name_bytes = r.read(name_count as usize * 2);
    node.name = if is_kart {
        let units: Vec<u16> = name_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(name_bytes).replace('\0', "")
    };
}

fn read_mesh_header(r: &mut Reader, node: &mut RawNode) {
    r.skip(4);
    node.rotation = read_rotation(r);
    node.position = r.vec3();
    node.scale = r.vec3();
    r.vec3();
    r.vec3(); // bbmin, bbmax
    r.skip(50);
}

fn read_dummy_transform_char(r: &mut Reader, node: &mut RawNode) {
    r.skip(32 + 4);
    let z = r.f32();
    let y = r.f32();
    r.skip(4);
    node.scale = r.vec3();
    r.vec3();
    r.vec3();
    r.skip(46);
    node.position = [0.0, -y, z];
}

fn read_dummy_transform_kart(r: &mut Reader, node: &mut RawNode) {
    node.rotation = read_rotation(r);
    node.position = r.vec3();
    node.scale = r.vec3();
    r.vec3();
    r.vec3();
    r.skip(46);
}

fn read_rotation(r: &mut Reader) -> Option<[f32; 9]> {
    let mut m = [0f32; 9];
    for v in m.iter_mut() {
        *v = r.f32();
    }
    let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
    let ident = near(m[0], 1.0)
        && near(m[4], 1.0)
        && near(m[8], 1.0)
        && near(m[1], 0.0)
        && near(m[2], 0.0)
        && near(m[3], 0.0)
        && near(m[5], 0.0)
        && near(m[6], 0.0)
        && near(m[7], 0.0);
    if ident {
        None
    } else {
        Some(m)
    }
}

fn read_mesh_geometry(r: &mut Reader, node: &mut RawNode, is_body: bool) {
    if r.u16() != MAGIC_VERTEX {
        return; // no geometry
    }
    node.mesh_index = r.i16() as i32;

    if is_body {
        let vc = r.i32() as usize;
        node.vertices = Vec::with_capacity(vc);
        node.normals = Vec::with_capacity(vc);
        node.bone_data = Vec::with_capacity(vc);
        for _ in 0..vc {
            node.vertices.push(r.vec3());
            node.normals.push(r.vec3());
            let mut bd = [0u8; 12];
            bd.copy_from_slice(r.read(12));
            node.bone_data.push(bd);
        }
    } else {
        let vc = r.i32() as usize;
        node.vertices = (0..vc).map(|_| r.vec3()).collect();
        let nc = r.i32() as usize;
        node.normals = (0..nc).map(|_| r.vec3()).collect();
    }

    // UV block: uv_count 12-byte entries (u, 1-v, 2 discarded i16). The face count
    // overlaps the last entry's trailing 4 bytes (rewind quirk).
    let uv_count = r.i32() as usize;
    r.skip(4);
    node.uvs = Vec::with_capacity(uv_count);
    for _ in 0..uv_count {
        let u = r.f32();
        let v = 1.0 - r.f32();
        node.uvs.push([u, v]);
        r.skip(4);
    }
    r.pos -= 4;
    let face_count = r.i32() as usize;

    // Faces: 10x i16 each — uv idx[3], adjacency[3], vertex idx[3], 1 discard.
    node.vertex_indices = Vec::with_capacity(face_count * 3);
    node.uv_indices = Vec::with_capacity(face_count * 3);
    for _ in 0..face_count {
        let uv0 = r.i16() as i32;
        let uv1 = r.i16() as i32;
        let uv2 = r.i16() as i32;
        r.i16();
        r.i16();
        r.i16(); // adjacency
        let v0 = r.i16() as i32;
        let v1 = r.i16() as i32;
        let v2 = r.i16() as i32;
        r.i16(); // discard
        node.uv_indices.extend_from_slice(&[uv0, uv1, uv2]);
        node.vertex_indices.extend_from_slice(&[v0, v1, v2]);
    }
}

fn read_bone_table(r: &mut Reader) -> Vec<RawBone> {
    let count = r.i32() as usize;
    let mut bones = Vec::with_capacity(count);
    for i in 0..count {
        let mut inv_bind = [0f32; 12];
        for v in inv_bind.iter_mut() {
            *v = r.f32();
        }
        let mut local = [0f32; 12];
        for v in local.iter_mut() {
            *v = r.f32();
        }
        let parent = r.u8();
        r.skip(3);
        bones.push(RawBone {
            index: i,
            parent: if parent == 0xFF { -1 } else { parent as i32 },
            inv_bind,
            local,
        });
    }
    r.skip(1);
    bones
}

struct Reader<'a> {
    d: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Reader { d, pos: 0 }
    }
    fn u8(&mut self) -> u8 {
        let v = self.d[self.pos];
        self.pos += 1;
        v
    }
    fn i16(&mut self) -> i16 {
        let v = i16::from_le_bytes([self.d[self.pos], self.d[self.pos + 1]]);
        self.pos += 2;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.d[self.pos], self.d[self.pos + 1]]);
        self.pos += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes([
            self.d[self.pos],
            self.d[self.pos + 1],
            self.d[self.pos + 2],
            self.d[self.pos + 3],
        ]);
        self.pos += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes([
            self.d[self.pos],
            self.d[self.pos + 1],
            self.d[self.pos + 2],
            self.d[self.pos + 3],
        ]);
        self.pos += 4;
        v
    }
    fn vec3(&mut self) -> Vec3 {
        [self.f32(), self.f32(), self.f32()]
    }
    fn read(&mut self, n: usize) -> &'a [u8] {
        let s = &self.d[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn skip(&mut self, n: usize) {
        self.pos += n;
    }
}

impl RawNode {
    /// Decode the body mesh's raw 12-byte per-vertex skin records into blend
    /// weights: u16 bone0 | u16 bone1 (0xFF → -1) | f32 w0 | f32 w1. Mirrors the
    /// validated C# `DecodeSkin`. Empty unless this is the skinned body node.
    pub fn skin_weights(&self) -> Vec<SkinWeight> {
        self.bone_data
            .iter()
            .map(|bd| {
                let i0 = (bd[0] as u16 | (bd[1] as u16) << 8) as i32;
                let i1 = (bd[2] as u16 | (bd[3] as u16) << 8) as i32;
                SkinWeight {
                    index0: i0,
                    index1: if i1 == 255 { -1 } else { i1 },
                    weight0: f32::from_le_bytes(bd[4..8].try_into().unwrap()),
                    weight1: f32::from_le_bytes(bd[8..12].try_into().unwrap()),
                }
            })
            .collect()
    }
}

impl RawModel {
    /// Flatten all mesh nodes (self + descendants) for stats/iteration.
    pub fn all_nodes(&self) -> Vec<&RawNode> {
        let mut out = Vec::new();
        fn walk<'a>(n: &'a RawNode, out: &mut Vec<&'a RawNode>) {
            out.push(n);
            for c in &n.children {
                walk(c, out);
            }
        }
        for n in &self.nodes {
            walk(n, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct B(Vec<u8>);
    impl B {
        fn bytes(mut self, b: &[u8]) -> Self {
            self.0.extend_from_slice(b);
            self
        }
        fn i16(mut self, v: i16) -> Self {
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
        fn vec3(self, x: f32, y: f32, z: f32) -> Self {
            self.f32(x).f32(y).f32(z)
        }
        fn zeros(mut self, n: usize) -> Self {
            self.0.extend(std::iter::repeat_n(0u8, n));
            self
        }
    }

    /// A minimal kart model: one ReMesh node holding a single triangle.
    fn one_triangle_kart() -> Vec<u8> {
        let mut b = B(MAGIC_KART.to_vec())
            .zeros(6) // unknown tail
            .i32(1) // num_objs
            .bytes(MAGIC_MESH)
            // name prefix (kart = UTF-16): node_id, name "m"
            .i16(5)
            .i32(1)
            .bytes(&[b'm', 0])
            // 138-byte mesh header
            .zeros(4) // skip
            .vec3(1.0, 0.0, 0.0)
            .vec3(0.0, 1.0, 0.0)
            .vec3(0.0, 0.0, 1.0) // identity rotation (→ None)
            .vec3(0.0, 0.0, 0.0) // position
            .vec3(1.0, 1.0, 1.0) // scale
            .vec3(0.0, 0.0, 0.0) // bbmin
            .vec3(0.0, 0.0, 0.0) // bbmax
            .zeros(50) // property block
            // geometry
            .i16(MAGIC_VERTEX as i16)
            .i16(0) // mesh_index
            .i32(3) // vertex count
            .vec3(0.0, 0.0, 0.0)
            .vec3(1.0, 0.0, 0.0)
            .vec3(0.0, 1.0, 0.0)
            .i32(3) // normal count
            .vec3(0.0, 0.0, 1.0)
            .vec3(0.0, 0.0, 1.0)
            .vec3(0.0, 0.0, 1.0)
            // UV block: count, skip4, then 3 × (u, 1-v, trailing4). The last
            // trailing4 overlaps the face count (the rewind quirk) → put 1 there.
            .i32(3) // uv_count
            .zeros(4)
            .f32(0.0)
            .f32(0.0)
            .zeros(4)
            .f32(1.0)
            .f32(0.0)
            .zeros(4)
            .f32(0.0)
            .f32(1.0)
            .i32(1); // last trailing 4 bytes = face_count
                     // 1 face: 10 × i16 = uv_idx[3], adjacency[3], vertex_idx[3], discard
        for v in [0i16, 1, 2, 0, 0, 0, 0, 1, 2, 0] {
            b = b.i16(v);
        }
        b.0
    }

    #[test]
    fn parses_one_triangle() {
        let d = one_triangle_kart();
        assert!(is_raw_model(&d));
        let m = parse(&d);
        assert_eq!(m.asset_type, "kart");
        assert_eq!(m.nodes.len(), 1);
        let n = &m.nodes[0];
        assert_eq!(n.name, "m");
        assert!(n.is_mesh);
        assert_eq!(n.vertices.len(), 3);
        assert_eq!(n.normals.len(), 3);
        assert_eq!(n.vertices[1], [1.0, 0.0, 0.0]);
        assert_eq!(n.vertex_indices, vec![0, 1, 2]);
        assert_eq!(n.uv_indices, vec![0, 1, 2]);
        assert_eq!(n.rotation, None); // identity rotation collapses to None
        assert_eq!(n.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_non_raw_model() {
        assert!(!is_raw_model(b"\xAA\x47\x98\x05\xF9\x28")); // TrackContainer, not a model
        assert!(!is_raw_model(b"short"));
    }
}
