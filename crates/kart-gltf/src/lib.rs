//! A minimal, self-contained glTF 2.0 / GLB writer — enough to export the
//! KartRider asset stack at parity with the C# `Kart1sToGltf` (SharpGLTF) tool:
//! nodes (matrix or TRS) + meshes (POSITION/NORMAL/TEXCOORD_0, optional
//! JOINTS_0/WEIGHTS_0) + materials (baseColor texture, double-side, alpha
//! mask/blend, extras) + embedded PNG images + skins (inverse-bind) + TRS
//! animations. Everything is packed into one binary chunk and emitted as a GLB.
//!
//! No external glTF dependency — the schema subset is hand-written with serde so
//! we control exactly what is produced (and keep the clean-room, minimal-deps
//! posture of the rest of the workspace).

use serde::Serialize;
use serde_json::Value;

mod model;
pub mod scene;

const F32: u32 = 5126;
const U32C: u32 = 5125;
const U16C: u32 = 5123;
const U8C: u32 = 5121;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Alpha handling for a material, mirroring glTF alphaMode.
#[derive(Clone, Copy, PartialEq)]
pub enum Alpha {
    Opaque,
    Mask(f32), // cutoff
    Blend,
}

/// Inputs for one material.
pub struct MaterialDef {
    pub name: String,
    pub base_color_image: Option<usize>, // image index from add_png
    /// Linear RGBA base-color factor (use `[1.0; 4]` for textured/neutral).
    pub base_color: [f32; 4],
    pub double_sided: bool,
    pub alpha: Alpha,
    pub unlit: bool,
    pub extras: Option<Value>,
}

/// A single mesh primitive: a vertex stream + index buffer + material.
pub struct Primitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub joints: Option<Vec<[u16; 4]>>,
    pub weights: Option<Vec<[f32; 4]>>,
    pub material: Option<usize>,
}

/// One animation sampler+channel target (path = "translation"|"rotation"|"scale").
pub struct AnimChannel {
    pub node: usize,
    pub path: &'static str,
    pub times: Vec<f32>,
    /// flattened values: 3 per key for T/S, 4 per key for R (xyzw)
    pub values: Vec<f32>,
}

pub struct Gltf {
    bin: Vec<u8>,
    buffer_views: Vec<BufferView>,
    accessors: Vec<Accessor>,
    meshes: Vec<Mesh>,
    materials: Vec<MaterialJson>,
    images: Vec<Image>,
    textures: Vec<Texture>,
    samplers: Vec<Sampler>,
    nodes: Vec<NodeJson>,
    skins: Vec<SkinJson>,
    animations: Vec<AnimationJson>,
    scene_roots: Vec<u32>,
}

impl Default for Gltf {
    fn default() -> Self {
        Self::new()
    }
}

impl Gltf {
    pub fn new() -> Self {
        Gltf {
            bin: Vec::new(),
            buffer_views: Vec::new(),
            accessors: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            nodes: Vec::new(),
            skins: Vec::new(),
            animations: Vec::new(),
            scene_roots: Vec::new(),
        }
    }

    // ---- binary / accessor plumbing ------------------------------------

    fn align(&mut self, to: usize) {
        while !self.bin.len().is_multiple_of(to) {
            self.bin.push(0);
        }
    }

    fn push_view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        self.align(4);
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.buffer_views.push(BufferView {
            buffer: 0,
            byte_offset: offset as u32,
            byte_length: bytes.len() as u32,
            target,
        });
        self.buffer_views.len() - 1
    }

    fn add_accessor(
        &mut self,
        view: usize,
        count: usize,
        ctype: u32,
        atype: &'static str,
        minmax: Option<([f32; 3], [f32; 3])>,
    ) -> usize {
        let (min, max) = match minmax {
            Some((mn, mx)) => (Some(mn.to_vec()), Some(mx.to_vec())),
            None => (None, None),
        };
        self.accessors.push(Accessor {
            buffer_view: view as u32,
            component_type: ctype,
            count: count as u32,
            type_: atype,
            min,
            max,
        });
        self.accessors.len() - 1
    }

    fn acc_vec3(&mut self, data: &[[f32; 3]], minmax: bool) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 12);
        for v in data {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, Some(ARRAY_BUFFER));
        let mm = if minmax && !data.is_empty() {
            let mut mn = [f32::INFINITY; 3];
            let mut mx = [f32::NEG_INFINITY; 3];
            for v in data {
                for i in 0..3 {
                    mn[i] = mn[i].min(v[i]);
                    mx[i] = mx[i].max(v[i]);
                }
            }
            Some((mn, mx))
        } else {
            None
        };
        self.add_accessor(view, data.len(), F32, "VEC3", mm)
    }

    fn acc_vec2(&mut self, data: &[[f32; 2]]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, Some(ARRAY_BUFFER));
        self.add_accessor(view, data.len(), F32, "VEC2", None)
    }

    fn acc_vec4(&mut self, data: &[[f32; 4]], target: Option<u32>) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 16);
        for v in data {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, target);
        self.add_accessor(view, data.len(), F32, "VEC4", None)
    }

    fn acc_joints(&mut self, data: &[[u16; 4]]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, Some(ARRAY_BUFFER));
        self.add_accessor(view, data.len(), U16C, "VEC4", None)
    }

    fn acc_indices(&mut self, data: &[u32]) -> usize {
        let max = data.iter().copied().max().unwrap_or(0);
        // pick the smallest legal component type
        if max <= u8::MAX as u32 {
            let bytes: Vec<u8> = data.iter().map(|&i| i as u8).collect();
            let view = self.push_view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
            self.add_accessor(view, data.len(), U8C, "SCALAR", None)
        } else if max <= u16::MAX as u32 {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &i in data {
                bytes.extend_from_slice(&(i as u16).to_le_bytes());
            }
            let view = self.push_view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
            self.add_accessor(view, data.len(), U16C, "SCALAR", None)
        } else {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &i in data {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            let view = self.push_view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
            self.add_accessor(view, data.len(), U32C, "SCALAR", None)
        }
    }

    fn acc_scalar_f32(&mut self, data: &[f32]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let view = self.push_view(&bytes, None);
        // animation time accessors need min/max (glTF validation), stored on 1-comp
        let (mn, mx) = (
            data.iter().cloned().fold(f32::INFINITY, f32::min),
            data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        let mut a = Accessor {
            buffer_view: view as u32,
            component_type: F32,
            count: data.len() as u32,
            type_: "SCALAR",
            min: None,
            max: None,
        };
        if !data.is_empty() {
            a.min = Some(vec![mn]);
            a.max = Some(vec![mx]);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }

    fn acc_scalar_vecn(&mut self, data: &[f32], comps: usize, atype: &'static str) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let view = self.push_view(&bytes, None);
        self.add_accessor(view, data.len() / comps, F32, atype, None)
    }

    fn acc_mat4(&mut self, data: &[[f32; 16]]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 64);
        for m in data {
            for c in m {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.push_view(&bytes, None);
        self.add_accessor(view, data.len(), F32, "MAT4", None)
    }

    // ---- public builders ----------------------------------------------

    /// Embed a PNG image into the binary chunk; returns its image index.
    pub fn add_png(&mut self, png: &[u8]) -> usize {
        let view = self.push_view(png, None);
        self.images.push(Image {
            buffer_view: view as u32,
            mime_type: "image/png",
        });
        self.images.len() - 1
    }

    fn repeat_sampler(&mut self) -> usize {
        if self.samplers.is_empty() {
            self.samplers.push(Sampler {
                wrap_s: 10497,
                wrap_t: 10497,
                mag_filter: Some(9729),
                min_filter: Some(9987),
            });
        }
        0
    }

    pub fn add_material(&mut self, m: MaterialDef) -> usize {
        let base_color_texture = m.base_color_image.map(|img| {
            let sampler = self.repeat_sampler();
            self.textures.push(Texture {
                source: img as u32,
                sampler: sampler as u32,
            });
            TextureRef {
                index: (self.textures.len() - 1) as u32,
                tex_coord: 0,
            }
        });
        let (alpha_mode, alpha_cutoff) = match m.alpha {
            Alpha::Opaque => (None, None),
            Alpha::Mask(c) => (Some("MASK"), Some(c)),
            Alpha::Blend => (Some("BLEND"), None),
        };
        let mut extensions = None;
        if m.unlit {
            extensions = Some(serde_json::json!({ "KHR_materials_unlit": {} }));
        }
        self.materials.push(MaterialJson {
            name: m.name,
            pbr_metallic_roughness: Pbr {
                base_color_factor: m.base_color,
                base_color_texture,
                metallic_factor: 0.0,
                roughness_factor: 1.0,
            },
            double_sided: m.double_sided,
            alpha_mode,
            alpha_cutoff,
            extensions,
            extras: m.extras,
        });
        self.materials.len() - 1
    }

    pub fn add_mesh(&mut self, name: &str, prims: Vec<Primitive>) -> usize {
        let mut primitives = Vec::new();
        for p in prims {
            let pos = self.acc_vec3(&p.positions, true);
            let mut attrs = Attributes {
                position: pos,
                normal: None,
                texcoord_0: None,
                joints_0: None,
                weights_0: None,
            };
            if !p.normals.is_empty() {
                attrs.normal = Some(self.acc_vec3(&p.normals, false) as u32);
            }
            if !p.uvs.is_empty() {
                attrs.texcoord_0 = Some(self.acc_vec2(&p.uvs) as u32);
            }
            if let Some(j) = &p.joints {
                attrs.joints_0 = Some(self.acc_joints(j) as u32);
            }
            if let Some(w) = &p.weights {
                attrs.weights_0 = Some(self.acc_vec4(w, Some(ARRAY_BUFFER)) as u32);
            }
            let indices = self.acc_indices(&p.indices);
            primitives.push(MeshPrimitive {
                attributes: attrs,
                indices: indices as u32,
                material: p.material.map(|m| m as u32),
            });
        }
        self.meshes.push(Mesh {
            name: name.to_string(),
            primitives,
        });
        self.meshes.len() - 1
    }

    /// Add a node; returns its index. Children are node indices (added earlier).
    pub fn add_node(&mut self, node: NodeDef) -> usize {
        self.nodes.push(NodeJson {
            name: node.name,
            matrix: node.matrix.map(|m| m.to_vec()),
            translation: node.translation,
            rotation: node.rotation,
            scale: node.scale,
            mesh: node.mesh.map(|m| m as u32),
            skin: node.skin.map(|s| s as u32),
            children: if node.children.is_empty() {
                None
            } else {
                Some(node.children.iter().map(|&c| c as u32).collect())
            },
        });
        self.nodes.len() - 1
    }

    /// Overwrite a node's children after creation (joints are wired parent→child
    /// post-hoc since a bone's children always have a higher index).
    pub fn set_children(&mut self, node: usize, children: &[usize]) {
        self.nodes[node].children = if children.is_empty() {
            None
        } else {
            Some(children.iter().map(|&c| c as u32).collect())
        };
    }

    /// Attach a skin to an already-created node.
    pub fn set_node_skin(&mut self, node: usize, skin: usize) {
        self.nodes[node].skin = Some(skin as u32);
    }

    pub fn add_skin(
        &mut self,
        joints: &[usize],
        inverse_bind: &[[f32; 16]],
        skeleton: Option<usize>,
    ) -> usize {
        let ibm = self.acc_mat4(inverse_bind);
        self.skins.push(SkinJson {
            inverse_bind_matrices: ibm as u32,
            joints: joints.iter().map(|&j| j as u32).collect(),
            skeleton: skeleton.map(|s| s as u32),
        });
        self.skins.len() - 1
    }

    /// Add an animation from a set of channels (each its own sampler, LINEAR).
    pub fn add_animation(&mut self, name: &str, channels: Vec<AnimChannel>) {
        let mut samplers = Vec::new();
        let mut chans = Vec::new();
        for ch in channels {
            if ch.times.is_empty() {
                continue;
            }
            let input = self.acc_scalar_f32(&ch.times);
            let comps = if ch.path == "rotation" { 4 } else { 3 };
            let atype = if comps == 4 { "VEC4" } else { "VEC3" };
            let output = self.acc_scalar_vecn(&ch.values, comps, atype);
            let si = samplers.len();
            samplers.push(AnimSampler {
                input: input as u32,
                output: output as u32,
                interpolation: "LINEAR",
            });
            chans.push(AnimChannelJson {
                sampler: si as u32,
                target: AnimTarget {
                    node: ch.node as u32,
                    path: ch.path,
                },
            });
        }
        if chans.is_empty() {
            return;
        }
        self.animations.push(AnimationJson {
            name: name.to_string(),
            samplers,
            channels: chans,
        });
    }

    pub fn set_scene_roots(&mut self, roots: &[usize]) {
        self.scene_roots = roots.iter().map(|&r| r as u32).collect();
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    // ---- emit ----------------------------------------------------------

    fn to_json(&self) -> Root {
        let mut ext_used = Vec::new();
        if self.materials.iter().any(|m| m.extensions.is_some()) {
            ext_used.push("KHR_materials_unlit".to_string());
        }
        Root {
            asset: Asset {
                version: "2.0",
                generator: "kart-gltf",
            },
            scene: if self.scene_roots.is_empty() {
                None
            } else {
                Some(0)
            },
            scenes: vec![SceneJson {
                nodes: self.scene_roots.clone(),
            }],
            nodes: self.nodes.clone(),
            meshes: self.meshes.clone(),
            materials: self.materials.clone(),
            textures: self.textures.clone(),
            images: self.images.clone(),
            samplers: self.samplers.clone(),
            skins: self.skins.clone(),
            animations: self.animations.clone(),
            accessors: self.accessors.clone(),
            buffer_views: self.buffer_views.clone(),
            buffers: vec![Buffer {
                byte_length: self.bin.len() as u32,
            }],
            extensions_used: ext_used,
        }
    }

    /// Serialize to a GLB byte vector.
    pub fn into_glb(mut self) -> Vec<u8> {
        // pad BIN to 4 bytes
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let json = serde_json::to_vec(&self.to_json()).expect("glTF json");
        let mut json_chunk = json;
        while !json_chunk.len().is_multiple_of(4) {
            json_chunk.push(b' ');
        }
        let total = 12 + 8 + json_chunk.len() + 8 + self.bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        // JSON chunk
        out.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        out.extend_from_slice(&json_chunk);
        // BIN chunk
        out.extend_from_slice(&(self.bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
        out.extend_from_slice(&self.bin);
        out
    }
}

/// Public node construction input.
#[derive(Default)]
pub struct NodeDef {
    pub name: Option<String>,
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub scale: Option<[f32; 3]>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub children: Vec<usize>,
}

// re-export serde structs for to_json use
use model::{
    Accessor, AnimChannelJson, AnimSampler, AnimTarget, AnimationJson, Asset, Attributes, Buffer,
    BufferView, Image, MaterialJson, Mesh, MeshPrimitive, NodeJson, Pbr, Root, Sampler, SceneJson,
    SkinJson, Texture, TextureRef,
};

#[allow(dead_code)]
fn _serialize_assert<T: Serialize>(_: &T) {}

// Sibling-clip discovery + animation loading moved to `kart_fmt::loader` — it is
// an asset-tree concern, not a glTF one, so every backend can share it without
// depending on this crate.
