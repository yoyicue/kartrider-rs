//! Hand-written serde structs for the glTF 2.0 JSON subset we emit. Field names
//! follow the spec (camelCase); empty collections are omitted so the output
//! stays small and validator-clean.

use serde::Serialize;
use serde_json::Value;

#[derive(Serialize, Clone)]
pub struct Root {
    pub asset: Asset,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scenes: Vec<SceneJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<NodeJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<Mesh>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<Texture>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub samplers: Vec<Sampler>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skins: Vec<SkinJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub animations: Vec<AnimationJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accessors: Vec<Accessor>,
    #[serde(rename = "bufferViews", skip_serializing_if = "Vec::is_empty")]
    pub buffer_views: Vec<BufferView>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<Buffer>,
    #[serde(rename = "extensionsUsed", skip_serializing_if = "Vec::is_empty")]
    pub extensions_used: Vec<String>,
}

#[derive(Serialize, Clone)]
pub struct Asset {
    pub version: &'static str,
    pub generator: &'static str,
}

#[derive(Serialize, Clone)]
pub struct SceneJson {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<u32>,
}

#[derive(Serialize, Clone)]
pub struct NodeJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translation: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skin: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<u32>>,
}

#[derive(Serialize, Clone)]
pub struct Mesh {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub primitives: Vec<MeshPrimitive>,
}

#[derive(Serialize, Clone)]
pub struct MeshPrimitive {
    pub attributes: Attributes,
    pub indices: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<u32>,
}

#[derive(Serialize, Clone)]
pub struct Attributes {
    #[serde(rename = "POSITION")]
    pub position: usize,
    #[serde(rename = "NORMAL", skip_serializing_if = "Option::is_none")]
    pub normal: Option<u32>,
    #[serde(rename = "TEXCOORD_0", skip_serializing_if = "Option::is_none")]
    pub texcoord_0: Option<u32>,
    #[serde(rename = "JOINTS_0", skip_serializing_if = "Option::is_none")]
    pub joints_0: Option<u32>,
    #[serde(rename = "WEIGHTS_0", skip_serializing_if = "Option::is_none")]
    pub weights_0: Option<u32>,
}

#[derive(Serialize, Clone)]
pub struct MaterialJson {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(rename = "pbrMetallicRoughness")]
    pub pbr_metallic_roughness: Pbr,
    #[serde(rename = "doubleSided", skip_serializing_if = "is_false")]
    pub double_sided: bool,
    #[serde(rename = "alphaMode", skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<&'static str>,
    #[serde(rename = "alphaCutoff", skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

#[derive(Serialize, Clone)]
pub struct Pbr {
    #[serde(rename = "baseColorFactor")]
    pub base_color_factor: [f32; 4],
    #[serde(rename = "baseColorTexture", skip_serializing_if = "Option::is_none")]
    pub base_color_texture: Option<TextureRef>,
    #[serde(rename = "metallicFactor")]
    pub metallic_factor: f32,
    #[serde(rename = "roughnessFactor")]
    pub roughness_factor: f32,
}

#[derive(Serialize, Clone)]
pub struct TextureRef {
    pub index: u32,
    #[serde(rename = "texCoord", skip_serializing_if = "is_zero")]
    pub tex_coord: u32,
}

#[derive(Serialize, Clone)]
pub struct Texture {
    pub source: u32,
    pub sampler: u32,
}

#[derive(Serialize, Clone)]
pub struct Image {
    #[serde(rename = "bufferView")]
    pub buffer_view: u32,
    #[serde(rename = "mimeType")]
    pub mime_type: &'static str,
}

#[derive(Serialize, Clone)]
pub struct Sampler {
    #[serde(rename = "wrapS")]
    pub wrap_s: u32,
    #[serde(rename = "wrapT")]
    pub wrap_t: u32,
    #[serde(rename = "magFilter", skip_serializing_if = "Option::is_none")]
    pub mag_filter: Option<u32>,
    #[serde(rename = "minFilter", skip_serializing_if = "Option::is_none")]
    pub min_filter: Option<u32>,
}

#[derive(Serialize, Clone)]
pub struct SkinJson {
    #[serde(rename = "inverseBindMatrices")]
    pub inverse_bind_matrices: u32,
    pub joints: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skeleton: Option<u32>,
}

#[derive(Serialize, Clone)]
pub struct AnimationJson {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub samplers: Vec<AnimSampler>,
    pub channels: Vec<AnimChannelJson>,
}

#[derive(Serialize, Clone)]
pub struct AnimSampler {
    pub input: u32,
    pub output: u32,
    pub interpolation: &'static str,
}

#[derive(Serialize, Clone)]
pub struct AnimChannelJson {
    pub sampler: u32,
    pub target: AnimTarget,
}

#[derive(Serialize, Clone)]
pub struct AnimTarget {
    pub node: u32,
    pub path: &'static str,
}

#[derive(Serialize, Clone)]
pub struct Accessor {
    #[serde(rename = "bufferView")]
    pub buffer_view: u32,
    #[serde(rename = "componentType")]
    pub component_type: u32,
    pub count: u32,
    #[serde(rename = "type")]
    pub type_: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<Vec<f32>>,
}

#[derive(Serialize, Clone)]
pub struct BufferView {
    pub buffer: u32,
    #[serde(rename = "byteOffset")]
    pub byte_offset: u32,
    #[serde(rename = "byteLength")]
    pub byte_length: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<u32>,
}

#[derive(Serialize, Clone)]
pub struct Buffer {
    #[serde(rename = "byteLength")]
    pub byte_length: u32,
}

fn is_false(b: &bool) -> bool {
    !*b
}
fn is_zero(v: &u32) -> bool {
    *v == 0
}
