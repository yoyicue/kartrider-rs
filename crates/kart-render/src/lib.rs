//! Headless wgpu renderer: a raw model → a textured, shaded PNG. Auto-frames the
//! model; each mesh node is drawn with its `<mesh_index>.{png,dds}` texture
//! (sibling of the model file). A winit window comes later; headless PNG is what
//! we verify here.

pub mod pose;
pub mod texture;

use glam::{Mat4, Vec3};
use kart_fmt::raw_model::{self, RawModel};
use std::path::{Path, PathBuf};
use texture::Image;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    light: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    // x = cutoff, y = alpha-test enabled
    alpha: [f32; 4],
}

/// One triangulated, textured mesh (the unit a [`Renderer`] uploads + draws).
pub struct MeshGeo {
    verts: Vec<Vertex>,
    image: Option<Image>,
    alpha_cutoff: Option<f32>,
}

#[derive(Clone, Copy)]
enum UpAxis {
    Z,
    Y,
}

#[derive(Clone, Copy)]
pub struct RenderVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

pub struct RenderMesh {
    pub vertices: Vec<RenderVertex>,
    pub image: Option<Image>,
    pub alpha_cutoff: Option<f32>,
}

/// Build per-mesh-node triangle lists (pos+normal+uv) + each node's texture index,
/// with the model AABB. **No disk I/O** — the reusable core for both texture
/// loading (`collect`) and per-frame reposing (`Renderer::repose_model`).
fn model_prims(model: &RawModel) -> (Vec<(Vec<Vertex>, i32)>, Vec3, Vec3) {
    let mut out = Vec::new();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for n in model.all_nodes() {
        if !n.is_mesh || n.vertices.is_empty() || n.vertex_indices.is_empty() {
            continue;
        }
        let per_vertex_normal = n.normals.len() == n.vertices.len();
        let mut verts = Vec::with_capacity(n.vertex_indices.len());
        for f in 0..n.vertex_indices.len() / 3 {
            let b = f * 3;
            let vi = [
                n.vertex_indices[b],
                n.vertex_indices[b + 1],
                n.vertex_indices[b + 2],
            ];
            let ui = [
                n.uv_indices.get(b).copied().unwrap_or(0),
                n.uv_indices.get(b + 1).copied().unwrap_or(0),
                n.uv_indices.get(b + 2).copied().unwrap_or(0),
            ];
            if vi.iter().any(|&v| v as usize >= n.vertices.len()) {
                continue;
            }
            let p: Vec<Vec3> = vi
                .iter()
                .map(|&v| Vec3::from(n.vertices[v as usize]))
                .collect();
            let face_n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            for k in 0..3 {
                let v = vi[k] as usize;
                let nrm = if per_vertex_normal {
                    Vec3::from(n.normals[v])
                } else {
                    face_n
                };
                let uv = n.uvs.get(ui[k] as usize).copied().unwrap_or([0.0, 0.0]);
                verts.push(Vertex {
                    pos: p[k].into(),
                    normal: nrm.into(),
                    uv,
                });
                min = min.min(p[k]);
                max = max.max(p[k]);
            }
        }
        if !verts.is_empty() {
            out.push((verts, n.mesh_index));
        }
    }
    (out, min, max)
}

/// Build per-mesh-node triangle lists and resolve each node's texture (disk I/O).
fn collect(model: &RawModel, tex_dir: &Path) -> (Vec<MeshGeo>, Vec3, Vec3) {
    let (prims, min, max) = model_prims(model);
    let out = prims
        .into_iter()
        .map(|(verts, mesh_index)| {
            let image = if mesh_index >= 0 {
                texture::load(&tex_dir.join(format!("{mesh_index}.png")))
                    .or_else(|| texture::load(&tex_dir.join(format!("{mesh_index}.dds"))))
            } else {
                None
            };
            MeshGeo {
                verts,
                image,
                alpha_cutoff: None,
            }
        })
        .collect();
    (out, min, max)
}

/// Render a raw `.1s` model file to a PNG; resolves textures from the model's directory.
pub fn render_model_png(model_path: &Path, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let data = std::fs::read(model_path).map_err(|e| format!("read: {e}"))?;
    let model = raw_model::parse(&data);
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let (meshes, min, max) = collect(&model, dir);
    if meshes.is_empty() {
        return Err("model has no triangles".into());
    }
    pollster::block_on(render(meshes, min, max, width, height, UpAxis::Z))
}

/// Render an already-triangulated scene. This is the low-level entry point used
/// by audit tooling that wants to render exported GLB geometry directly.
pub fn render_meshes_png(
    meshes: Vec<RenderMesh>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    render_meshes_png_with_up(meshes, width, height, UpAxis::Z)
}

/// Render an already-triangulated glTF-style scene whose up axis is +Y.
pub fn render_meshes_y_up_png(
    meshes: Vec<RenderMesh>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    render_meshes_png_with_up(meshes, width, height, UpAxis::Y)
}

/// Like [`render_meshes_y_up_png`], but the camera frames the explicit AABB
/// `[frame_min, frame_max]` instead of the meshes' full extent. All meshes still
/// render — this lets a caller frame on a robust / outlier-trimmed box so a
/// single stray vertex can't push the whole asset off-screen.
pub fn render_meshes_y_up_png_framed(
    meshes: Vec<RenderMesh>,
    frame_min: [f32; 3],
    frame_max: [f32; 3],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for mesh in meshes {
        let verts: Vec<Vertex> = mesh
            .vertices
            .iter()
            .map(|v| Vertex {
                pos: v.pos,
                normal: v.normal,
                uv: v.uv,
            })
            .collect();
        if !verts.is_empty() {
            out.push(MeshGeo {
                verts,
                image: mesh.image,
                alpha_cutoff: mesh.alpha_cutoff,
            });
        }
    }
    if out.is_empty() {
        return Err("scene has no triangles".into());
    }
    pollster::block_on(render(
        out,
        Vec3::from(frame_min),
        Vec3::from(frame_max),
        width,
        height,
        UpAxis::Y,
    ))
}

fn render_meshes_png_with_up(
    meshes: Vec<RenderMesh>,
    width: u32,
    height: u32,
    up_axis: UpAxis,
) -> Result<Vec<u8>, String> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut out = Vec::new();
    for mesh in meshes {
        let mut verts = Vec::with_capacity(mesh.vertices.len());
        for v in mesh.vertices {
            let p = Vec3::from(v.pos);
            min = min.min(p);
            max = max.max(p);
            verts.push(Vertex {
                pos: v.pos,
                normal: v.normal,
                uv: v.uv,
            });
        }
        if !verts.is_empty() {
            out.push(MeshGeo {
                verts,
                image: mesh.image,
                alpha_cutoff: mesh.alpha_cutoff,
            });
        }
    }
    if out.is_empty() {
        return Err("scene has no triangles".into());
    }
    pollster::block_on(render(out, min, max, width, height, up_axis))
}

/// Render a character `.1s` model deformed by a skeletal animation clip
/// (`fXX.1s`) at `time_ms`, to a PNG. The body mesh is CPU-skinned to the pose;
/// rigid sub-meshes stay at their bind positions.
pub fn render_posed_character_png(
    model_path: &Path,
    anim_path: &Path,
    time_ms: i32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let mdata = std::fs::read(model_path).map_err(|e| format!("read model: {e}"))?;
    let model = raw_model::parse(&mdata);
    let adata = std::fs::read(anim_path).map_err(|e| format!("read anim: {e}"))?;
    let clip = kart_fmt::anim_clip::parse(&adata, "pose").ok_or("not a character anim clip")?;
    let posed = pose::pose_model(&model, &clip, time_ms);
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let (meshes, min, max) = collect(&posed, dir);
    if meshes.is_empty() {
        return Err("model has no triangles".into());
    }
    pollster::block_on(render(meshes, min, max, width, height, UpAxis::Z))
}

/// Render a KartObject track/stage `.1s` (TrackContainer / Relement scene) to a
/// PNG. Node geometry is baked to world space via the scene hierarchy; each
/// node's `TexProperty` name resolves to `<tex_dir>/<name>.{dds,png}`. The
/// camera auto-frames the whole track.
pub fn render_track_png(
    track_path: &Path,
    tex_dir: &Path,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    render_track_png_with_tex_dirs(track_path, &[tex_dir.to_path_buf()], width, height)
}

/// Render a KartObject `.1s` with ordered texture fallbacks. This is useful for
/// virtual-mount outputs where a track can resolve textures from its registry
/// theme first, then filename-derived theme dirs, then common theme archives.
pub fn render_track_png_with_tex_dirs(
    track_path: &Path,
    tex_dirs: &[PathBuf],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let data = std::fs::read(track_path).map_err(|e| format!("read: {e}"))?;
    let track = kart_fmt::kart_object::parse(&data).map_err(|e| format!("parse: {e}"))?;
    let (meshes, min, max) = collect_track(&track, track_path, tex_dirs);
    if meshes.is_empty() {
        return Err("track has no triangles".into());
    }
    pollster::block_on(render(meshes, min, max, width, height, UpAxis::Z))
}

/// Flatten a parsed KartObject track to world-space triangulated meshes with
/// resolved textures, plus its world AABB.
fn collect_track(
    track: &kart_fmt::kart_object::Track,
    track_path: &Path,
    tex_dirs: &[PathBuf],
) -> (Vec<MeshGeo>, Vec3, Vec3) {
    let fallback_dirs;
    let tex_dirs = if tex_dirs.is_empty() {
        fallback_dirs = vec![track_path.parent().unwrap_or(Path::new(".")).to_path_buf()];
        fallback_dirs.as_slice()
    } else {
        tex_dirs
    };
    let mut tex_cache: std::collections::HashMap<String, Option<usize>> =
        std::collections::HashMap::new();
    let mut images: Vec<Image> = Vec::new();
    let mut meshes = Vec::new();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for (idx, world) in track.flatten() {
        let n = &track.nodes[idx];
        let Some(g) = &n.geom else { continue };
        if g.vertices.is_empty() || g.indices.is_empty() {
            continue;
        }
        let m = Mat4::from_cols_array(&world);
        let normal_mat = Mat4::from_cols_array(&world); // rotation+scale; translation ignored for vectors
                                                        // expand the (flat-list or strip) index buffer into triangles
        let tri_indices: Vec<[u32; 3]> = if g.strip {
            (0..g.indices.len().saturating_sub(2))
                .map(|i| {
                    if i % 2 == 0 {
                        [g.indices[i], g.indices[i + 1], g.indices[i + 2]]
                    } else {
                        [g.indices[i + 1], g.indices[i], g.indices[i + 2]]
                    }
                })
                .collect()
        } else {
            g.indices
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect()
        };
        let mut verts = Vec::with_capacity(tri_indices.len() * 3);
        let per_vertex_normal = g.normals.len() == g.vertices.len();
        for tri in &tri_indices {
            if tri.iter().any(|&v| v as usize >= g.vertices.len()) {
                continue;
            }
            let wp: Vec<Vec3> = tri
                .iter()
                .map(|&v| m.transform_point3(Vec3::from(g.vertices[v as usize])))
                .collect();
            let face_n = (wp[1] - wp[0]).cross(wp[2] - wp[0]).normalize_or_zero();
            for k in 0..3 {
                let vi = tri[k] as usize;
                let nrm = if per_vertex_normal {
                    normal_mat
                        .transform_vector3(Vec3::from(g.normals[vi]))
                        .normalize_or_zero()
                } else {
                    face_n
                };
                let uv = g.uvs.get(vi).copied().unwrap_or([0.0, 0.0]);
                verts.push(Vertex {
                    pos: wp[k].into(),
                    normal: nrm.into(),
                    uv: [uv[0], 1.0 - uv[1]],
                });
                min = min.min(wp[k]);
                max = max.max(wp[k]);
            }
        }
        if verts.is_empty() {
            continue;
        }
        // resolve texture (cached): <name>.dds then <name>.png
        let image_idx = if let Some(name) = &n.tex {
            *tex_cache.entry(name.clone()).or_insert_with(|| {
                let img = tex_dirs.iter().find_map(|dir| {
                    texture::load(&dir.join(format!("{name}.dds")))
                        .or_else(|| texture::load(&dir.join(format!("{name}.png"))))
                });
                img.map(|im| {
                    images.push(im);
                    images.len() - 1
                })
            })
        } else {
            None
        };
        meshes.push((verts, image_idx));
    }

    let mesh_geo: Vec<MeshGeo> = meshes
        .into_iter()
        .map(|(verts, img_idx)| MeshGeo {
            verts,
            image: img_idx.map(|i| images[i].clone()),
            alpha_cutoff: None,
        })
        .collect();
    (mesh_geo, min, max)
}

// ============================ reusable GPU renderer ========================

/// Re-exported so downstream crates (e.g. a viewer) build wgpu targets with the
/// exact wgpu version this crate links against.
pub use wgpu;

/// Directional light used by the built-in shading (matches the headless PNGs).
pub const DEFAULT_LIGHT: [f32; 4] = [0.4, -0.6, 0.7, 0.0];

/// A triangulated, textured scene plus its world AABB (for auto-framing).
pub struct Scene {
    pub meshes: Vec<MeshGeo>,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Scene {
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
    pub fn radius(&self) -> f32 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * 0.5).max(1e-3)
    }
}

/// Parse a raw-magic model `.1s` into a scene (textures from the model's dir).
pub fn scene_from_model(path: &Path) -> Result<Scene, String> {
    let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let model = raw_model::parse(&data);
    let dir = path.parent().unwrap_or(Path::new("."));
    scene_of_model(&model, dir)
}

/// Build a scene from an already-parsed (possibly posed) raw model — used for
/// live skeletal-animation playback (`pose::pose_model` per frame).
pub fn scene_of_model(model: &RawModel, tex_dir: &Path) -> Result<Scene, String> {
    let (meshes, min, max) = collect(model, tex_dir);
    if meshes.is_empty() {
        return Err("model has no triangles".into());
    }
    Ok(Scene {
        meshes,
        min: min.into(),
        max: max.into(),
    })
}

/// Parse a KartObject `.1s` (track/scene) into a scene, resolving textures from
/// `tex_dirs` in order (empty → the file's own dir).
pub fn scene_from_track(path: &Path, tex_dirs: &[PathBuf]) -> Result<Scene, String> {
    let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let track = kart_fmt::kart_object::parse(&data).map_err(|e| format!("parse: {e}"))?;
    let (meshes, min, max) = collect_track(&track, path, tex_dirs);
    if meshes.is_empty() {
        return Err("track has no triangles".into());
    }
    Ok(Scene {
        meshes,
        min: min.into(),
        max: max.into(),
    })
}

/// Load any `.1s` — raw model or KartObject — into a scene.
pub fn scene_from_file(path: &Path, tex_dirs: &[PathBuf]) -> Result<Scene, String> {
    let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    if raw_model::is_raw_model(&data) {
        scene_from_model(path)
    } else {
        scene_from_track(path, tex_dirs)
    }
}

/// Build a solid-colored square-section tube mesh along a polyline — used to
/// overlay AI/track paths (KAP) on a scene. Color is applied via a 1×1 texture
/// so it shows under the standard textured shader.
pub fn path_tube_mesh(points: &[[f32; 3]], width: f32, color: [u8; 4]) -> MeshGeo {
    let n = points.len();
    let mut rings: Vec<[Vec3; 4]> = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vec3::from(points[i]);
        let dir = (Vec3::from(points[(i + 1).min(n - 1)])
            - Vec3::from(points[i.saturating_sub(1)]))
        .normalize_or_zero();
        let axis = if dir.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
        let right = dir.cross(axis).normalize_or_zero();
        let up = right.cross(dir).normalize_or_zero();
        rings.push([
            p + (right + up) * width,
            p + (up - right) * width,
            p - (right + up) * width,
            p + (right - up) * width,
        ]);
    }
    let mut verts = Vec::new();
    for i in 0..n.saturating_sub(1) {
        for k in 0..4 {
            let k2 = (k + 1) % 4;
            let (a, b, c, d) = (rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]);
            let nrm = (b - a).cross(c - a).normalize_or_zero().into();
            for v in [a, b, c, a, c, d] {
                verts.push(Vertex {
                    pos: v.into(),
                    normal: nrm,
                    uv: [0.5, 0.5],
                });
            }
        }
    }
    MeshGeo {
        verts,
        image: Some(Image {
            width: 1,
            height: 1,
            rgba: color.to_vec(),
        }),
        alpha_cutoff: None,
    }
}

/// An uploaded, drawable mesh (owns its GPU buffers + bind group).
pub struct GpuMesh {
    vbuf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    count: u32,
}

/// A persistent GPU renderer: the shared pipeline + a camera uniform. Upload
/// meshes once, then draw them each frame to any target (an offscreen texture
/// for PNG, or a window surface). Colour format is fixed at construction.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    mesh_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
}

impl Renderer {
    /// Depth format the renderer's pipeline expects for the depth attachment.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cam_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&cam_bgl, &mesh_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: Self::DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            mesh_bgl,
            sampler,
            cam_buf,
            cam_bg,
        }
    }

    /// Upload a scene's meshes into GPU buffers + per-mesh bind groups.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[MeshGeo],
    ) -> Vec<GpuMesh> {
        let white = Image {
            width: 1,
            height: 1,
            rgba: vec![255, 255, 255, 255],
        };
        meshes
            .iter()
            .map(|m| {
                let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&m.verts),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
                let view = upload_texture(device, queue, m.image.as_ref().unwrap_or(&white));
                let mat = MaterialUniform {
                    alpha: [
                        m.alpha_cutoff.unwrap_or(0.0),
                        if m.alpha_cutoff.is_some() { 1.0 } else { 0.0 },
                        0.0,
                        0.0,
                    ],
                };
                let mbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&mat),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.mesh_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: mbuf.as_entire_binding(),
                        },
                    ],
                });
                GpuMesh {
                    vbuf,
                    bg,
                    count: m.verts.len() as u32,
                }
            })
            .collect()
    }

    /// Update the camera uniform: `mvp` = projection·view, plus a `light` dir.
    pub fn set_camera(&self, queue: &wgpu::Queue, mvp: [[f32; 4]; 4], light: [f32; 4]) {
        queue.write_buffer(
            &self.cam_buf,
            0,
            bytemuck::bytes_of(&Uniforms { mvp, light }),
        );
    }

    /// Record draw calls for the uploaded meshes into an active render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, meshes: &'a [GpuMesh]) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cam_bg, &[]);
        for m in meshes {
            pass.set_bind_group(1, &m.bg, &[]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.draw(0..m.count, 0..1);
        }
    }

    /// Re-pose GPU meshes previously uploaded from `model`: rewrite only the
    /// vertex buffers from a new (posed) `RawModel`, reusing the existing textures
    /// / bind groups. `gpu` must be the exact `upload` result of the same model
    /// (same mesh order + vertex counts — skeletal posing changes positions, not
    /// topology). No disk I/O, no GPU resource churn — the cheap animation path.
    pub fn repose_model(&self, queue: &wgpu::Queue, gpu: &[GpuMesh], model: &RawModel) -> bool {
        let (prims, _, _) = model_prims(model);
        if gpu.len() != prims.len() {
            return false;
        }
        for (g, (verts, _)) in gpu.iter().zip(prims.iter()) {
            if g.count as usize != verts.len() {
                return false;
            }
            queue.write_buffer(&g.vbuf, 0, bytemuck::cast_slice(verts));
        }
        true
    }
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, img: &Image) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img.rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(img.width * 4),
            rows_per_image: Some(img.height),
        },
        wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&Default::default())
}

/// Auto-framing camera fitting `[min,max]` in view; returns (mvp, light).
fn auto_camera(min: Vec3, max: Vec3, aspect: f32, up_axis: UpAxis) -> ([[f32; 4]; 4], [f32; 4]) {
    let center = (min + max) * 0.5;
    let radius = ((max - min).length() * 0.5).max(1e-3);
    let (eye, up) = match up_axis {
        UpAxis::Z => (
            center + Vec3::new(radius * 1.8, -radius * 2.4, radius * 1.6),
            Vec3::Z,
        ),
        UpAxis::Y => (
            center + Vec3::new(radius * 1.8, radius * 1.6, radius * 2.4),
            Vec3::Y,
        ),
    };
    let view = Mat4::look_at_rh(eye, center, up);
    let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, radius * 0.05, radius * 20.0);
    ((proj * view).to_cols_array_2d(), DEFAULT_LIGHT)
}

async fn render(
    meshes: Vec<MeshGeo>,
    min: Vec3,
    max: Vec3,
    width: u32,
    height: u32,
    up_axis: UpAxis,
) -> Result<Vec<u8>, String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok_or("no GPU adapter")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|e| format!("device: {e}"))?;

    let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let renderer = Renderer::new(&device, color_format);
    let gpu = renderer.upload(&device, &queue, &meshes);
    let (mvp, light) = auto_camera(min, max, width as f32 / height as f32, up_axis);
    renderer.set_camera(&queue, mvp, light);

    let make_rt = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color_tex = make_rt(
        color_format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let color_view = color_tex.create_view(&Default::default());
    let depth_view = make_rt(
        Renderer::DEPTH_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
    .create_view(&Default::default());

    let bpp = 4u32;
    let unpadded = width * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.10,
                        g: 0.11,
                        b: 0.13,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.draw(&mut pass, &gpu);
    }
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &out_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));

    let slice = out_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().map_err(|e| format!("map: {e}"))?;
    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let s = (row * padded) as usize;
        rgba.extend_from_slice(&data[s..s + unpadded as usize]);
    }
    drop(data);
    out_buf.unmap();

    let mut png = Vec::new();
    {
        let mut e = png::Encoder::new(&mut png, width, height);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        let mut w = e.write_header().map_err(|e| format!("png: {e}"))?;
        w.write_image_data(&rgba).map_err(|e| format!("png: {e}"))?;
    }
    Ok(png)
}

const SHADER: &str = r#"
struct Uniforms { mvp: mat4x4<f32>, light: vec4<f32> };
struct Material { alpha: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var<uniform> mat: Material;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) uv: vec2<f32> };
@vertex fn vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var o: VOut;
    o.clip = u.mvp * vec4<f32>(pos, 1.0);
    o.normal = normal;
    o.uv = uv;
    return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
    let albedo = textureSample(tex, samp, in.uv);
    if mat.alpha.y > 0.5 && albedo.a < mat.alpha.x {
        discard;
    }
    let n = normalize(in.normal);
    let l = normalize(u.light.xyz);
    let d = max(abs(dot(n, l)), 0.0) * 0.7 + 0.35;
    return vec4<f32>(albedo.rgb * d, 1.0);
}
"#;
