//! Interactive viewer for KartRider `.1s` assets (route A).
//!
//! `kart-viewer <asset.1s|dir> [tex-dir]... [--kap <file.kap>...]`
//!
//! Controls:
//!   - drag            orbit / look
//!   - scroll          zoom (orbit) · none (fly)
//!   - F               toggle orbit / free-fly camera
//!   - W A S D / Q E   move (free-fly)
//!   - P               toggle AI-path (KAP) overlay
//!   - Space           play / pause skeletal animation (boned characters)
//!   - , / .           previous / next animation clip
//!   - N / B           next / previous asset in the folder
//!
//! Reuses `kart_render::Renderer` (the same pipeline the headless PNGs validate),
//! driven per-frame against the window surface.

use glam::{Mat4, Vec3};
use kart_fmt::anim_clip::AnimClip;
use kart_fmt::raw_model::{self, RawModel};
use kart_render::pose::pose_model;
use kart_render::wgpu;
use kart_render::{path_tube_mesh, GpuMesh, MeshGeo, Renderer, Scene, DEFAULT_LIGHT};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Live skeletal-animation state for an animatable (boned) character model.
struct Anim {
    model: RawModel,
    clips: Vec<AnimClip>,
    idx: usize,
    time_ms: i32,
    playing: bool,
    last: Instant,
    repose_warned: bool,
}
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const PALETTE: &[[u8; 4]] = &[
    [235, 40, 40, 255],
    [40, 235, 70, 255],
    [70, 110, 235, 255],
    [235, 210, 40, 255],
    [40, 235, 235, 255],
    [235, 60, 235, 255],
];

enum Mode {
    Orbit,
    Fly,
}

/// Shared camera: orbit around a target, or free-fly from a position. `dir` is
/// the unit vector from the look point toward the eye (yaw around +Z, pitch up).
struct Camera {
    mode: Mode,
    target: Vec3,
    dist: f32,
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32, // scene radius, for step/zoom scaling
}

impl Camera {
    fn fit(scene: &Scene) -> Self {
        let r = scene.radius();
        Camera {
            mode: Mode::Orbit,
            target: Vec3::from(scene.center()),
            dist: r * 3.2,
            pos: Vec3::ZERO,
            yaw: -0.9,
            pitch: 0.5,
            radius: r,
        }
    }

    fn back_dir(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(cp * self.yaw.cos(), cp * self.yaw.sin(), self.pitch.sin())
    }

    fn mvp(&self, aspect: f32) -> [[f32; 4]; 4] {
        let (eye, target) = match self.mode {
            Mode::Orbit => (self.target + self.back_dir() * self.dist, self.target),
            // fly looks along -back_dir (continuous with orbit's view direction)
            Mode::Fly => (self.pos, self.pos - self.back_dir()),
        };
        let far = (self.radius * 20.0).max(self.dist * 4.0) + 1.0;
        let view = Mat4::look_at_rh(eye, target, Vec3::Z);
        let proj = Mat4::perspective_rh(
            50f32.to_radians(),
            aspect.max(0.01),
            (self.radius * 0.01).max(1e-3),
            far,
        );
        (proj * view).to_cols_array_2d()
    }

    fn toggle_mode(&mut self) {
        match self.mode {
            Mode::Orbit => {
                self.pos = self.target + self.back_dir() * self.dist; // start fly where the orbit eye is
                self.mode = Mode::Fly;
            }
            Mode::Fly => self.mode = Mode::Orbit,
        }
    }

    /// Free-fly movement in camera space: x=right, y=forward, z=world up.
    fn fly(&mut self, right: f32, fwd: f32, up: f32) {
        let forward = -self.back_dir();
        let r = forward.cross(Vec3::Z).normalize_or_zero();
        let step = (self.radius * 0.05).max(0.05);
        self.pos += (r * right + forward * fwd + Vec3::Z * up) * step;
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    renderer: Renderer,
    scene_meshes: Vec<GpuMesh>,
    path_meshes: Vec<GpuMesh>,
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Renderer::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Decode + parse KAP files into colored, lifted path-tube meshes (world space).
/// Failures are reported (not silently dropped) so a wrong `--kap` is diagnosable.
fn path_meshes_of(kaps: &[PathBuf], radius: f32) -> Vec<MeshGeo> {
    let width = (radius * 0.01).max(1.0);
    let lift = width * 4.0;
    let mut out = Vec::new();
    for (i, p) in kaps.iter().enumerate() {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("kap {}: read failed: {e}", p.display());
                continue;
            }
        };
        let clear = match kart_rho::encoded_block::decode_len_prefixed(&bytes) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("kap {}: decode failed: {e}", p.display());
                continue;
            }
        };
        let Some(kap) = kart_fmt::kap::parse(&clear) else {
            eprintln!("kap {}: not a valid KAP payload", p.display());
            continue;
        };
        if kap.points.len() < 2 {
            eprintln!(
                "kap {}: only {} point(s), skipping",
                p.display(),
                kap.points.len()
            );
            continue;
        }
        let lifted: Vec<[f32; 3]> = kap
            .points
            .iter()
            .map(|q| [q[0], q[1], q[2] + lift])
            .collect();
        out.push(path_tube_mesh(&lifted, width, PALETTE[i % PALETTE.len()]));
    }
    out
}

struct App {
    assets: Vec<PathBuf>,
    index: usize,
    tex_dirs: Vec<PathBuf>,
    kaps: Vec<PathBuf>,
    camera: Camera,
    gpu: Option<Gpu>,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    show_paths: bool,
    anim: Option<Anim>,
}

impl App {
    fn load_current(&mut self) {
        let path = self.assets[self.index].clone();
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("read {}: {e}", path.display());
                return;
            }
        };

        // Animatable = boned raw model with sibling character clips.
        self.anim = None;
        let scene = if raw_model::is_raw_model(&data) {
            let model = raw_model::parse(&data);
            let clips = kart_fmt::loader::load_char_anims(&dir);
            let base = if !model.bones.is_empty() && !clips.is_empty() {
                pose_model(&model, &clips[0], 0)
            } else {
                model.clone()
            };
            let scene = kart_render::scene_of_model(&base, &dir);
            if !model.bones.is_empty() && !clips.is_empty() {
                self.anim = Some(Anim {
                    model,
                    clips,
                    idx: 0,
                    time_ms: 0,
                    playing: true,
                    last: Instant::now(),
                    repose_warned: false,
                });
            }
            scene
        } else {
            kart_render::scene_from_track(&path, &self.tex_dirs)
        };
        let scene = match scene {
            Ok(s) => s,
            Err(e) => {
                eprintln!("load {}: {e}", path.display());
                return;
            }
        };

        let paths = path_meshes_of(&self.kaps, scene.radius());
        self.camera = Camera::fit(&scene);
        let anims = self.anim.as_ref().map(|a| a.clips.len()).unwrap_or(0);
        if let Some(gpu) = &mut self.gpu {
            gpu.scene_meshes = gpu.renderer.upload(&gpu.device, &gpu.queue, &scene.meshes);
            gpu.path_meshes = gpu.renderer.upload(&gpu.device, &gpu.queue, &paths);
            gpu.window.set_title(&format!(
                "kart-viewer — {} [{}/{}]",
                path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                self.index + 1,
                self.assets.len()
            ));
            gpu.window.request_redraw();
        }
        println!(
            "[{}/{}] {} — {} mesh(es), {} AI path(s), {} anim clip(s)",
            self.index + 1,
            self.assets.len(),
            path.display(),
            scene.meshes.len(),
            paths.len(),
            anims
        );
    }

    /// Advance skeletal animation and re-pose the GPU meshes. Only the vertex
    /// buffers are rewritten (textures/bind groups are reused) — no disk I/O or
    /// GPU resource churn per frame. Returns true while a clip is playing.
    fn tick_anim(&mut self) -> bool {
        let posed = {
            let Some(a) = &mut self.anim else {
                return false;
            };
            if !a.playing || a.clips.is_empty() {
                return false;
            }
            let now = Instant::now();
            let dt = (now - a.last).as_millis() as i32;
            a.last = now;
            let dur = (a.clips[a.idx].duration_ms as i32).max(1);
            a.time_ms = (a.time_ms + dt).rem_euclid(dur);
            pose_model(&a.model, &a.clips[a.idx], a.time_ms)
        };
        if let Some(gpu) = &self.gpu {
            let ok = gpu
                .renderer
                .repose_model(&gpu.queue, &gpu.scene_meshes, &posed);
            if !ok {
                if let Some(a) = &mut self.anim {
                    if !a.repose_warned {
                        let label = self
                            .assets
                            .get(self.index)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "<unknown>".to_string());
                        eprintln!("animation re-pose skipped for {label}: mesh topology or vertex count changed; keeping previous GPU buffers");
                        a.repose_warned = true;
                    }
                }
            }
        }
        true
    }

    fn render(&mut self) {
        let playing = self.tick_anim();
        let Some(gpu) = &self.gpu else { return };
        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        gpu.renderer
            .set_camera(&gpu.queue, self.camera.mvp(aspect), DEFAULT_LIGHT);

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                    view: &gpu.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            gpu.renderer.draw(&mut pass, &gpu.scene_meshes);
            if self.show_paths {
                gpu.renderer.draw(&mut pass, &gpu.path_meshes);
            }
        }
        gpu.queue.submit(Some(enc.finish()));
        frame.present();
        if playing {
            gpu.window.request_redraw(); // keep the animation loop running
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if let Some(gpu) = &mut self.gpu {
            if w > 0 && h > 0 {
                gpu.config.width = w;
                gpu.config.height = h;
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.depth = make_depth(&gpu.device, w, h);
            }
        }
    }

    fn on_key(&mut self, code: KeyCode) {
        let mut redraw = true;
        match code {
            KeyCode::KeyF => self.camera.toggle_mode(),
            KeyCode::KeyP => self.show_paths = !self.show_paths,
            KeyCode::Space => {
                if let Some(a) = &mut self.anim {
                    a.playing = !a.playing;
                    a.last = Instant::now();
                }
            }
            KeyCode::Comma | KeyCode::Period => {
                if let Some(a) = &mut self.anim {
                    let n = a.clips.len();
                    a.idx = if code == KeyCode::Comma {
                        (a.idx + n - 1) % n
                    } else {
                        (a.idx + 1) % n
                    };
                    a.time_ms = 0;
                    a.last = Instant::now();
                    println!("clip {}/{}: {}", a.idx + 1, n, a.clips[a.idx].name);
                }
            }
            KeyCode::KeyW => self.camera.fly(0.0, 1.0, 0.0),
            KeyCode::KeyS => self.camera.fly(0.0, -1.0, 0.0),
            KeyCode::KeyA => self.camera.fly(-1.0, 0.0, 0.0),
            KeyCode::KeyD => self.camera.fly(1.0, 0.0, 0.0),
            KeyCode::KeyE => self.camera.fly(0.0, 0.0, 1.0),
            KeyCode::KeyQ => self.camera.fly(0.0, 0.0, -1.0),
            KeyCode::KeyN => {
                self.index = (self.index + 1) % self.assets.len();
                self.load_current();
                return;
            }
            KeyCode::KeyB => {
                self.index = (self.index + self.assets.len() - 1) % self.assets.len();
                self.load_current();
                return;
            }
            _ => redraw = false,
        }
        if redraw {
            if let Some(g) = &self.gpu {
                g.window.request_redraw();
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("kart-viewer"))
                .expect("create window"),
        );
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("request device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth = make_depth(&device, config.width, config.height);
        let renderer = Renderer::new(&device, format);
        self.gpu = Some(Gpu {
            window,
            surface,
            device,
            queue,
            config,
            depth,
            renderer,
            scene_meshes: Vec::new(),
            path_meshes: Vec::new(),
        });
        self.load_current();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size.width, size.height),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.on_key(code),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x, position.y);
                if self.dragging {
                    if let Some((lx, ly)) = self.last_cursor {
                        self.camera.yaw -= ((p.0 - lx) as f32) * 0.008;
                        self.camera.pitch =
                            (self.camera.pitch + ((p.1 - ly) as f32) * 0.008).clamp(-1.5, 1.5);
                        if let Some(g) = &self.gpu {
                            g.window.request_redraw();
                        }
                    }
                }
                self.last_cursor = Some(p);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if matches!(self.camera.mode, Mode::Orbit) {
                    let d = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                    };
                    self.camera.dist =
                        (self.camera.dist * (1.0 - d * 0.1)).clamp(1e-2, self.camera.radius * 50.0);
                    if let Some(g) = &self.gpu {
                        g.window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

/// Cheap header sniff: is this `.1s` a renderable model or KartObject scene (vs a
/// `fXX` animation clip, which has no geometry to show)?
fn is_renderable(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 16];
    let n = f.read(&mut buf).unwrap_or(0);
    let d = &buf[..n];
    raw_model::is_raw_model(d) || kart_fmt::kart_object::is_track(d)
}

fn collect_assets(path: &Path) -> Vec<PathBuf> {
    let (dir, focus) = if path.is_dir() {
        (path.to_path_buf(), None)
    } else {
        (
            path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            Some(path.to_path_buf()),
        )
    };
    // renderable .1s only — skip the sibling fXX animation clips (they load via
    // the model, not as standalone assets).
    let mut list: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("1s") && is_renderable(p))
        .collect();
    list.sort();
    // always keep the explicitly-named file, even if the sniff missed it.
    if let Some(f) = &focus {
        if !list.contains(f) {
            list.insert(0, f.clone());
        }
    }
    list
}

fn main() {
    let mut it = std::env::args().skip(1);
    let Some(path) = it.next() else {
        eprintln!("usage: kart-viewer <asset.1s|dir> [tex-dir]... [--kap <file.kap>...]");
        std::process::exit(2);
    };
    let (mut tex_dirs, mut kaps, mut kap_mode) = (Vec::new(), Vec::new(), false);
    for a in it {
        if a == "--kap" {
            kap_mode = true;
        } else if kap_mode {
            kaps.push(PathBuf::from(a));
        } else {
            tex_dirs.push(PathBuf::from(a));
        }
    }

    let path = PathBuf::from(&path);
    let assets = collect_assets(&path);
    let index = assets.iter().position(|p| *p == path).unwrap_or(0);
    if assets.is_empty() {
        eprintln!("no .1s assets found at {}", path.display());
        std::process::exit(1);
    }

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App {
        assets,
        index,
        tex_dirs,
        kaps,
        camera: Camera {
            mode: Mode::Orbit,
            target: Vec3::ZERO,
            dist: 1.0,
            pos: Vec3::ZERO,
            yaw: -0.9,
            pitch: 0.5,
            radius: 1.0,
        },
        gpu: None,
        dragging: false,
        last_cursor: None,
        show_paths: true,
        anim: None,
    };
    event_loop.run_app(&mut app).expect("run");
}
