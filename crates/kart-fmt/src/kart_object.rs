//! Clean-room parser for KartRider's **KartObject protocol** — the marker-based
//! object serialization used by track/stage `.1s` files (`TrackContainer`,
//! `skydome.1s`, etc.), distinct from the raw-magic character/kart model format.
//!
//! The stream is a DAG of typed objects with NO length prefixes, so every class
//! that can appear must be consumed byte-exactly or the stream desyncs. Object
//! and field nodes are interned by a file-provided index and may be replaced by
//! a back-reference (`0x47BB` object, `0x27BB` field). Layouts were reverse-
//! engineered as format facts (see docs/kartrider-1s-format.md §3) and validated
//! byte-for-byte against the reference C# converter's output.
//!
//! What we keep: the Relement scene tree (name, transform, position, scale),
//! mesh geometry from `ReTriList`/`ReTriStrip` `VertexData`, each node's
//! `TexProperty` (texture name + UV controller curves) and `AlphaProperty`, and
//! retained `PRSTontroller` keyframes (track scenery animation). Cross-validated
//! against both the C# converter and the oracle-local karttools.

use std::collections::HashMap;

pub type Vec3 = [f32; 3];
pub type Vec2 = [f32; 2];
/// A FloatTontroller scalar curve: keyframes `(time_ms, value, tangent_out,
/// tangent_in)` plus the controller's interpolation `kf_type` (1 = linear,
/// 3 = step/NoEasing, otherwise cubic-Hermite — same family as
/// `keyframe::FloatAxis`).
#[derive(Clone, Debug, Default)]
pub struct FloatCurve {
    pub kf_type: i32,
    pub keys: Vec<(i32, f32, f32, f32)>,
}

/// A TexProperty's UV-scroll/scale controller curves: `(param_name, curve)`.
pub type UvCurves = Vec<(&'static str, FloatCurve)>;

// ---- class stamps: Adler32 of the UTF-8 class name -------------------------

// KartRider seeds the running sum with adler=0 (a=0, b=0), not the standard 1.
const fn adler32(s: &[u8]) -> u32 {
    let (mut a, mut b) = (0u32, 0u32);
    let mut i = 0;
    while i < s.len() {
        a = (a + s[i] as u32) % 65521;
        b = (b + a) % 65521;
        i += 1;
    }
    (b << 16) | a
}

const TRACK_CONTAINER: u32 = adler32(b"TrackContainer");
const RELEMENT: u32 = adler32(b"Relement");
const RE_TRI_LIST: u32 = adler32(b"ReTriList");
const RE_TRI_STRIP: u32 = adler32(b"ReTriStrip");
const RE_BILLBOARD: u32 = adler32(b"ReBillboard");
const RE_CAMERA: u32 = adler32(b"ReCamera");
const RE_KART: u32 = adler32(b"ReKart");
const RE_TOON_RIGID: u32 = adler32(b"ReToonRigid");
const ALPHA_PROPERTY: u32 = adler32(b"AlphaProperty");
const BACKFACE_PROPERTY: u32 = adler32(b"BackFaceProperty");
const FOG_PROPERTY: u32 = adler32(b"FogProperty");
const MTL_PROPERTY: u32 = adler32(b"MtlProperty");
const TEX_PROPERTY: u32 = adler32(b"TexProperty");
const TOON_PROPERTY: u32 = adler32(b"ToonProperty");
const WIRE_PROPERTY: u32 = adler32(b"WireProperty");
const ZBUF_PROPERTY: u32 = adler32(b"ZBufProperty");
const VIS_TONTROLLER: u32 = adler32(b"VisTontroller");
const PRS_TONTROLLER: u32 = adler32(b"PRSTontroller");
const FLOAT_TONTROLLER: u32 = adler32(b"FloatTontroller");
const COLOR_TONTROLLER: u32 = adler32(b"ColorTontroller");
const MORPH_TONTROLLER: u32 = adler32(b"MorphTontroller");

// ---- public output ---------------------------------------------------------

#[derive(Clone)]
pub struct Geom {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>, // per-vertex (texcoord set 0)
    pub indices: Vec<u32>,
    pub strip: bool, // ReTriStrip: indices form a triangle strip
}

/// AlphaProperty fields needed to map to a glTF alpha mode.
#[derive(Clone, Copy)]
pub struct AlphaInfo {
    pub use_blend: bool,
    pub use_alpha_test: bool,
    pub alpha_ref: u8,
}

#[derive(Clone, Debug, Default)]
pub struct BinaryXmlTag {
    pub name: String,
    pub text: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<BinaryXmlTag>,
}

/// Retained PRSTontroller keyframes (track scenery animation). Rotation is only
/// produced for the ThreeAxis (type-4) form, sampled to quaternions at the union
/// of the three axes' keyframe times — exactly what the C# KartLibAnim surfaces.
#[derive(Default, Clone)]
pub struct PrsAnim {
    pub pos: Vec<(i32, Vec3)>,
    pub scale: Vec<(i32, Vec3)>,
    pub rot: Vec<(i32, [f32; 4])>,
    /// Authored ThreeAxis curves retained alongside the key-time quaternion
    /// projection. Sampling only `rot` loses complete revolutions because
    /// 0 and 2π encode equivalent endpoint quaternions and shortest-arc
    /// interpolation collapses the entire turn.
    pub rot_axes: Option<[crate::keyframe::FloatAxis; 3]>,
    /// Raw Tontroller base words. The last two words are the authored
    /// start/end time fields; retaining all words allows playback semantics to
    /// be established without reparsing or guessing discarded flags.
    pub controller_base: Option<[i32; 8]>,
    /// Six PRSTontroller-specific words following the three channel slots.
    pub controller_tail: Option<[i32; 6]>,
}

impl PrsAnim {
    pub fn has_anim(&self) -> bool {
        !self.pos.is_empty()
            || !self.scale.is_empty()
            || !self.rot.is_empty()
            || self.rot_axes.is_some()
    }

    fn channel_time(&self, time_ms: i32, range_offset: usize) -> i32 {
        let Some(base) = self.controller_base else {
            return time_ms;
        };
        let Some(tail) = self.controller_tail else {
            return time_ms;
        };
        let start = tail[range_offset];
        let end = tail[range_offset + 1];
        // KartLibrary's decoded PRSTontroller.GetPosition uses u2 == 0 as
        // repeat mode and wraps only after the authored endpoint:
        // ((t - end) % (end - start)) + start. The tail stores one start/end
        // pair per P/R/S channel; applying the same controller rule to all
        // three channels is supported by their mutually exclusive real-asset
        // ranges.
        if base[1] == 0 && time_ms > end && end > start {
            let duration = i64::from(end - start);
            let offset = (i64::from(time_ms) - i64::from(end)).rem_euclid(duration);
            return (i64::from(start) + offset) as i32;
        }
        time_ms
    }

    pub fn position_time(&self, time_ms: i32) -> i32 {
        self.channel_time(time_ms, 0)
    }

    pub fn rotation_time(&self, time_ms: i32) -> i32 {
        self.channel_time(time_ms, 2)
    }

    pub fn scale_time(&self, time_ms: i32) -> i32 {
        self.channel_time(time_ms, 4)
    }
}

#[derive(Default)]
pub struct Node {
    /// Adler32 class stamp of the concrete KartObject (`Relement`,
    /// `ReTriList`, `ReTriStrip`, ...).
    pub class_stamp: u32,
    pub name: String,
    pub transform: [f32; 9], // 3 columns of the rotation basis (file order)
    pub position: Vec3,
    pub scale: Vec3,
    pub geom: Option<Geom>,
    pub tex: Option<String>,
    pub alpha: Option<AlphaInfo>,
    pub extra_tag: Option<BinaryXmlTag>,
    pub prs: Option<PrsAnim>,
    /// UV-scroll/scale controller curves from this node's TexProperty (uObj1..5).
    pub uv_curves: UvCurves,
    pub children: Vec<usize>,
}

#[derive(Default)]
pub struct Track {
    pub nodes: Vec<Node>, // arena; children reference by index
    pub root: Option<usize>,
    /// Number of arena nodes owned by the ordinary scene graph. Event
    /// subtrees are decoded into the same arena after this prefix so their
    /// visuals can reuse the normal node decoder without changing legacy
    /// scene-only inventories and query protocols.
    pub scene_node_count: usize,
    /// TrackContainer event count (To* triggers). Like KartLibrary and the
    /// oracle, the events themselves are counted but not decoded.
    pub event_count: i32,
    /// The decoded To* trigger events (E1 of
    /// docs/kart-fmt-to-events-handoff-2026-07-30.md). Segmented from the
    /// byte stream right after `event_count` using the strictly contiguous
    /// KartObject instance ids; every event keeps its raw body bytes.
    /// See analysis/track_world/to_events_identification_factory_I02_20260730.md.
    pub events: Vec<TrackEvent>,
    /// Typed top-level parse of the same event section (E2 of the same
    /// handoff): one entry per top-level KartObject of the section, in stream
    /// order. The event section is serialized with the SAME recursive
    /// KartObject format as the scene tree — the "unnamed" stamps of the E1
    /// identification doc are exactly this crate's scene classes (Relement,
    /// ReTriList/ReTriStrip, the properties and tontrollers; adler32 equality,
    /// same evidence standard as the named To* classes). Relement-family
    /// roots link into the shared `nodes` arena; their inline children are
    /// parsed by the same decoder as scene nodes.
    pub event_roots: Vec<EventRoot>,
    /// Typed runtime placements serialized after the declared TrackContainer
    /// event table.  factory_I02 stores its third propeller-robot banana here
    /// as a ToMovableObject + Relement pair even though neither object is
    /// included in `event_count`.  Keeping these roots separate preserves the
    /// declared event-table/TSV contract while making the authored runtime
    /// object available to render and world-query consumers.
    pub post_event_roots: Vec<EventRoot>,
}

impl Track {
    /// All typed event-like roots in stream order: the declared table first,
    /// followed by recovered post-table runtime objects.
    pub fn runtime_event_roots(&self) -> impl Iterator<Item = &EventRoot> {
        self.event_roots.iter().chain(self.post_event_roots.iter())
    }
}

/// One To* trigger event from the TrackContainer event table.
#[derive(Clone, Debug)]
pub struct TrackEvent {
    /// KartObject instance id (sequential across the event table).
    pub index: i16,
    /// Class stamp (Adler32 of the class name, seed 0).
    pub stamp: u32,
    /// Resolved class name when identified with evidence (see the
    /// identification doc); None for stamps no source names.
    pub class_name: Option<&'static str>,
    /// The event's own name when its body leads with a KRString.
    pub name: Option<String>,
    /// Typed payload for ToRoad events (full layout from track_nav).
    pub to_road: Option<crate::track_nav::ToRoad>,
    /// Raw body bytes (after the 8-byte sync/stamp/id header) — always
    /// retained, decoded or not.
    pub body: Vec<u8>,
}

/// Typed placement carried by ToDummy and ToMovableObject event bodies.
/// Layout (factory_I02, all four ToMovableObject + the ToDummy 'start'
/// bodies, verified byte-exact): KRString name, u8 property gate, when the
/// gate is 1 a 0x27AA/0x27BB field holding a `property` BinaryXmlTag (e.g.
/// `object type=banana` + `sound filename=mo_프로펠라로봇`), then the
/// Relement-style transform triple (9 basis columns + position + scale).
/// Evidence for the gate byte: mo_cylinder387 has gate 0 and no field;
/// the other three movables have gate 1 + field. The ToDummy sample has
/// gate 0; the gate-1 branch is by analogy with ToMovableObject.
#[derive(Clone, Debug, Default)]
pub struct ToPlacement {
    pub name: String,
    pub property: Option<BinaryXmlTag>,
    /// 3 columns of the rotation basis (file order), like Node::transform.
    pub transform: [f32; 9],
    pub position: Vec3,
    pub scale: Vec3,
}

/// One TOP-LEVEL object of the event section (E2 typed parse). Unlike the
/// flat `TrackEvent` segmentation (which lists nested objects too), an
/// EventRoot owns its whole inline subtree.
#[derive(Clone, Debug)]
pub struct EventRoot {
    /// KartObject instance id of the root frame.
    pub index: i16,
    /// Class stamp of the root frame.
    pub stamp: u32,
    pub class_name: Option<&'static str>,
    /// Arena index (`Track::nodes`) of the parsed subtree root when this is
    /// an Relement-family object.
    pub node: Option<usize>,
    /// Typed placement for ToDummy / ToMovableObject roots.
    pub placement: Option<ToPlacement>,
}

const TO_DUMMY_STAMP: u32 = adler32(b"ToDummy");
const TO_MOVABLE_OBJECT_STAMP: u32 = adler32(b"ToMovableObject");
const TO_MINIMAP_STAMP: u32 = adler32(b"ToMinimap");
const TO_BLACK_PLANE_STAMP: u32 = adler32(b"ToBlackPlane");
const TO_ITEM_CUBE_STAMP: u32 = adler32(b"ToItemCube");

fn to_event_class_name(stamp: u32) -> Option<&'static str> {
    // Names proven by adler32 against the fixcrt35.exe string table and
    // kart-fmt's track_nav ToRoad decoder (identification doc §已命名类).
    // The scene-class arms were added in E2: the identification doc's ten
    // "unnamed" stamps are exactly this crate's scene-graph classes (same
    // adler32 evidence standard; the event section embeds full Relement
    // subtrees — meshes, properties, controllers and all).
    match stamp {
        crate::track_nav::TO_ROAD_STAMP => Some("ToRoad"),
        TO_DUMMY_STAMP => Some("ToDummy"),
        TO_MOVABLE_OBJECT_STAMP => Some("ToMovableObject"),
        TO_MINIMAP_STAMP => Some("ToMinimap"),
        TO_BLACK_PLANE_STAMP => Some("ToBlackPlane"),
        TO_ITEM_CUBE_STAMP => Some("ToItemCube"),
        RELEMENT => Some("Relement"),
        RE_TRI_LIST => Some("ReTriList"),
        RE_TRI_STRIP => Some("ReTriStrip"),
        ALPHA_PROPERTY => Some("AlphaProperty"),
        BACKFACE_PROPERTY => Some("BackFaceProperty"),
        TEX_PROPERTY => Some("TexProperty"),
        ZBUF_PROPERTY => Some("ZBufProperty"),
        PRS_TONTROLLER => Some("PRSTontroller"),
        FLOAT_TONTROLLER => Some("FloatTontroller"),
        _ => None,
    }
}

/// Segments and decodes the event table that follows `event_count`.
/// `start` is the byte offset immediately after the event_count field.
/// Segmentation relies on the strictly contiguous KartObject instance ids:
/// event k's body ends where event k+1 begins (next 0x47AA frame whose id is
/// the previous id + 1); the last event ends at the following section's first
/// frame or EOF. Malformed input degrades to a shorter best-effort list —
/// never panics and never errors out the track parse.
/// Class stamps that `Parser::decode` consumes (scene-graph classes) — used
/// by the event-section typed pass to decide between a recursive read and an
/// opaque leaf skip.
fn is_scene_class(stamp: u32) -> bool {
    matches!(
        stamp,
        RELEMENT
            | RE_TRI_LIST
            | RE_TRI_STRIP
            | RE_BILLBOARD
            | RE_CAMERA
            | RE_KART
            | RE_TOON_RIGID
            | ALPHA_PROPERTY
            | BACKFACE_PROPERTY
            | FOG_PROPERTY
            | MTL_PROPERTY
            | TEX_PROPERTY
            | TOON_PROPERTY
            | WIRE_PROPERTY
            | ZBUF_PROPERTY
            | VIS_TONTROLLER
            | PRS_TONTROLLER
            | FLOAT_TONTROLLER
            | COLOR_TONTROLLER
            | MORPH_TONTROLLER
    )
}

/// continues the strictly contiguous sequence from `start`, with the frame's
/// byte span. Shared by the flat `decode_events` (E1) and the typed
/// `read_event_roots` pass (E2). Never panics; malformed input shortens the
/// list.
struct EventSeg {
    frame_off: usize,
    stamp: u32,
    index: i16,
    /// Naive body end: the next frame's offset (or EOF). For frames with
    /// inline children this CUTS the true object short — the typed pass
    /// consumes the real span; the naive end only bounds leaf bodies.
    seg_end: usize,
}

fn segment_events(d: &[u8], start: usize, count: i32) -> Vec<EventSeg> {
    let mut segs = Vec::new();
    let mut pos = start;
    for _ in 0..count.max(0) {
        if pos + 8 > d.len() || d[pos] != 0xAA || d[pos + 1] != 0x47 {
            break; // malformed table: keep what we have
        }
        let stamp = u32::from_le_bytes([d[pos + 2], d[pos + 3], d[pos + 4], d[pos + 5]]);
        let index = i16::from_le_bytes([d[pos + 6], d[pos + 7]]);
        // Bound this event's body by the next frame (id == index + 1).
        let mut end = d.len();
        let mut scan = pos + 8;
        while scan + 8 <= d.len() {
            if d[scan] == 0xAA
                && d[scan + 1] == 0x47
                && i16::from_le_bytes([d[scan + 6], d[scan + 7]]) == index.wrapping_add(1)
            {
                end = scan;
                break;
            }
            scan += 1;
        }
        segs.push(EventSeg {
            frame_off: pos,
            stamp,
            index,
            seg_end: end,
        });
        pos = end;
    }
    segs
}

fn build_track_events(d: &[u8], segs: &[EventSeg]) -> Vec<TrackEvent> {
    segs.iter()
        .map(|seg| {
            let body = d[seg.frame_off + 8..seg.seg_end].to_vec();
            let name = parse_kr_string(&body);
            let to_road = if seg.stamp == crate::track_nav::TO_ROAD_STAMP {
                crate::track_nav::parse_to_road(d, seg.frame_off)
            } else {
                None
            };
            TrackEvent {
                index: seg.index,
                stamp: seg.stamp,
                class_name: to_event_class_name(seg.stamp),
                name,
                to_road,
                body,
            }
        })
        .collect()
}

/// KRString: i32 char-count + UTF-16LE chars, used as the leading name field
/// of most event classes. Returns None when the body does not lead with one.
fn parse_kr_string(body: &[u8]) -> Option<String> {
    if body.len() < 4 {
        return None;
    }
    let n = i32::from_le_bytes([body[0], body[1], body[2], body[3]]);
    if n <= 0 || n as usize > 64 || 4 + n as usize * 2 > body.len() {
        return None;
    }
    let units: Vec<u16> = (0..n as usize)
        .map(|i| u16::from_le_bytes([body[4 + i * 2], body[5 + i * 2]]))
        .collect();
    let s = String::from_utf16_lossy(&units);
    // Event names are printable (ascii or hangul); reject control garbage.
    if s.chars().any(|c| c.is_control()) {
        return None;
    }
    Some(s)
}

pub fn is_track(d: &[u8]) -> bool {
    if d.len() < 6 || d[0] != 0xAA || d[1] != 0x47 {
        return false;
    }
    let stamp = u32::from_le_bytes([d[2], d[3], d[4], d[5]]);
    matches!(
        stamp,
        TRACK_CONTAINER | RELEMENT | RE_TRI_LIST | RE_TRI_STRIP | RE_BILLBOARD | RE_CAMERA
    )
}

/// Whether a class stamp is one this parser implements — mirrors the C#
/// `KartObjectManager.ContainsClass`, used by the pipeline to decide if a
/// KartObject `.1s` is convertible (a known root class) vs anim/other (an
/// unregistered class, e.g. boss titanLoader animation clips).
pub fn is_known_class(stamp: u32) -> bool {
    matches!(
        stamp,
        TRACK_CONTAINER
            | RELEMENT
            | RE_TRI_LIST
            | RE_TRI_STRIP
            | RE_BILLBOARD
            | RE_CAMERA
            | RE_KART
            | RE_TOON_RIGID
            | ALPHA_PROPERTY
            | BACKFACE_PROPERTY
            | FOG_PROPERTY
            | MTL_PROPERTY
            | TEX_PROPERTY
            | TOON_PROPERTY
            | WIRE_PROPERTY
            | ZBUF_PROPERTY
            | VIS_TONTROLLER
            | PRS_TONTROLLER
            | FLOAT_TONTROLLER
            | COLOR_TONTROLLER
            | MORPH_TONTROLLER
    )
}

/// C#-`IsConvertible`-equivalent: a `.1s` is convertible if it is a raw model or
/// a KartObject whose **root** stamp is a known class.
pub fn is_convertible(d: &[u8]) -> bool {
    if crate::raw_model::is_raw_model(d) {
        return true;
    }
    if d.len() < 6 || d[0] != 0xAA || d[1] != 0x47 {
        return false;
    }
    is_known_class(u32::from_le_bytes([d[2], d[3], d[4], d[5]]))
}

/// Parse a KartObject `.1s` (track/stage/skydome) into a scene tree. Returns an
/// error string at the first unknown class stamp or malformed marker.
pub fn parse(d: &[u8]) -> Result<Track, String> {
    let mut p = Parser {
        r: Reader { d, p: 0 },
        nodes: Vec::new(),
        obj_map: HashMap::new(),
        field_map: HashMap::new(),
        event_count: 0,
        events: Vec::new(),
        event_roots: Vec::new(),
        post_event_roots: Vec::new(),
        scene_node_count: 0,
        legacy_alpha_property: false,
    };
    let root = match p.read_kart_object()? {
        Cached::Node(i) => Some(i),
        Cached::ContainerScene(i) => Some(i),
        _ => None,
    };
    // Standalone models do not have a TrackContainer/event section, so their
    // entire arena is the ordinary scene prefix.
    if p.scene_node_count == 0 {
        p.scene_node_count = p.nodes.len();
    }
    Ok(Track {
        events: p.events,
        event_roots: p.event_roots,
        post_event_roots: p.post_event_roots,
        nodes: p.nodes,
        root,
        scene_node_count: p.scene_node_count,
        event_count: p.event_count,
    })
}

// ---- parser ----------------------------------------------------------------

#[derive(Clone)]
enum Cached {
    Node(usize),
    ContainerScene(usize),
    Tex(Option<String>, UvCurves, Option<BinaryXmlTag>),
    Alpha(AlphaInfo),
    Prs(PrsAnim),
    FloatCurve(FloatCurve),
    Other,
}

#[derive(Clone)]
enum FieldVal {
    Mesh(Geom),
    Text(String),
    BinaryXmlTag(BinaryXmlTag),
    Other,
}

struct Parser<'a> {
    r: Reader<'a>,
    nodes: Vec<Node>,
    obj_map: HashMap<i16, Cached>,
    field_map: HashMap<i16, FieldVal>,
    event_count: i32,
    events: Vec<TrackEvent>,
    event_roots: Vec<EventRoot>,
    post_event_roots: Vec<EventRoot>,
    scene_node_count: usize,
    legacy_alpha_property: bool,
}

impl<'a> Parser<'a> {
    /// Read one object: a new object (`0x47AA` + stamp + index + body) or a
    /// back-reference (`0x47BB` + index).
    fn read_kart_object(&mut self) -> Result<Cached, String> {
        let sync = self.r.u16();
        if sync == 0x47BB {
            let idx = self.r.i16();
            return self
                .obj_map
                .get(&idx)
                .cloned()
                .ok_or_else(|| format!("dangling obj back-ref {idx}"));
        }
        if sync != 0x47AA {
            return Err(format!("bad object sync 0x{sync:04x} @ {}", self.r.p - 2));
        }
        let stamp = self.r.u32();
        let idx = self.r.i16();
        let val = self.decode(stamp)?;
        self.obj_map.insert(idx, val.clone());
        Ok(val)
    }

    fn decode(&mut self, stamp: u32) -> Result<Cached, String> {
        match stamp {
            TRACK_CONTAINER => {
                let _name = self.r.kr_string();
                let scene = match self.read_kart_object()? {
                    Cached::Node(i) => Some(i),
                    _ => None,
                };
                self.scene_node_count = self.nodes.len();
                self.event_count = self.r.i32();
                let segs = segment_events(self.r.d, self.r.p, self.event_count);
                self.events = build_track_events(self.r.d, &segs);
                self.event_roots = self.read_event_roots(&segs);
                let post_start = self.r.p;
                self.post_event_roots = self.read_post_event_runtime_roots(post_start);
                Ok(scene.map(Cached::ContainerScene).unwrap_or(Cached::Other))
            }
            RELEMENT | RE_TRI_LIST | RE_TRI_STRIP | RE_BILLBOARD | RE_CAMERA | RE_KART
            | RE_TOON_RIGID => {
                let id = self.decode_relement(stamp)?;
                Ok(Cached::Node(id))
            }
            TEX_PROPERTY => {
                let (name, curves, tag) = self.decode_tex_property()?;
                Ok(Cached::Tex(name, curves, tag))
            }
            ALPHA_PROPERTY => {
                let (use_blend, use_alpha_test, alpha_ref) = if self.legacy_alpha_property {
                    self.r.skip(4 + 4 + 4); // src + dst + compare func
                    (true, false, self.r.u8())
                } else {
                    let use_blend = self.r.u8() == 1;
                    self.r.skip(4 + 4); // src + dst color factor
                    let use_alpha_test = self.r.u8() == 1;
                    self.r.skip(4); // compare func
                    (use_blend, use_alpha_test, self.r.u8())
                };
                Ok(Cached::Alpha(AlphaInfo {
                    use_blend,
                    use_alpha_test,
                    alpha_ref,
                }))
            }
            BACKFACE_PROPERTY => {
                self.r.skip(4);
                Ok(Cached::Other)
            }
            ZBUF_PROPERTY => {
                self.r.skip(4 + 1);
                Ok(Cached::Other)
            }
            FOG_PROPERTY => {
                self.r.skip(24);
                Ok(Cached::Other)
            }
            TOON_PROPERTY => {
                self.r.skip(38);
                Ok(Cached::Other)
            }
            WIRE_PROPERTY => {
                self.r.skip(1);
                Ok(Cached::Other)
            }
            MTL_PROPERTY => {
                self.r.skip(4 * 5 + 1 + 4); // u1..u5, u6(byte), u7
                for _ in 0..4 {
                    if self.r.u8() != 0 {
                        self.read_kart_object()?;
                    }
                }
                Ok(Cached::Other)
            }
            VIS_TONTROLLER => {
                self.tontroller_base();
                self.read_field(|p| {
                    let ty = p.r.i32();
                    let count = p.r.i32();
                    if ty == 3 {
                        p.r.skip(count as usize * 5); // Func43: i32 + u8
                    }
                    FieldVal::Other
                });
                Ok(Cached::Other)
            }
            PRS_TONTROLLER => Ok(Cached::Prs(self.decode_prs())),
            FLOAT_TONTROLLER => {
                self.tontroller_base();
                let mut curve = FloatCurve::default();
                self.read_field(|p| {
                    let ty = p.r.i32();
                    let count = p.r.i32();
                    curve.kf_type = ty;
                    let size = float_kf_size(ty);
                    for _ in 0..count {
                        let t = p.r.i32();
                        let v = p.r.f32();
                        // Key layout after (time, value): ty0 Cubic carries two
                        // f32 tangents (out, in), ty2 CubicAlt three (first two
                        // retained), Linear/NoEasing none.
                        let mut extra = [0.0f32; 3];
                        for (i, slot) in extra.iter_mut().enumerate() {
                            let offset = 8 + i * 4;
                            *slot = if offset + 4 <= size { p.r.f32() } else { 0.0 };
                        }
                        p.r.skip(size.saturating_sub(8 + 12));
                        curve.keys.push((t, v, extra[0], extra[1]));
                    }
                    FieldVal::Other
                });
                Ok(Cached::FloatCurve(curve))
            }
            COLOR_TONTROLLER => {
                self.tontroller_base();
                self.read_field(|p| {
                    let ty = p.r.i32();
                    let count = p.r.i32();
                    p.r.skip(count as usize * int_kf_size(ty));
                    FieldVal::Other
                });
                Ok(Cached::Other)
            }
            MORPH_TONTROLLER => {
                self.decode_morph();
                Ok(Cached::Other)
            }
            _ => Err(format!("unknown class stamp 0x{stamp:08x} @ {}", self.r.p)),
        }
    }

    // ---- event section typed parse (E2) ------------------------------------

    /// Typed top-level parse of the event section (see `Track::event_roots`).
    /// Walks the naive frame segments; a frame whose offset lies inside the
    /// byte span consumed by a previous root is that root's inline child, not
    /// a top-level object. Per-root errors and panics degrade to an opaque
    /// entry bounded by the naive segment end — the track parse never fails
    /// because of the event section (E1 contract: no panic on any inventory
    /// track). Back-references (object and field) resolve against the scene
    /// parse because the event section follows the scene in the stream.
    fn read_event_roots(&mut self, segs: &[EventSeg]) -> Vec<EventRoot> {
        let mut roots = Vec::new();
        let mut covered_until = segs.first().map(|s| s.frame_off).unwrap_or(0);
        for seg in segs {
            if seg.frame_off < covered_until {
                continue; // nested inside a previous root
            }
            self.r.p = seg.frame_off;
            let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.read_event_root(seg)
            }));
            match parsed {
                Ok(Ok(root)) => {
                    // Successful consumption is exact — and may legitimately
                    // run past the naive section end: the LAST event's subtree
                    // (#mo_banana07+ in factory_I02) has inline children whose
                    // ids continue into the byte region the E1 identification
                    // doc called the post-event section.
                    covered_until = covered_until.max(self.r.p);
                    if std::env::var("KART_FMT_DEBUG_EVENTS").is_ok() {
                        eprintln!(
                            "root idx={} stamp=0x{:08x} frame={} consumed_to={}",
                            seg.index, seg.stamp, seg.frame_off, self.r.p
                        );
                    }
                    roots.push(root);
                }
                _ => {
                    self.r.p = seg.seg_end;
                    covered_until = covered_until.max(seg.seg_end);
                    roots.push(EventRoot {
                        index: seg.index,
                        stamp: seg.stamp,
                        class_name: to_event_class_name(seg.stamp),
                        node: None,
                        placement: None,
                    });
                }
            }
        }
        roots
    }

    fn read_event_root(&mut self, seg: &EventSeg) -> Result<EventRoot, String> {
        let mut root = EventRoot {
            index: seg.index,
            stamp: seg.stamp,
            class_name: to_event_class_name(seg.stamp),
            node: None,
            placement: None,
        };
        match seg.stamp {
            TO_DUMMY_STAMP | TO_MOVABLE_OBJECT_STAMP => {
                let sync = self.r.u16();
                if sync != 0x47AA {
                    return Err(format!("bad event sync 0x{sync:04x} @ {}", self.r.p - 2));
                }
                let _stamp = self.r.u32();
                let _idx = self.r.i16();
                root.placement = Some(self.decode_to_placement()?);
            }
            _ if is_scene_class(seg.stamp) => {
                if let Cached::Node(i) = self.read_kart_object()? {
                    root.node = Some(i);
                }
            }
            // Self-contained leaf bodies (ToRoad's typed payload already lives
            // on the flat events; ToMinimap and the per-track 'track' settings
            // object stay opaque): the naive segment end is the true end.
            _ => self.r.p = seg.seg_end,
        }
        Ok(root)
    }

    /// Recover tightly scoped runtime-object pairs that follow the declared
    /// event table.  Some factory tracks serialize banana hazards there as a
    /// ToMovableObject immediately followed by its `#name` Relement subtree.
    /// A candidate is accepted only when the placement declares
    /// `object:type=banana`, the object ids are contiguous, and the decoded
    /// subtree name matches exactly; failed probes restore parser state.
    fn read_post_event_runtime_roots(&mut self, start: usize) -> Vec<EventRoot> {
        let mut roots = Vec::new();
        let mut scan = start;
        while scan + 8 <= self.r.d.len() {
            if self.r.d[scan] != 0xAA || self.r.d[scan + 1] != 0x47 {
                scan += 1;
                continue;
            }
            let stamp = u32::from_le_bytes([
                self.r.d[scan + 2],
                self.r.d[scan + 3],
                self.r.d[scan + 4],
                self.r.d[scan + 5],
            ]);
            if stamp != TO_MOVABLE_OBJECT_STAMP {
                scan += 1;
                continue;
            }
            let index = i16::from_le_bytes([self.r.d[scan + 6], self.r.d[scan + 7]]);
            let saved_pos = self.r.p;
            let saved_nodes = self.nodes.len();
            let saved_obj_map = self.obj_map.clone();
            let saved_field_map = self.field_map.clone();
            self.r.p = scan;
            let placement_seg = EventSeg {
                frame_off: scan,
                stamp,
                index,
                seg_end: self.r.d.len(),
            };
            let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let placement_root = self.read_event_root(&placement_seg)?;
                let placement = placement_root
                    .placement
                    .as_ref()
                    .ok_or_else(|| "post-event movable has no placement".to_string())?;
                let is_banana = placement.property.as_ref().is_some_and(|tag| {
                    tag.children.iter().any(|child| {
                        child.name == "object"
                            && child
                                .attrs
                                .iter()
                                .any(|(key, value)| key == "type" && value == "banana")
                    })
                });
                if !is_banana || self.r.p + 8 > self.r.d.len() {
                    return Err("post-event movable is not a banana pair".to_string());
                }
                let child_off = self.r.p;
                let child_stamp = u32::from_le_bytes([
                    self.r.d[child_off + 2],
                    self.r.d[child_off + 3],
                    self.r.d[child_off + 4],
                    self.r.d[child_off + 5],
                ]);
                let child_index =
                    i16::from_le_bytes([self.r.d[child_off + 6], self.r.d[child_off + 7]]);
                if self.r.d[child_off] != 0xAA
                    || self.r.d[child_off + 1] != 0x47
                    || child_index != index.wrapping_add(1)
                    || !is_scene_class(child_stamp)
                {
                    return Err("post-event banana subtree frame mismatch".to_string());
                }
                let child_seg = EventSeg {
                    frame_off: child_off,
                    stamp: child_stamp,
                    index: child_index,
                    seg_end: self.r.d.len(),
                };
                let child_root = self.read_event_root(&child_seg)?;
                let child_node = child_root
                    .node
                    .ok_or_else(|| "post-event banana subtree has no node".to_string())?;
                if self.nodes[child_node].name.to_lowercase()
                    != format!("#{}", placement.name).to_lowercase()
                {
                    return Err("post-event banana subtree name mismatch".to_string());
                }
                Ok::<_, String>((placement_root, child_root))
            }));
            match parsed {
                Ok(Ok((placement, child))) => {
                    roots.push(placement);
                    roots.push(child);
                    scan = self.r.p;
                }
                _ => {
                    self.r.p = saved_pos;
                    self.nodes.truncate(saved_nodes);
                    self.obj_map = saved_obj_map;
                    self.field_map = saved_field_map;
                    scan += 1;
                }
            }
        }
        roots
    }

    /// ToDummy / ToMovableObject body: name, property gate + optional
    /// `property` BinaryXmlTag field, transform/position/scale (see
    /// `ToPlacement` for the byte evidence).
    fn decode_to_placement(&mut self) -> Result<ToPlacement, String> {
        let name = self.r.kr_string();
        let property = if self.r.u8() == 1 {
            match self.read_field(|p| FieldVal::BinaryXmlTag(p.read_binary_xml_tag())) {
                FieldVal::BinaryXmlTag(tag) => Some(tag),
                _ => None,
            }
        } else {
            None
        };
        let mut transform = [0f32; 9];
        for v in transform.iter_mut() {
            *v = self.r.f32();
        }
        let position = self.r.vec3();
        let scale = self.r.vec3();
        Ok(ToPlacement {
            name,
            property,
            transform,
            position,
            scale,
        })
    }

    /// Relement common body + subclass tail. Returns the arena node index.
    fn decode_relement(&mut self, stamp: u32) -> Result<usize, String> {
        let name = self.r.kr_string();
        let element_count = self.r.i32();
        let mut children = Vec::with_capacity(element_count.max(0) as usize);
        for _ in 0..element_count {
            if let Cached::Node(i) = self.read_kart_object()? {
                children.push(i);
            }
        }
        // transform: 3 vectors stored as columns of the basis
        let transform = {
            let mut m = [0f32; 9];
            for v in m.iter_mut() {
                *v = self.r.f32();
            }
            m
        };
        let position = self.r.vec3();
        let scale = self.r.vec3();
        // boundingBox(24) + byte + f32 + boundingBox(24) + f32 + byte
        self.r.skip(24 + 1 + 4 + 24 + 4 + 1);

        // One legacy smoke effect stores ReTriList children under ReBillboard
        // with a compact tail: an 8-byte billboard prefix, ten zero property
        // gates plus an additional-prop zero, then VertexData directly. Guard
        // the exact byte shape so the normal Relement property contract remains
        // unchanged for every other asset.
        let compact_billboard_trilist =
            stamp == RE_TRI_LIST && self.r.looks_like_compact_billboard_trilist();
        let shifted_billboard_prs = !compact_billboard_trilist
            && stamp == RE_BILLBOARD
            && self.r.looks_like_shifted_billboard_prs();
        let shifted_empty_relement = !compact_billboard_trilist
            && !shifted_billboard_prs
            && stamp == RELEMENT
            && self.r.looks_like_shifted_empty_legacy_properties();

        // 11 optional properties, each gated by a presence byte.
        let mut tex: Option<String> = None;
        let mut alpha: Option<AlphaInfo> = None;
        let mut extra_tag: Option<BinaryXmlTag> = None;
        let mut prs: Option<PrsAnim> = None;
        let mut uv_curves = Vec::new();
        if compact_billboard_trilist {
            self.r.skip(8 + 10);
        } else {
            if shifted_billboard_prs || shifted_empty_relement {
                self.r.skip(4);
            }
            let property_slots = if shifted_billboard_prs || shifted_empty_relement {
                10
            } else {
                11
            };
            for slot in 0..property_slots {
                if self.r.u8() == 1 {
                    let prev_legacy_alpha = self.legacy_alpha_property;
                    self.legacy_alpha_property = shifted_billboard_prs && slot == 3;
                    let v = self.read_kart_object();
                    self.legacy_alpha_property = prev_legacy_alpha;
                    let v = v?;
                    match (slot, v) {
                        (1, Cached::Prs(p)) => prs = Some(p),
                        (3, Cached::Alpha(a)) => alpha = Some(a),
                        (7, Cached::Tex(name, curves, tag)) => {
                            tex = name;
                            uv_curves = curves;
                            if tag.is_some() {
                                extra_tag = tag;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        // additional BinaryXmlTag
        if self.r.u8() == 1 {
            if let FieldVal::BinaryXmlTag(tag) =
                self.read_field(|p| FieldVal::BinaryXmlTag(p.read_binary_xml_tag()))
            {
                extra_tag = Some(tag);
            }
        }

        // subclass tail
        let mut geom = None;
        match stamp {
            RE_TRI_LIST | RE_TRI_STRIP => {
                if !compact_billboard_trilist {
                    let _unknown = self.r.i32();
                }
                if let FieldVal::Mesh(g) =
                    self.read_field(|p| FieldVal::Mesh(p.read_vertex_data(stamp == RE_TRI_STRIP)))
                {
                    geom = Some(g);
                }
                self.try_read_trailing_morph();
            }
            RE_TOON_RIGID => {
                let _unknown = self.r.i32();
                if let FieldVal::Mesh(g) =
                    self.read_field(|p| FieldVal::Mesh(p.read_toon_rigid_mesh()))
                {
                    geom = Some(g);
                }
            }
            RE_BILLBOARD => {
                let _u1 = self.r.i32();
            }
            RE_CAMERA => {
                self.r.skip(1 + 4); // u1(byte) + u2(f32)
                if self.r.u8() == 1 {
                    self.read_kart_object()?;
                }
                self.r.skip(4);
                if self.r.u8() == 1 {
                    self.read_kart_object()?;
                }
                self.r.skip(4);
                if self.r.u8() == 1 {
                    self.read_kart_object()?;
                }
            }
            _ => {}
        }

        self.nodes.push(Node {
            class_stamp: stamp,
            name,
            transform,
            position,
            scale,
            geom,
            tex,
            alpha,
            extra_tag,
            prs,
            uv_curves,
            children,
        });
        Ok(self.nodes.len() - 1)
    }

    fn decode_tex_property(
        &mut self,
    ) -> Result<(Option<String>, UvCurves, Option<BinaryXmlTag>), String> {
        let _u1 = self.r.i32();
        let mut name = None;
        if self.r.u8() != 0 {
            if let FieldVal::Text(s) = self.read_field(|p| FieldVal::Text(p.r.kr_string())) {
                name = Some(s);
            }
        }
        self.r.skip(4 * 6); // u4..u9 address/filter modes
        const UV_NAMES: [&str; 5] = ["uObj1", "uObj2", "uObj3", "uObj4", "uObj5"];
        let mut curves = Vec::new();
        for nm in UV_NAMES {
            if self.r.u8() != 0 {
                if let Cached::FloatCurve(curve) = self.read_kart_object()? {
                    if !curve.keys.is_empty() {
                        curves.push((nm, curve));
                    }
                }
            }
        }
        self.r.skip(4); // u15
        if self.r.u8() != 0 {
            self.read_kart_object()?; // alpha FloatTontroller
        }
        let mut extra_tag = None;
        if self.r.u8() != 0 {
            if let FieldVal::BinaryXmlTag(tag) =
                self.read_field(|p| FieldVal::BinaryXmlTag(p.read_binary_xml_tag()))
            {
                extra_tag = Some(tag);
            }
        }
        Ok((name, curves, extra_tag))
    }

    /// VertexData: vertex/normal/uv arrays + a flat index buffer (synthesized
    /// sequential when the count slot is actually the next sync marker / would
    /// overrun). Indices address vertices and texcoord set 0 alike.
    fn read_vertex_data(&mut self, strip: bool) -> Geom {
        let vcount = self.r.i16() as usize;
        let mut vertices = vec![[0f32; 3]; vcount];
        if self.r.u8() != 0 {
            for v in vertices.iter_mut() {
                *v = self.r.vec3();
            }
        }
        let mut normals = vec![[0f32; 3]; vcount];
        if self.r.u8() != 0 {
            for n in normals.iter_mut() {
                *n = self.r.vec3();
            }
        }
        if self.r.u8() != 0 {
            self.r.skip(vcount * 4); // Unknown2 floats
        }
        let tcpv = self.r.i16() as usize;
        let mut uvs = vec![[0f32; 2]; vcount];
        for i in 0..vcount {
            for j in 0..tcpv {
                let uv = self.r.vec2();
                if j == 0 {
                    uvs[i] = uv;
                }
            }
        }
        let _u9 = self.r.u8();
        let index_count = self.r.u16();
        let is_marker = matches!(index_count, 0x47AA | 0x47BB | 0x27AA | 0x27BB);
        let remaining = self.r.d.len() - self.r.p;
        let indices: Vec<u32> = if is_marker || (index_count as usize) * 2 > remaining {
            self.r.p -= 2; // un-read the bogus count
            (0..vcount as u32).collect()
        } else {
            (0..index_count).map(|_| self.r.i16() as u32).collect()
        };
        Geom {
            vertices,
            normals,
            uvs,
            indices,
            strip,
        }
    }

    /// ReToonRigid carries an inline mesh with *decoupled* vertex/normal/uv
    /// indices per face corner (like the raw-model face table). Expand each
    /// face corner into its own vertex so the result is a flat triangle list
    /// matching the renderer's per-corner layout.
    fn read_toon_rigid_mesh(&mut self) -> Geom {
        let vc = self.r.i32().max(0) as usize;
        let verts: Vec<Vec3> = (0..vc).map(|_| self.r.vec3()).collect();
        let nc = self.r.i32().max(0) as usize;
        let norms: Vec<Vec3> = (0..nc).map(|_| self.r.vec3()).collect();
        let tc = self.r.i32().max(0) as usize;
        let texc: Vec<Vec3> = (0..tc).map(|_| self.r.vec3()).collect();
        let fc = self.r.i32().max(0) as usize;
        let mut g = Geom {
            vertices: Vec::with_capacity(fc * 3),
            normals: Vec::with_capacity(fc * 3),
            uvs: Vec::with_capacity(fc * 3),
            indices: Vec::with_capacity(fc * 3),
            strip: false,
        };
        let in_range = |i: i16, len: usize| i >= 0 && (i as usize) < len;
        for _ in 0..fc {
            let tci = [self.r.i16(), self.r.i16(), self.r.i16()];
            let ni = [self.r.i16(), self.r.i16(), self.r.i16()];
            let vi = [self.r.i16(), self.r.i16(), self.r.i16()];
            let _unknown = self.r.i16();
            // C# EmitToonRigid skips a face whose vertex index is out of range.
            if !vi.iter().all(|&i| in_range(i, vc)) {
                continue;
            }
            let p = [
                verts[vi[0] as usize],
                verts[vi[1] as usize],
                verts[vi[2] as usize],
            ];
            let use_n = nc > 0 && ni.iter().all(|&i| in_range(i, nc));
            let face_n = face_normal(p[0], p[1], p[2]);
            for k in 0..3 {
                let idx = g.vertices.len() as u32;
                g.vertices.push(p[k]);
                g.normals
                    .push(if use_n { norms[ni[k] as usize] } else { face_n });
                let uv = if tc > 0 && in_range(tci[k], tc) {
                    texc[tci[k] as usize]
                } else {
                    [0.0; 3]
                };
                g.uvs.push([uv[0], uv[1]]);
                g.indices.push(idx);
            }
        }
        g
    }

    // ---- controllers (parsed only to advance the cursor) -------------------

    fn tontroller_base(&mut self) -> [i32; 8] {
        std::array::from_fn(|_| self.r.i32()) // u1..u6, startTime, endTime
    }

    fn decode_prs(&mut self) -> PrsAnim {
        let legacy_base = self.r.looks_like_legacy_prs_base();
        let mut prs = PrsAnim::default();
        if legacy_base {
            self.r.skip(6 * 4);
        } else {
            prs.controller_base = Some(self.tontroller_base());
        }
        // position
        if self.r.u8() == 1 {
            let mut keys = Vec::new();
            self.read_field(|p| {
                read_vec3_keys(&mut p.r, &mut keys);
                FieldVal::Other
            });
            prs.pos = keys;
        }
        // rotation (only the ThreeAxis form yields keyframes, sampled to quats)
        if self.r.u8() == 1 {
            let mut rot = Vec::new();
            let mut rot_axes = None;
            self.read_field(|p| {
                read_prs_rotation(&mut p.r, &mut rot, &mut rot_axes);
                FieldVal::Other
            });
            prs.rot = rot;
            prs.rot_axes = rot_axes;
        }
        // scale
        if self.r.u8() == 1 {
            let mut keys = Vec::new();
            self.read_field(|p| {
                read_vec3_keys(&mut p.r, &mut keys);
                FieldVal::Other
            });
            prs.scale = keys;
        }
        if !legacy_base {
            prs.controller_tail = Some(std::array::from_fn(|_| self.r.i32()));
        }
        prs
    }

    fn decode_morph(&mut self) {
        self.tontroller_base();
        self.read_field(|p| {
            let count = p.r.i32();
            for _ in 0..count {
                let num_v = p.r.i16() as usize;
                // presence-gated Vector3 delta arrays until the weight-curve marker
                while p.r.peek_u8() != 0xAA {
                    if p.r.u8() != 0 {
                        p.r.skip(num_v * 12);
                    }
                }
                p.read_field(|p2| {
                    let ty = p2.r.i32();
                    let cnt = p2.r.i32();
                    p2.r.skip(cnt as usize * float_kf_size(ty));
                    FieldVal::Other
                });
            }
            let idx_count = p.r.i16() as usize;
            p.r.skip(idx_count * 2);
            FieldVal::Other
        });
    }

    /// A geometry node may carry a trailing MorphTontroller after its VertexData,
    /// with no presence byte. Only consume it when the next header is
    /// unambiguously a MorphTontroller; otherwise leave the bytes untouched.
    fn try_read_trailing_morph(&mut self) {
        let start = self.r.p;
        if self.r.d.len() - start < 8 {
            return;
        }
        let sync = u16::from_le_bytes([self.r.d[start], self.r.d[start + 1]]);
        let stamp = u32::from_le_bytes(self.r.d[start + 2..start + 6].try_into().unwrap());
        if sync == 0x47AA && stamp == MORPH_TONTROLLER {
            // We don't keep it. A malformed morph (a variant we can't model) may
            // error OR run a desynced read off the end of the buffer (a panic);
            // catch either, rewind, and leave the bytes for the host node's next
            // sibling — exactly the C# TryReadTrailing try/catch behaviour. We
            // catch_unwind locally (a deliberate probe-and-recover) but do NOT
            // touch the process-global panic hook: hook ownership belongs to the
            // caller (kart-pipeline installs a quiet one once at startup), per the
            // crate-root malformed-input contract.
            let r =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.read_kart_object()));
            if !matches!(r, Ok(Ok(_))) {
                self.r.p = start;
            }
        }
    }

    // ---- field framing -----------------------------------------------------

    fn read_field<F: FnOnce(&mut Self) -> FieldVal>(&mut self, decode: F) -> FieldVal {
        let token = self.r.u16();
        match token {
            0x27AA => {
                let idx = self.r.i16();
                let v = decode(self);
                self.field_map.insert(idx, v.clone());
                v
            }
            0x27BB => {
                let idx = self.r.i16();
                self.field_map.get(&idx).cloned().unwrap_or(FieldVal::Other)
            }
            _ => FieldVal::Other, // malformed; caller-level desync would already show
        }
    }

    fn read_binary_xml_tag(&mut self) -> BinaryXmlTag {
        let name = self.r.text();
        let text = self.r.text();
        let att_count = self.r.i32().max(0) as usize;
        let mut attrs = Vec::with_capacity(att_count);
        for _ in 0..att_count {
            attrs.push((self.r.text(), self.r.text()));
        }
        let sub_count = self.r.i32().max(0) as usize;
        let mut children = Vec::with_capacity(sub_count);
        for _ in 0..sub_count {
            children.push(self.read_binary_xml_tag());
        }
        BinaryXmlTag {
            name,
            text,
            attrs,
            children,
        }
    }
}

// ---- PRS keyframe capture / rotation sampling -----------------------------

fn read_vec3_keys(r: &mut Reader, out: &mut Vec<(i32, Vec3)>) {
    let ty = r.i32();
    let count = r.i32();
    let size = vec3_kf_size(ty);
    for _ in 0..count {
        let t = r.i32();
        let v = [r.f32(), r.f32(), r.f32()];
        r.skip(size - 16);
        out.push((t, v));
    }
}

fn read_prs_rotation(
    r: &mut Reader,
    out: &mut Vec<(i32, [f32; 4])>,
    axes_out: &mut Option<[crate::keyframe::FloatAxis; 3]>,
) {
    let ty = r.i32();
    let count = r.i32();
    match ty {
        0 | 1 | 3 => r.skip(count as usize * 20),
        2 => r.skip(count as usize * 32),
        4 => {
            let (rot, axes) = read_three_axis(r);
            *out = rot;
            *axes_out = Some(axes);
        }
        _ => {}
    }
}

/// Read a ThreeAxis rotation and sample it to (time, quaternion[xyzw]) pairs at
/// the union of the three axes' keyframe times (euler q = qZ·qY·qX). The float
/// axis read/eval is the shared `keyframe` module (also used by anim_clip).
fn read_three_axis(r: &mut Reader) -> (Vec<(i32, [f32; 4])>, [crate::keyframe::FloatAxis; 3]) {
    use crate::keyframe::{eval_float_axis, read_float_axis};
    r.skip(4 + 16); // time(i32) + 4 floats header
    let x = read_float_axis(r.d, &mut r.p);
    let y = read_float_axis(r.d, &mut r.p);
    let z = read_float_axis(r.d, &mut r.p);
    let mut times = std::collections::BTreeSet::new();
    for ax in [&x, &y, &z] {
        for k in &ax.keys {
            times.insert(k.0);
        }
    }
    let sampled = times
        .into_iter()
        .map(|t| {
            let euler = [
                eval_float_axis(&x, t as f64) as f32,
                eval_float_axis(&y, t as f64) as f32,
                eval_float_axis(&z, t as f64) as f32,
            ];
            (t, crate::kart_anim::euler_to_quat(euler))
        })
        .collect();
    (sampled, [x, y, z])
}

fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
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

// keyframe payload sizes (bytes per keyframe), by wire type tag
fn vec3_kf_size(ty: i32) -> usize {
    match ty {
        0 => 40,     // Cubic: i32 + 9 f32
        1 | 3 => 16, // Linear / NoEasing: i32 + 3 f32
        _ => 16,
    }
}
fn float_kf_size(ty: i32) -> usize {
    match ty {
        0 => 16,    // Cubic: i32 + 3 f32
        1 | 3 => 8, // Linear / NoEasing: i32 + f32
        2 => 20,    // CubicAlt / Func02: i32 + 4 f32
        _ => 8,
    }
}
fn int_kf_size(ty: i32) -> usize {
    match ty {
        0 => 16,    // Cubic: i32 + i32 + 2 f32
        1 | 3 => 8, // Linear / NoEasing: i32 + i32
        _ => 8,
    }
}

// ---- byte reader -----------------------------------------------------------

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.d[self.p];
        self.p += 1;
        v
    }
    fn peek_u8(&self) -> u8 {
        self.d[self.p]
    }
    fn i16(&mut self) -> i16 {
        let v = i16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn vec3(&mut self) -> Vec3 {
        [self.f32(), self.f32(), self.f32()]
    }
    fn vec2(&mut self) -> Vec2 {
        [self.f32(), self.f32()]
    }
    fn skip(&mut self, n: usize) {
        self.p += n;
    }
    fn looks_like_compact_billboard_trilist(&self) -> bool {
        let p = self.p;
        p + 21 <= self.d.len()
            && self.d[p..p + 8] == [0, 0, 0, 1, 0, 0, 0, 0]
            && self.d[p + 8..p + 19].iter().all(|&b| b == 0)
            && self.d[p + 19] == 0xAA
            && self.d[p + 20] == 0x27
    }
    fn looks_like_shifted_billboard_prs(&self) -> bool {
        let p = self.p;
        p + 12 <= self.d.len()
            && self.d[p..p + 4] == [0, 0, 0, 1]
            && self.d[p + 4] == 0
            && self.d[p + 5] == 1
            && self.d[p + 6] == 0xAA
            && self.d[p + 7] == 0x47
            && u32::from_le_bytes(self.d[p + 8..p + 12].try_into().unwrap()) == PRS_TONTROLLER
    }
    fn looks_like_shifted_empty_legacy_properties(&self) -> bool {
        let p = self.p;
        p + 15 == self.d.len()
            && self.d[p..p + 4] == [0, 0, 0, 1]
            && self.d[p + 4..p + 15].iter().all(|&b| b == 0)
    }
    fn looks_like_legacy_prs_base(&self) -> bool {
        let p = self.p;
        p + 29 <= self.d.len()
            && self.d[p + 24] == 1
            && self.d[p + 25] == 0xAA
            && self.d[p + 26] == 0x27
    }
    /// KRString: i32 length (chars), then length*2 bytes UTF-16LE.
    fn kr_string(&mut self) -> String {
        let len = self.i32().max(0) as usize;
        let bytes = &self.d[self.p..self.p + len * 2];
        self.p += len * 2;
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    }
    fn text(&mut self) -> String {
        self.kr_string()
    }
}

impl Track {
    /// Depth-first iterate node indices with their accumulated world transform.
    /// World follows the engine's `Scale · Transform · Translation · parent`
    /// (row-vector) composition; returned as a column-major 4x4 (`M·v`).
    pub fn flatten(&self) -> Vec<(usize, [f32; 16])> {
        let mut out = Vec::new();
        if let Some(root) = self.root {
            self.walk(root, identity4(), &mut out);
        }
        out
    }

    /// Depth-first world transforms with PRSTontroller channels sampled at
    /// ``time_ms``. Authored position/rotation/scale channels replace the
    /// corresponding base local component; omitted channels retain the base.
    pub fn flatten_at(&self, time_ms: i32) -> Vec<(usize, [f32; 16])> {
        let mut out = Vec::new();
        if let Some(root) = self.root {
            self.walk_at(root, identity4(), time_ms, &mut out);
        }
        out
    }

    /// Depth-first transforms for one arena subtree sampled at `time_ms`.
    ///
    /// Event-section Relement roots live after [`Self::scene_node_count`] and
    /// are intentionally unreachable from [`Self::root`], so
    /// [`Self::flatten_at`] cannot expose them. ToMovableObject world queries
    /// use this entry point with the matched `EventRoot::node`; the subtree's
    /// authored root transform already carries the event placement.
    pub fn flatten_subtree_at(&self, root: usize, time_ms: i32) -> Vec<(usize, [f32; 16])> {
        let mut out = Vec::new();
        if root < self.nodes.len() {
            self.walk_at(root, identity4(), time_ms, &mut out);
        }
        out
    }

    fn walk(&self, i: usize, parent: [f32; 16], out: &mut Vec<(usize, [f32; 16])>) {
        let n = &self.nodes[i];
        let local = local_matrix(&n.transform, n.position, n.scale);
        let world = mul4(parent, local); // parent * local (column-major M·v)
        out.push((i, world));
        for &c in &n.children {
            self.walk(c, world, out);
        }
    }

    fn walk_at(
        &self,
        i: usize,
        parent: [f32; 16],
        time_ms: i32,
        out: &mut Vec<(usize, [f32; 16])>,
    ) {
        let node = &self.nodes[i];
        let (transform, position, scale) = sampled_local(node, time_ms);
        let world = mul4(parent, local_matrix(&transform, position, scale));
        out.push((i, world));
        for &child in &node.children {
            self.walk_at(child, world, time_ms, out);
        }
    }
}

fn sample_vec3(keys: &[(i32, Vec3)], time_ms: i32) -> Option<Vec3> {
    let first = keys.first()?;
    if time_ms <= first.0 {
        return Some(first.1);
    }
    let last = keys.last().expect("nonempty key list");
    if time_ms >= last.0 {
        return Some(last.1);
    }
    for window in keys.windows(2) {
        let (t0, a) = window[0];
        let (t1, b) = window[1];
        if time_ms >= t0 && time_ms <= t1 {
            let alpha = (time_ms - t0) as f32 / (t1 - t0).max(1) as f32;
            return Some([
                a[0] + (b[0] - a[0]) * alpha,
                a[1] + (b[1] - a[1]) * alpha,
                a[2] + (b[2] - a[2]) * alpha,
            ]);
        }
    }
    Some(last.1)
}

fn normalize_quat_xyzw(value: [f32; 4]) -> [f32; 4] {
    let length = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if length <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [
        value[0] / length,
        value[1] / length,
        value[2] / length,
        value[3] / length,
    ]
}

fn sample_quat_xyzw(keys: &[(i32, [f32; 4])], time_ms: i32) -> Option<[f32; 4]> {
    let first = keys.first()?;
    if time_ms <= first.0 {
        return Some(normalize_quat_xyzw(first.1));
    }
    let last = keys.last().expect("nonempty key list");
    if time_ms >= last.0 {
        return Some(normalize_quat_xyzw(last.1));
    }
    for window in keys.windows(2) {
        let (t0, left_raw) = window[0];
        let (t1, right_raw) = window[1];
        if time_ms < t0 || time_ms > t1 {
            continue;
        }
        let left = normalize_quat_xyzw(left_raw);
        let mut right = normalize_quat_xyzw(right_raw);
        let mut cosine = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
        if cosine < 0.0 {
            right = [-right[0], -right[1], -right[2], -right[3]];
            cosine = -cosine;
        }
        cosine = cosine.clamp(-1.0, 1.0);
        let alpha = (time_ms - t0) as f32 / (t1 - t0).max(1) as f32;
        if cosine > 0.9995 {
            return Some(normalize_quat_xyzw([
                left[0] + (right[0] - left[0]) * alpha,
                left[1] + (right[1] - left[1]) * alpha,
                left[2] + (right[2] - left[2]) * alpha,
                left[3] + (right[3] - left[3]) * alpha,
            ]));
        }
        let angle = cosine.acos();
        let denominator = angle.sin();
        let left_weight = ((1.0 - alpha) * angle).sin() / denominator;
        let right_weight = (alpha * angle).sin() / denominator;
        return Some([
            left[0] * left_weight + right[0] * right_weight,
            left[1] * left_weight + right[1] * right_weight,
            left[2] * left_weight + right[2] * right_weight,
            left[3] * left_weight + right[3] * right_weight,
        ]);
    }
    Some(normalize_quat_xyzw(last.1))
}

fn quat_xyzw_to_transform([x, y, z, w]: [f32; 4]) -> [f32; 9] {
    let columns = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    [
        columns[0][0],
        columns[1][0],
        columns[2][0],
        columns[0][1],
        columns[1][1],
        columns[2][1],
        columns[0][2],
        columns[1][2],
        columns[2][2],
    ]
}

fn sampled_local(node: &Node, time_ms: i32) -> ([f32; 9], Vec3, Vec3) {
    let Some(prs) = node.prs.as_ref() else {
        return (node.transform, node.position, node.scale);
    };
    let rotation_time = prs.rotation_time(time_ms);
    let sampled_rotation = prs
        .rot_axes
        .as_ref()
        .map(|axes| {
            crate::kart_anim::euler_to_quat([
                crate::keyframe::eval_float_axis(&axes[0], rotation_time as f64) as f32,
                crate::keyframe::eval_float_axis(&axes[1], rotation_time as f64) as f32,
                crate::keyframe::eval_float_axis(&axes[2], rotation_time as f64) as f32,
            ])
        })
        .or_else(|| sample_quat_xyzw(&prs.rot, rotation_time));
    (
        sampled_rotation
            .map(quat_xyzw_to_transform)
            .unwrap_or(node.transform),
        sample_vec3(&prs.pos, prs.position_time(time_ms)).unwrap_or(node.position),
        sample_vec3(&prs.scale, prs.scale_time(time_ms)).unwrap_or(node.scale),
    )
}

// Build the node's local matrix as column-major (M·v) from the engine's
// row-vector form  v' = v · (S · R · T).  Transposing that gives  M = Tᵀ·Rᵀ·Sᵀ,
// i.e. translation in the last column, basis columns = the 3 stored vectors.
fn local_matrix(t: &[f32; 9], pos: Vec3, scale: Vec3) -> [f32; 16] {
    // The 3 file vectors are the rotation's columns in C#'s row-vector form; the
    // column-major (M·v) transpose puts column c = (t[c], t[3+c], t[6+c]), each
    // scaled by the matching axis scale.
    let (sx, sy, sz) = (scale[0], scale[1], scale[2]);
    [
        t[0] * sx,
        t[3] * sx,
        t[6] * sx,
        0.0,
        t[1] * sy,
        t[4] * sy,
        t[7] * sy,
        0.0,
        t[2] * sz,
        t[5] * sz,
        t[8] * sz,
        0.0,
        pos[0],
        pos[1],
        pos[2],
        1.0,
    ]
}

fn identity4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

// column-major 4x4 multiply: returns a*b (so transform = a·b applied as a·(b·v))
fn mul4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + row] * b[col * 4 + k];
            }
            o[col * 4 + row] = s;
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 6-byte KartObject header for a class whose Adler32 stamp the caller picks.
    fn header(stamp: u32) -> Vec<u8> {
        let mut d = vec![0xAA, 0x47];
        d.extend_from_slice(&stamp.to_le_bytes());
        d
    }
    fn push_i16(d: &mut Vec<u8>, v: i16) {
        d.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i32(d: &mut Vec<u8>, v: i32) {
        d.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32(d: &mut Vec<u8>, v: f32) {
        d.extend_from_slice(&v.to_le_bytes());
    }
    fn push_object_header(d: &mut Vec<u8>, stamp: u32, idx: i16) {
        d.extend_from_slice(&header(stamp));
        push_i16(d, idx);
    }

    #[test]
    fn adler32_matches_reference() {
        // KartRider's adler seeds a=b=0 (not the standard 1). Hand-check "AB":
        // a = 0+65+66 = 131; b = 65 + 131 = 196 → (196<<16)|131.
        assert_eq!(adler32(b"AB"), (196 << 16) | 131);
        assert_eq!(adler32(b""), 0);
    }

    #[test]
    fn classifies_track_and_known_classes() {
        // A real track root (TrackContainer) is both a track and convertible.
        let tc = header(TRACK_CONTAINER);
        assert!(is_track(&tc));
        assert!(is_known_class(TRACK_CONTAINER));
        assert!(is_convertible(&tc));

        // A property class is "known" (registered) but not a track root.
        assert!(is_known_class(TEX_PROPERTY));
        assert!(!is_track(&header(TEX_PROPERTY)));

        // PRSTontroller is registered (so a clip rooted at it would convert),
        // matching the C# manager, but is not a track-scene root class.
        assert!(is_known_class(PRS_TONTROLLER));
        assert!(!is_track(&header(PRS_TONTROLLER)));
    }

    #[test]
    fn rejects_unknown_and_non_kartobject() {
        // boss animation roots (e.g. titanLoader classes) are unregistered → the
        // pipeline routes them to anim/other rather than counting them as failures.
        let unknown = adler32(b"TitanLoaderAnimation");
        assert!(!is_known_class(unknown));
        assert!(!is_convertible(&header(unknown)));
        assert!(!is_track(&header(unknown)));

        // Non-`AA 47` data is never a KartObject.
        assert!(!is_track(b"\x00\x01\x02\x03\x04\x05"));
        assert!(!is_convertible(b"not a kartobject"));
        assert!(!is_convertible(b"\xAA")); // too short
    }

    #[test]
    fn raw_models_are_convertible_but_not_tracks() {
        // A raw-magic kart model header is convertible (handled by raw_model),
        // yet must not be misread as a KartObject track root.
        let kart = [0xAA, 0x47, 0x49, 0x02, 0x8C, 0x07];
        assert!(is_convertible(&kart));
        assert!(!is_track(&kart));
    }

    fn identity_node(name: &str, position: Vec3) -> Node {
        Node {
            name: name.to_owned(),
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            position,
            scale: [1.0, 1.0, 1.0],
            ..Node::default()
        }
    }

    #[test]
    fn samples_prs_position_at_requested_time() {
        let mut node = identity_node("moving", [100.0, 0.0, 0.0]);
        node.prs = Some(PrsAnim {
            pos: vec![(0, [0.0, 0.0, 0.0]), (1000, [10.0, 0.0, 0.0])],
            ..PrsAnim::default()
        });
        let track = Track {
            nodes: vec![node],
            root: Some(0),
            ..Track::default()
        };

        let sampled = track.flatten_at(500);
        assert_eq!(sampled.len(), 1);
        assert!((sampled[0].1[12] - 5.0).abs() < 1e-6);
        assert_eq!(sampled[0].1[13], 0.0);
        assert_eq!(sampled[0].1[14], 0.0);
    }

    #[test]
    fn repeats_prs_channels_with_authored_controller_ranges() {
        let mut node = identity_node("looping", [100.0, 0.0, 0.0]);
        node.prs = Some(PrsAnim {
            pos: vec![(0, [0.0, 0.0, 0.0]), (1000, [10.0, 0.0, 0.0])],
            controller_base: Some([0, 0, 0, 0, 1.0f32.to_bits() as i32, 0, 0, 1000]),
            controller_tail: Some([0, 1000, 0, 0, 0, 0]),
            ..PrsAnim::default()
        });
        let track = Track {
            nodes: vec![node],
            root: Some(0),
            ..Track::default()
        };

        assert!((track.flatten_at(1000)[0].1[12] - 10.0).abs() < 1e-6);
        assert!((track.flatten_at(1001)[0].1[12] - 0.01).abs() < 1e-6);
        assert!(track.flatten_at(2000)[0].1[12].abs() < 1e-6);
    }

    #[test]
    fn nonrepeating_prs_controller_clamps_to_last_key() {
        let mut node = identity_node("once", [0.0, 0.0, 0.0]);
        node.prs = Some(PrsAnim {
            pos: vec![(0, [0.0, 0.0, 0.0]), (1000, [10.0, 0.0, 0.0])],
            controller_base: Some([0, 1, 0, 0, 1.0f32.to_bits() as i32, 0, 0, 1000]),
            controller_tail: Some([0, 1000, 0, 0, 0, 0]),
            ..PrsAnim::default()
        });
        let track = Track {
            nodes: vec![node],
            root: Some(0),
            ..Track::default()
        };

        assert!((track.flatten_at(2000)[0].1[12] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn composes_sampled_parent_with_child_local_transform() {
        let mut parent = identity_node("platform", [0.0, 0.0, 0.0]);
        parent.children.push(1);
        parent.prs = Some(PrsAnim {
            pos: vec![(0, [0.0, 0.0, 0.0]), (1000, [10.0, 0.0, 0.0])],
            ..PrsAnim::default()
        });
        let child = identity_node("road", [0.0, 3.0, 0.0]);
        let track = Track {
            nodes: vec![parent, child],
            root: Some(0),
            ..Track::default()
        };

        let sampled = track.flatten_at(500);
        assert_eq!(sampled[1].0, 1);
        assert!((sampled[1].1[12] - 5.0).abs() < 1e-6);
        assert!((sampled[1].1[13] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn slerps_prs_rotation_and_rotates_child_offset() {
        let mut parent = identity_node("turntable", [0.0, 0.0, 0.0]);
        parent.children.push(1);
        parent.prs = Some(PrsAnim {
            rot: vec![(0, [0.0, 0.0, 0.0, 1.0]), (1000, [0.0, 0.0, 1.0, 0.0])],
            ..PrsAnim::default()
        });
        let child = identity_node("road", [1.0, 0.0, 0.0]);
        let track = Track {
            nodes: vec![parent, child],
            root: Some(0),
            ..Track::default()
        };

        let sampled = track.flatten_at(500);
        assert!(sampled[1].1[12].abs() < 1e-5);
        assert!((sampled[1].1[13] - 1.0).abs() < 1e-5);
        assert!(sampled[1].1[14].abs() < 1e-5);
    }

    #[test]
    fn samples_authored_three_axis_full_turn_without_shortest_arc_collapse() {
        use crate::keyframe::FloatAxis;

        let constant_axis = || FloatAxis {
            kf_type: 1,
            keys: vec![(0, 0.0, 0.0, 0.0), (1000, 0.0, 0.0, 0.0)],
        };
        let mut parent = identity_node("full-turn", [0.0, 0.0, 0.0]);
        parent.children.push(1);
        parent.prs = Some(PrsAnim {
            // The endpoint quaternion is equivalent to the start and would
            // collapse to identity if it alone were shortest-arc slerped.
            rot: vec![(0, [0.0, 0.0, 0.0, 1.0]), (1000, [0.0, 0.0, 0.0, -1.0])],
            rot_axes: Some([
                constant_axis(),
                constant_axis(),
                FloatAxis {
                    kf_type: 1,
                    keys: vec![(0, 0.0, 0.0, 0.0), (1000, std::f64::consts::TAU, 0.0, 0.0)],
                },
            ]),
            ..PrsAnim::default()
        });
        let child = identity_node("tooth", [1.0, 0.0, 0.0]);
        let track = Track {
            nodes: vec![parent, child],
            root: Some(0),
            ..Track::default()
        };

        let midpoint = track.flatten_at(500);
        assert!((midpoint[1].1[12] + 1.0).abs() < 1e-5);
        assert!(midpoint[1].1[13].abs() < 1e-5);
        let quarter = track.flatten_at(250);
        assert!(quarter[1].1[12].abs() < 1e-5);
        assert!((quarter[1].1[13] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parses_compact_billboard_trilist_variant() {
        let mut d = Vec::new();
        push_object_header(&mut d, RE_TRI_LIST, 0);
        push_i32(&mut d, 0); // empty name
        push_i32(&mut d, 0); // no children
        for v in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
            push_f32(&mut d, v);
        }
        for v in [0.0, 0.0, 0.0] {
            push_f32(&mut d, v); // position
        }
        for v in [1.0, 1.0, 1.0] {
            push_f32(&mut d, v); // scale
        }
        d.extend_from_slice(&[0; 24]);
        d.push(0);
        push_f32(&mut d, 0.0);
        d.extend_from_slice(&[0; 24]);
        push_f32(&mut d, 0.0);
        d.push(0);

        d.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 0]); // compact billboard prefix
        d.extend_from_slice(&[0; 10]); // compact property gates
        d.push(0); // no additional BinaryXmlTag
        d.extend_from_slice(&[0xAA, 0x27]); // VertexData field
        push_i16(&mut d, 0);
        push_i16(&mut d, 0); // vertex count
        d.extend_from_slice(&[0, 0, 0]); // no positions/normals/unknown2
        push_i16(&mut d, 0); // texcoords per vertex
        d.push(0); // u9
        push_i16(&mut d, 0); // index count

        let track = parse(&d).expect("compact ReTriList should parse");
        assert_eq!(track.nodes.len(), 1);
        assert!(track.nodes[0].geom.is_some());
    }
}

#[cfg(test)]
mod event_tests {
    use super::*;

    fn factory() -> Option<Track> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../analysis/rail/factory_I02.track.1s"
        );
        let data = std::fs::read(path).ok()?;
        parse(&data).ok()
    }

    #[test]
    fn factory_event_table_decodes_all_events() {
        let Some(track) = factory() else {
            eprintln!("skipping: factory_I02 track asset not present");
            return;
        };
        assert_eq!(track.event_count, 115);
        assert_eq!(track.events.len(), 115);
        // Sequential ids across the whole table.
        for (i, ev) in track.events.iter().enumerate() {
            assert_eq!(ev.index, 1720 + i as i16);
        }
        // Named classes identified by the evidence table: the E1 To* names
        // plus the scene-graph classes the E2 typed parse matched by adler32
        // (every event except the per-track 'track' settings object).
        let named = track
            .events
            .iter()
            .filter(|ev| ev.class_name.is_some())
            .count();
        assert_eq!(named, 114, "all but the unnamed 0x1953044c 'track' object");
        // The ToDummy start event keeps its name and full raw body.
        let dummy = track
            .events
            .iter()
            .find(|ev| ev.class_name == Some("ToDummy"))
            .expect("ToDummy event");
        assert_eq!(dummy.name.as_deref(), Some("start"));
        assert_eq!(dummy.body.len(), 75);
        // ToRoad events carry the typed payload.
        assert_eq!(
            track
                .events
                .iter()
                .filter(|ev| ev.to_road.is_some())
                .count(),
            20
        );
        // E2 typed roots: 20 ToRoad + ToMinimap + 'track' + ToDummy +
        // 4 ToMovableObject + 4 '#mo_' Relement subtrees = 31 top-level
        // objects; the remaining 84 frames are their inline children.
        assert_eq!(track.event_roots.len(), 31);
        let relement_roots = track
            .event_roots
            .iter()
            .filter(|r| r.node.is_some())
            .count();
        assert_eq!(relement_roots, 4, "the four '#mo_' movable subtrees");
        assert_eq!(
            track.post_event_roots.len(),
            2,
            "factory_I02 post-table banana placement + subtree"
        );
        let banana08 = track
            .post_event_roots
            .iter()
            .find(|root| root.index == 2150)
            .and_then(|root| root.placement.as_ref())
            .expect("post-table mo_banana08+ placement");
        assert_eq!(banana08.name, "mo_banana08+");
        assert_eq!(banana08.position, [208.499_28, 507.754_36, 72.481_57]);
        let banana08_node = track
            .post_event_roots
            .iter()
            .find(|root| root.index == 2151)
            .and_then(|root| root.node)
            .expect("post-table #mo_banana08+ subtree");
        assert_eq!(track.nodes[banana08_node].name, "#mo_banana08+");
        assert!(track.nodes[banana08_node]
            .prs
            .as_ref()
            .is_some_and(PrsAnim::has_anim));
        assert!(
            track.scene_node_count < track.nodes.len(),
            "event visual nodes must follow the ordinary scene arena prefix"
        );
        assert!(
            track
                .event_roots
                .iter()
                .filter_map(|root| root.node)
                .all(|index| index >= track.scene_node_count),
            "event roots must not enter the legacy scene-node inventory"
        );
        let movables: Vec<_> = track
            .event_roots
            .iter()
            .filter_map(|r| r.placement.as_ref())
            .collect();
        // 4 ToMovableObject + the ToDummy 'start'.
        assert_eq!(movables.len(), 5);
        let gear = track
            .event_roots
            .iter()
            .filter_map(|r| r.placement.as_ref())
            .find(|p| p.name == "mo_dummy01-톱니바퀴a")
            .expect("gear movable placement");
        assert!((gear.position[0] - 158.49).abs() < 0.01);
        assert!((gear.position[1] - 831.38).abs() < 0.01);
        assert!((gear.position[2] - 68.26).abs() < 0.01);
        // Every movable names its '#'-prefixed subtree root in the arena.
        for root in track.event_roots.iter().filter(|r| r.node.is_some()) {
            let node = &track.nodes[root.node.unwrap()];
            assert!(node.name.starts_with("#mo_"), "{}", node.name);
        }
    }

    #[test]
    fn factory_event_subtree_has_independent_time_sampled_world_walk() {
        let Some(track) = factory() else {
            eprintln!("skipping: factory_I02 track asset not present");
            return;
        };
        let root = track
            .event_roots
            .iter()
            .filter_map(|root| root.node)
            .find(|&index| track.nodes[index].name.to_lowercase() == "#mo_dummy01-톱니바퀴a")
            .expect("gear event subtree");
        assert!(
            track.flatten_at(0).iter().all(|(index, _)| *index != root),
            "event roots must remain outside the ordinary scene walk"
        );
        let sampled = track.flatten_subtree_at(root, 0);
        assert!(sampled.len() > 1);
        assert_eq!(sampled[0].0, root);
        let world = sampled[0].1;
        assert!((world[12] - 158.490_26).abs() < 0.001);
        assert!((world[13] - 831.380_6).abs() < 0.001);
        assert!((world[14] - 68.262_83).abs() < 0.001);
    }
}
