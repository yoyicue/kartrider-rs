//! Typed `TrackWorld` adapter: the Rust-native implementation of the
//! `KartTrackWorld` client protocol in `scripts/kart_track_world.py`.
//!
//! P2 (performance handoff): the static-geometry queries no longer round
//! trip through the CLI TSV renderers. They call the SAME selection
//! functions in [`crate::query`] that the renderers use (one implementation
//! of triangle selection/ordering), then quantize each f32 field through
//! [`q9`] — `format!("{:.9}", x).parse::<f64>()` — which is exactly the
//! f32 -> `{:.9}` text -> f64 round trip the TSV path produced (Rust and
//! CPython both correctly-round decimal text, so same bits as
//! `float(row[key])`). Query inputs are rounded `f64 -> f32` with `as`
//! exactly like the CLI's `format!("{}", value as f32)` ->
//! `parse::<f32>()` (Display prints the shortest round-trip form, so both
//! are correctly-rounded conversions of the same real value).
//! String/integer fields and the rail/checkpoint JsonValue payloads keep
//! their previous shapes; rail contexts/entries, KAP markers, dynamic
//! state, and (on maps with animated road geometry) moving-contact
//! candidates still go through the cached TSV renderers.
//!
//! Pure-Python helpers (`_orthonormal_column_axes`,
//! `match_rail_entry_boundary`, `_rail_triangle_direction`, the checkpoint
//! and moving-contact compositions) are ported with the crate numeric
//! discipline: CPython `sum()` is Neumaier-compensated (`num::sum_f64` /
//! `num::dot3_sum`), 2-arg `min`/`max` are `num::min2_f64`/`num::max2_f64`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use kart_physics::num::{dot3_sum, max2_f64, min2_f64, sum_f64};
use kart_physics::state::JsonValue;
use kart_physics::world::{
    BananaTrigger, CheckpointRespawnCandidate, CheckpointTargetCandidate, EventObstacleContact,
    RailEntryBoundaryMatch, TakeoffLandingContact, World, WorldCapabilities, WorldGroundResult,
    WorldObbOverlap,
};

use crate::cli;
use crate::geom::{geom_triangles, swept_obb_triangle_intersection, validate_obb, xform, P3};
use crate::query::{self, TrackQueryIndex};
use crate::track::tag_attr;

const FACTORY_BIG_GEAR_POSITION: [f32; 3] = [
    f32::from_bits(0x431e_7d82),
    f32::from_bits(0x444f_d85c),
    f32::from_bits(0x4288_8692),
];

const DESERT_I08_UNLABELLED_TAKEOFF_NODE: &str = "RoadObj01@s4";
const DESERT_I08_UNLABELLED_TAKEOFF_FACE_TRI: i64 = 15;
const DESERT_I08_UNLABELLED_TAKEOFF_FACE_NORMAL: [f64; 3] = [-0.176746011, 0.001102233, 0.98425591];
const DESERT_I08_UNLABELLED_TAKEOFF_FACE_TRIANGLE: [[f64; 3]; 3] = [
    [618.901977539, 714.991333008, 43.128147125],
    [619.048339844, 674.989624023, 43.199226379],
    [619.739135742, 714.985961914, 43.278484344],
];
const DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRI: i64 = 13;
const DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_NORMAL: [f64; 3] =
    [-0.175547913, 0.001106998, 0.984470308];
const DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRIANGLE: [[f64; 3]; 3] = [
    [618.901977539, 714.991333008, 43.128147125],
    [618.297119141, 674.994262695, 43.065265656],
    [619.048339844, 674.989624023, 43.199226379],
];

// ---------------------------------------------------------------------------
// TSV parse layer (mirrors the Python parse_world_*_tsv structurings).
// ---------------------------------------------------------------------------

/// Header + rows of a TSV render, addressed by column name. Missing cells
/// read as `""`, matching how the Python parsers treat short rows.
struct TsvTable {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl TsvTable {
    fn parse(text: &str) -> Self {
        let mut lines = text.split('\n').filter(|line| !line.is_empty());
        let header = lines
            .next()
            .map(|line| line.split('\t').map(str::to_string).collect())
            .unwrap_or_default();
        let rows = lines
            .map(|line| line.split('\t').map(str::to_string).collect())
            .collect();
        Self { header, rows }
    }

    /// Only the legacy TSV reference (test module) addresses by column
    /// existence; production reads known columns.
    #[allow(dead_code)]
    fn has_column(&self, key: &str) -> bool {
        self.header.iter().any(|name| name == key)
    }

    fn get(&self, row: usize, key: &str) -> &str {
        let Some(index) = self.header.iter().position(|name| name == key) else {
            return "";
        };
        self.rows
            .get(row)
            .and_then(|fields| fields.get(index))
            .map(String::as_str)
            .unwrap_or("")
    }
}

/// `float(row[key])` — Rust and CPython both correctly-round decimal text.
fn pf(value: &str) -> f64 {
    value.parse().unwrap_or(0.0)
}

/// `int(row[key])`.
fn pi(value: &str) -> i64 {
    value.parse().unwrap_or(0)
}

fn pf3(table: &TsvTable, row: usize, keys: [&str; 3]) -> [f64; 3] {
    [
        pf(table.get(row, keys[0])),
        pf(table.get(row, keys[1])),
        pf(table.get(row, keys[2])),
    ]
}

/// The exact f64 the TSV round trip produced for an f32 field:
/// `{:.9}`-formatted text correctly-rounded back to f64 (same bits as
/// `float(text)`; `unwrap_or(0.0)` mirrors [`pf`]).
fn q9(value: f32) -> f64 {
    format!("{value:.9}").parse().unwrap_or(0.0)
}

fn q93(value: P3) -> [f64; 3] {
    [q9(value[0]), q9(value[1]), q9(value[2])]
}

/// `f64 -> f32` input rounding: bit-identical to the old argv path
/// (`format!("{}", value as f32)` round-trips through `parse::<f32>()` to
/// the same f32, because Display prints the shortest round-trip form).
fn f32x3(value: [f64; 3]) -> P3 {
    [value[0] as f32, value[1] as f32, value[2] as f32]
}

fn f32x9(value: [f64; 9]) -> [P3; 3] {
    [
        [value[0] as f32, value[1] as f32, value[2] as f32],
        [value[3] as f32, value[4] as f32, value[5] as f32],
        [value[6] as f32, value[7] as f32, value[8] as f32],
    ]
}

/// `parse_world_query_tsv` (scripts/kart_track_world.py:57-77) row shape.
/// Production keeps the struct (the rail-contact candidate builds it from
/// the in-memory selection); the TSV text parser lives in the test module
/// as the legacy reference the grid-equivalence gate compares against.
#[derive(Debug, Clone, Default)]
struct GroundRow {
    track_id: String,
    node: String,
    tri_index: i64,
    road_surface: String,
    surface_id: i64,
    road_rail: String,
    distance: f64,
    point: [f64; 3],
    normal: [f64; 3],
    triangle: Option<[[f64; 3]; 3]>,
    /// Read only by the legacy TSV reference (test module): the rail
    /// candidate never carries these columns.
    #[allow(dead_code)]
    contact_basis_44: Option<[f64; 3]>,
}

// `parse_world_sweep_tsv` / `parse_world_obb_sweep_tsv` /
// `parse_world_obb_overlaps_tsv` row shapes: moved to the test module
// (legacy TSV reference for the grid-equivalence gate); production builds
// the corresponding values from the in-memory selections instead.

/// `parse_world_rail_contexts_tsv` record (scripts/kart_track_world.py:103-149).
#[derive(Debug, Clone, Default)]
struct RailRecord {
    position: [f64; 3],
    secondary: [f64; 3],
    normal_or_side: [f64; 3],
}

// ---------------------------------------------------------------------------
// Process-level rail context/entry caches (P4): these used to be per-TrackWorld
// OnceLocks, so every fresh kart-wasm session paid ~25 ms + ~25 ms of TSV
// render+parse inside its first step. The values are pure functions of the
// track asset, so they share the query-index cache's keying and invalidation.
// ---------------------------------------------------------------------------

type RailContextCache = OnceLock<Mutex<BTreeMap<PathBuf, Option<Arc<Vec<RailContext>>>>>>;
type RailEntryCache = OnceLock<Mutex<BTreeMap<PathBuf, Option<Arc<Vec<RailEntry>>>>>>;
static RAIL_CONTEXT_CACHE: RailContextCache = OnceLock::new();
static RAIL_ENTRY_CACHE: RailEntryCache = OnceLock::new();

/// Drops the cached rail contexts/entries for `path` (virtual-bytes byte
/// swap; keyed identically to the static-track parse cache).
pub(crate) fn invalidate_rail_caches(path: &Path) {
    let key = crate::track::static_track_cache_key(path);
    if let Ok(mut guard) = RAIL_CONTEXT_CACHE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
    {
        guard.remove(path);
        guard.remove(&key);
    }
    if let Ok(mut guard) = RAIL_ENTRY_CACHE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
    {
        guard.remove(path);
        guard.remove(&key);
    }
}

fn rail_context_cache(track_path: &Path) -> Option<Arc<Vec<RailContext>>> {
    let key = crate::track::static_track_cache_key(track_path);
    let cache = RAIL_CONTEXT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(hit) = cache.lock().ok()?.get(&key).cloned() {
        return hit;
    }
    let computed = cli::render_world_rail_contexts(&[track_path.to_string_lossy().into_owned()])
        .ok()
        .map(|text| Arc::new(parse_rail_contexts(&TsvTable::parse(&text))));
    cache.lock().ok()?.insert(key, computed.clone());
    computed
}

fn rail_entry_cache(track_path: &Path) -> Option<Arc<Vec<RailEntry>>> {
    let key = crate::track::static_track_cache_key(track_path);
    let cache = RAIL_ENTRY_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(hit) = cache.lock().ok()?.get(&key).cloned() {
        return hit;
    }
    let computed = cli::render_world_rail_entries(&[track_path.to_string_lossy().into_owned()])
        .ok()
        .map(|text| Arc::new(parse_rail_entries(&TsvTable::parse(&text))));
    cache.lock().ok()?.insert(key, computed.clone());
    computed
}

#[derive(Debug, Clone, Default)]
struct RailContext {
    context_index: i64,
    road: String,
    element: i64,
    total_length: f64,
    successor_indices: Vec<i64>,
    records: Vec<RailRecord>,
}

fn parse_rail_contexts(table: &TsvTable) -> Vec<RailContext> {
    let mut contexts: Vec<RailContext> = Vec::new();
    for row in 0..table.rows.len() {
        let context_index = pi(table.get(row, "context_index"));
        if contexts
            .last()
            .is_none_or(|context| context.context_index != context_index)
        {
            contexts.push(RailContext {
                context_index,
                road: table.get(row, "road").to_string(),
                element: pi(table.get(row, "element")),
                total_length: pf(table.get(row, "total_length")),
                successor_indices: table
                    .get(row, "successor_indices")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(pi)
                    .collect(),
                records: Vec::new(),
            });
        }
        if let Some(context) = contexts.last_mut() {
            context.records.push(RailRecord {
                position: pf3(table, row, ["px", "py", "pz"]),
                secondary: pf3(table, row, ["sx", "sy", "sz"]),
                normal_or_side: pf3(table, row, ["nx", "ny", "nz"]),
            });
        }
    }
    contexts
}

/// `parse_world_rail_entries_tsv` entry (scripts/kart_track_world.py:152-195).
#[derive(Debug, Clone, Default)]
struct RailEntry {
    context_index: i64,
    road: String,
    element: i64,
    marker: String,
    successor_indices: Vec<i64>,
    boundary: Vec<[f64; 3]>,
    path_start: [f64; 3],
    tangent: [f64; 3],
    normal_or_side: [f64; 3],
}

fn parse_rail_entries(table: &TsvTable) -> Vec<RailEntry> {
    let mut entries: Vec<RailEntry> = Vec::new();
    for row in 0..table.rows.len() {
        let context_index = pi(table.get(row, "context_index"));
        if entries
            .last()
            .is_none_or(|entry| entry.context_index != context_index)
        {
            entries.push(RailEntry {
                context_index,
                road: table.get(row, "road").to_string(),
                element: pi(table.get(row, "element")),
                marker: table.get(row, "marker").to_string(),
                successor_indices: table
                    .get(row, "successor_indices")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(pi)
                    .collect(),
                boundary: Vec::new(),
                path_start: pf3(table, row, ["px", "py", "pz"]),
                tangent: pf3(table, row, ["sx", "sy", "sz"]),
                normal_or_side: pf3(table, row, ["nx", "ny", "nz"]),
            });
        }
        if let Some(entry) = entries.last_mut() {
            entry.boundary.push(pf3(table, row, ["bx", "by", "bz"]));
        }
    }
    entries
}

/// `parse_world_moving_query_tsv` (scripts/kart_track_world.py:474-504).
#[derive(Debug, Clone, Default)]
struct MovingRow {
    track_id: String,
    hit: bool,
    time_ms: i64,
    owner_index: i64,
    owner_node: String,
    node: String,
    tri_index: i64,
    road_surface: String,
    surface_id: i64,
    road_rail: String,
    distance: f64,
    point: [f64; 3],
    normal: [f64; 3],
    triangle: [[f64; 3]; 3],
}

fn parse_moving_row(table: &TsvTable) -> MovingRow {
    let hit = table.get(0, "hit") == "1";
    let mut row = MovingRow {
        track_id: table.get(0, "track_id").to_string(),
        hit,
        time_ms: pi(table.get(0, "time_ms")),
        ..MovingRow::default()
    };
    if hit {
        row.owner_index = pi(table.get(0, "owner_index"));
        row.owner_node = table.get(0, "owner_node").to_string();
        row.node = table.get(0, "node").to_string();
        row.tri_index = pi(table.get(0, "tri_index"));
        row.road_surface = table.get(0, "road_surface").to_string();
        row.surface_id = pi(table.get(0, "surface_id"));
        row.road_rail = table.get(0, "road_rail").to_string();
        row.distance = pf(table.get(0, "distance"));
        row.point = pf3(table, 0, ["qx", "qy", "qz"]);
        row.normal = pf3(table, 0, ["nx", "ny", "nz"]);
        row.triangle = [
            pf3(table, 0, ["v0x", "v0y", "v0z"]),
            pf3(table, 0, ["v1x", "v1y", "v1z"]),
            pf3(table, 0, ["v2x", "v2y", "v2z"]),
        ];
    }
    row
}

/// `parse_world_dynamic_state_tsv` (scripts/kart_track_world.py:447-471).
#[derive(Debug, Clone, Default)]
struct DynamicStateRow {
    track_id: String,
    node_index: i64,
    node: String,
    time_ms: i64,
    delta_ms: i64,
    position: [f64; 3],
    matrix: [f64; 9],
    velocity: [f64; 3],
    angular_velocity: [f64; 3],
}

fn parse_dynamic_state_row(table: &TsvTable) -> DynamicStateRow {
    let mut matrix = [0.0; 9];
    for matrix_row in 0..3 {
        for column in 0..3 {
            matrix[matrix_row * 3 + column] = pf(table.get(0, &format!("m{matrix_row}{column}")));
        }
    }
    DynamicStateRow {
        track_id: table.get(0, "track_id").to_string(),
        node_index: pi(table.get(0, "node_index")),
        node: table.get(0, "node").to_string(),
        time_ms: pi(table.get(0, "time_ms")),
        delta_ms: pi(table.get(0, "delta_ms")),
        position: pf3(table, 0, ["px", "py", "pz"]),
        matrix,
        velocity: pf3(table, 0, ["vx", "vy", "vz"]),
        angular_velocity: pf3(table, 0, ["wx", "wy", "wz"]),
    }
}

/// `parse_kap_markers_tsv` row (scripts/kart_track_world.py:507-519).
/// `slot` / `point_index` / `point_count` / `tangent` are parsed for
/// payload fidelity with the Python dicts; the trait-facing
/// `CheckpointTargetCandidate` only carries id/ordinal/position/distance.
#[derive(Debug, Clone, Default)]
struct KapMarkerRow {
    track_id: String,
    #[allow(dead_code)]
    slot: i64,
    marker_ordinal: i64,
    #[allow(dead_code)]
    point_index: i64,
    #[allow(dead_code)]
    point_count: i64,
    position: [f64; 3],
    #[allow(dead_code)]
    tangent: [f64; 3],
}

fn parse_kap_marker_rows(table: &TsvTable) -> Vec<KapMarkerRow> {
    (0..table.rows.len())
        .map(|row| KapMarkerRow {
            track_id: table.get(row, "track_id").to_string(),
            slot: pi(table.get(row, "slot")),
            marker_ordinal: pi(table.get(row, "marker_ordinal")),
            point_index: pi(table.get(row, "point_index")),
            point_count: pi(table.get(row, "point_count")),
            position: pf3(table, row, ["px", "py", "pz"]),
            tangent: pf3(table, row, ["tx", "ty", "tz"]),
        })
        .collect()
}

/// One spatially-deduplicated KAP event-marker cluster
/// (`KartTrackWorld.checkpoint_markers`, scripts/kart_track_world.py:613-656).
#[derive(Debug, Clone, Default)]
struct CheckpointMarker {
    marker_id: String,
    marker_ordinal: i64,
    position: [f64; 3],
    source_count: usize,
}

/// Spatial dedup (distance squared <= 1.0, first-seen cluster wins).
fn cluster_checkpoint_markers(markers: &[KapMarkerRow]) -> Vec<CheckpointMarker> {
    let mut clusters: Vec<CheckpointMarker> = Vec::new();
    for marker in markers {
        let existing = clusters.iter_mut().find(|cluster| {
            sum_f64(&[
                (marker.position[0] - cluster.position[0]).powi(2),
                (marker.position[1] - cluster.position[1]).powi(2),
                (marker.position[2] - cluster.position[2]).powi(2),
            ]) <= 1.0
        });
        match existing {
            Some(cluster) => cluster.source_count += 1,
            None => clusters.push(CheckpointMarker {
                marker_id: format!("{}:{}", marker.track_id, clusters.len()),
                marker_ordinal: marker.marker_ordinal,
                position: marker.position,
                source_count: 1,
            }),
        }
    }
    clusters
}

// ---------------------------------------------------------------------------
// Pure-Python helper ports (f64, Neumaier-compensated like CPython `sum()`).
// ---------------------------------------------------------------------------

/// `_orthonormal_column_axes` (scripts/kart_track_world.py:21-54): publish a
/// drift-safe rigid orientation from a row-major engine matrix. Returns the
/// three orthonormal column axes flattened as `[a0.., a1.., a2..]`, matching
/// the CLI argument layout. Python raises on degenerate axes; the typed API
/// surfaces that as `None`.
fn orthonormal_column_axes(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let columns: [[f64; 3]; 3] = [
        [matrix[0], matrix[3], matrix[6]],
        [matrix[1], matrix[4], matrix[7]],
        [matrix[2], matrix[5], matrix[8]],
    ];
    let normalize = |vector: [f64; 3]| -> Option<[f64; 3]> {
        let magnitude = sum_f64(&[
            vector[0] * vector[0],
            vector[1] * vector[1],
            vector[2] * vector[2],
        ])
        .sqrt();
        if !magnitude.is_finite() || magnitude <= 1.0e-9 {
            return None;
        }
        Some([
            vector[0] / magnitude,
            vector[1] / magnitude,
            vector[2] / magnitude,
        ])
    };
    let first = normalize(columns[0])?;
    let projection = dot3_sum(columns[1], first);
    let second = normalize([
        columns[1][0] - projection * first[0],
        columns[1][1] - projection * first[1],
        columns[1][2] - projection * first[2],
    ])?;
    let mut third = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    let mut second = second;
    if dot3_sum(third, columns[2]) < 0.0 {
        second = [-second[0], -second[1], -second[2]];
        third = [-third[0], -third[1], -third[2]];
    }
    Some([
        first[0], first[1], first[2], second[0], second[1], second[2], third[0], third[1], third[2],
    ])
}

/// `_rail_triangle_direction` (scripts/kart_track_world.py:282-305): the
/// deterministic authored-face tangent for initial rail capture.
fn rail_triangle_direction(triangle: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let edges: [[f64; 3]; 3] = [
        [
            triangle[1][0] - triangle[0][0],
            triangle[1][1] - triangle[0][1],
            triangle[1][2] - triangle[0][2],
        ],
        [
            triangle[2][0] - triangle[0][0],
            triangle[2][1] - triangle[0][1],
            triangle[2][2] - triangle[0][2],
        ],
        [
            triangle[2][0] - triangle[1][0],
            triangle[2][1] - triangle[1][1],
            triangle[2][2] - triangle[1][2],
        ],
    ];
    let squared =
        |edge: [f64; 3]| sum_f64(&[edge[0] * edge[0], edge[1] * edge[1], edge[2] * edge[2]]);
    // Python `max(..., key=...)`: first maximal element wins ties.
    let mut edge = edges[0];
    let mut best = squared(edge);
    for candidate in edges.into_iter().skip(1) {
        let value = squared(candidate);
        if value > best {
            best = value;
            edge = candidate;
        }
    }
    let magnitude = best.sqrt();
    if !magnitude.is_finite() || magnitude <= 1.0e-9 {
        return None;
    }
    let mut direction = [
        edge[0] / magnitude,
        edge[1] / magnitude,
        edge[2] / magnitude,
    ];
    let mut dominant = 0usize;
    for axis in 1..3 {
        if direction[axis].abs() > direction[dominant].abs() {
            dominant = axis;
        }
    }
    if direction[dominant] < 0.0 {
        direction = [-direction[0], -direction[1], -direction[2]];
    }
    Some(direction)
}

/// `match_rail_entry_boundary` result (scripts/kart_track_world.py:198-279).
/// Carries every key of the Python mapping; the trait-facing
/// [`RailEntryBoundaryMatch`] drops the extras the engine never reads.
#[derive(Debug, Clone, Default)]
struct RailEntryMatch {
    context_id: i64,
    context_index: i64,
    road: String,
    element: i64,
    marker: String,
    signed_plane_distance: f64,
    plane_distance: f64,
    projected_coordinates: [f64; 2],
    path_start: [f64; 3],
    tangent: [f64; 3],
    tangent_progress: f64,
    normal_or_side: [f64; 3],
}

/// `match_rail_entry_boundary`: fit the entry plane from the two longest
/// independent boundary spans and test projected min/max coordinates
/// (invariant to the authored strip ordering).
fn match_rail_entry_boundary(
    position: [f64; 3],
    entry: &RailEntry,
    plane_tolerance: f64,
    edge_tolerance: f64,
) -> Option<RailEntryMatch> {
    let vertices = &entry.boundary;
    if vertices.len() < 3 {
        return None;
    }
    let count = vertices.len() as f64;
    let center = [
        sum_f64(&vertices.iter().map(|vertex| vertex[0]).collect::<Vec<_>>()) / count,
        sum_f64(&vertices.iter().map(|vertex| vertex[1]).collect::<Vec<_>>()) / count,
        sum_f64(&vertices.iter().map(|vertex| vertex[2]).collect::<Vec<_>>()) / count,
    ];
    let offsets: Vec<[f64; 3]> = vertices
        .iter()
        .map(|vertex| {
            [
                vertex[0] - center[0],
                vertex[1] - center[1],
                vertex[2] - center[2],
            ]
        })
        .collect();
    let normalize = |vector: [f64; 3]| -> Option<[f64; 3]> {
        let magnitude = dot3_sum(vector, vector).sqrt();
        if magnitude <= 1.0e-9 {
            return None;
        }
        Some([
            vector[0] / magnitude,
            vector[1] / magnitude,
            vector[2] / magnitude,
        ])
    };
    // Python `max(..., key=dot(value, value))`: first maximal wins ties.
    let longest = |candidates: &[[f64; 3]]| -> [f64; 3] {
        let mut best = candidates[0];
        let mut best_squared = dot3_sum(best, best);
        for candidate in candidates.iter().skip(1) {
            let value = dot3_sum(*candidate, *candidate);
            if value > best_squared {
                best_squared = value;
                best = *candidate;
            }
        }
        best
    };
    let first = normalize(longest(&offsets))?;
    let residuals: Vec<[f64; 3]> = offsets
        .iter()
        .map(|value| {
            let projection = dot3_sum(*value, first);
            [
                value[0] - projection * first[0],
                value[1] - projection * first[1],
                value[2] - projection * first[2],
            ]
        })
        .collect();
    let second = normalize(longest(&residuals))?;
    let normal = normalize([
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ])?;
    let relative = [
        position[0] - center[0],
        position[1] - center[1],
        position[2] - center[2],
    ];
    let plane_distance = dot3_sum(relative, normal);
    if plane_distance.abs() > plane_tolerance {
        return None;
    }
    let coordinates = [dot3_sum(relative, first), dot3_sum(relative, second)];
    for (index, basis) in [first, second].into_iter().enumerate() {
        let mut authored_min = f64::INFINITY;
        let mut authored_max = f64::NEG_INFINITY;
        for value in &offsets {
            let projection = dot3_sum(*value, basis);
            authored_min = min2_f64(authored_min, projection);
            authored_max = max2_f64(authored_max, projection);
        }
        let coordinate = coordinates[index];
        if coordinate < authored_min - edge_tolerance || coordinate > authored_max + edge_tolerance
        {
            return None;
        }
    }
    let path_relative = [
        position[0] - entry.path_start[0],
        position[1] - entry.path_start[1],
        position[2] - entry.path_start[2],
    ];
    let tangent_progress = dot3_sum(path_relative, entry.tangent);
    Some(RailEntryMatch {
        context_id: entry.context_index,
        context_index: entry.context_index,
        road: entry.road.clone(),
        element: entry.element,
        marker: entry.marker.clone(),
        signed_plane_distance: plane_distance,
        plane_distance: plane_distance.abs(),
        projected_coordinates: coordinates,
        path_start: entry.path_start,
        tangent: entry.tangent,
        tangent_progress,
        normal_or_side: entry.normal_or_side,
    })
}

// ---------------------------------------------------------------------------
// JsonValue construction helpers (Python mapping shapes).
// ---------------------------------------------------------------------------

fn jnum(value: f64) -> JsonValue {
    JsonValue::Number(value)
}

fn jint(value: i64) -> JsonValue {
    JsonValue::Number(value as f64)
}

fn jstr(value: impl Into<String>) -> JsonValue {
    JsonValue::String(value.into())
}

fn jbool(value: bool) -> JsonValue {
    JsonValue::Bool(value)
}

fn jvec3(value: [f64; 3]) -> JsonValue {
    JsonValue::Array(value.into_iter().map(JsonValue::Number).collect())
}

fn jvec9(value: [f64; 9]) -> JsonValue {
    JsonValue::Array(value.into_iter().map(JsonValue::Number).collect())
}

fn jmap(entries: Vec<(&str, JsonValue)>) -> BTreeMap<String, JsonValue> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// The `parse_world_rail_contexts_tsv` context mapping, as consumed by
/// `evaluate_rail_state1_query` (always the successor-graph authority, since
/// the renderer always emits the `successor_indices` column).
fn rail_context_json(context: &RailContext) -> BTreeMap<String, JsonValue> {
    jmap(vec![
        ("context_id", jint(context.context_index)),
        ("context_index", jint(context.context_index)),
        ("road", jstr(context.road.clone())),
        ("element", jint(context.element)),
        ("total_length", jnum(context.total_length)),
        (
            "successor_indices",
            JsonValue::Array(context.successor_indices.iter().map(|i| jint(*i)).collect()),
        ),
        (
            "records",
            JsonValue::Array(
                context
                    .records
                    .iter()
                    .map(|record| {
                        JsonValue::Object(jmap(vec![
                            ("position", jvec3(record.position)),
                            ("secondary", jvec3(record.secondary)),
                            ("normal_or_side", jvec3(record.normal_or_side)),
                        ]))
                    })
                    .collect(),
            ),
        ),
        (
            "authority",
            jstr("track_ToRoad_forward_course_successor_graph"),
        ),
    ])
}

/// The `parse_world_rail_entries_tsv` entry mapping.
fn rail_entry_json(entry: &RailEntry) -> BTreeMap<String, JsonValue> {
    jmap(vec![
        ("context_id", jint(entry.context_index)),
        ("context_index", jint(entry.context_index)),
        ("road", jstr(entry.road.clone())),
        ("element", jint(entry.element)),
        ("marker", jstr(entry.marker.clone())),
        (
            "successor_indices",
            JsonValue::Array(entry.successor_indices.iter().map(|i| jint(*i)).collect()),
        ),
        (
            "boundary",
            JsonValue::Array(entry.boundary.iter().map(|v| jvec3(*v)).collect()),
        ),
        ("path_start", jvec3(entry.path_start)),
        ("tangent", jvec3(entry.tangent)),
        ("normal_or_side", jvec3(entry.normal_or_side)),
        (
            "authority",
            jstr("track_ToRoad_element_boundary_and_first_nav_record"),
        ),
    ])
}

/// The `match_rail_entry_boundary` mapping embedded as `entry_gate` in the
/// rail-contact selector (scripts/kart_track_world.py:265-279).
fn rail_entry_match_json(matched: &RailEntryMatch) -> BTreeMap<String, JsonValue> {
    jmap(vec![
        ("context_id", jint(matched.context_id)),
        ("context_index", jint(matched.context_index)),
        ("road", jstr(matched.road.clone())),
        ("element", jint(matched.element)),
        ("marker", jstr(matched.marker.clone())),
        ("signed_plane_distance", jnum(matched.signed_plane_distance)),
        ("plane_distance", jnum(matched.plane_distance)),
        (
            "projected_coordinates",
            JsonValue::Array(
                matched
                    .projected_coordinates
                    .into_iter()
                    .map(JsonValue::Number)
                    .collect(),
            ),
        ),
        ("path_start", jvec3(matched.path_start)),
        ("tangent", jvec3(matched.tangent)),
        ("tangent_progress", jnum(matched.tangent_progress)),
        ("normal_or_side", jvec3(matched.normal_or_side)),
        (
            "authority",
            jstr("track_ToRoad_element_entry_boundary_plane"),
        ),
    ])
}

/// The `parse_world_rail_query_tsv` candidate mapping embedded as `geometry`
/// in the rail-contact selector (scripts/kart_track_world.py:80-100).
fn rail_candidate_json(row: &GroundRow) -> BTreeMap<String, JsonValue> {
    let mut entries = vec![
        ("track_id", jstr(row.track_id.clone())),
        ("hit", jbool(true)),
        ("node", jstr(row.node.clone())),
        ("tri_index", jint(row.tri_index)),
        ("road_surface", jstr(row.road_surface.clone())),
        ("surface_id", jint(row.surface_id)),
        ("road_rail", jstr(row.road_rail.clone())),
        ("distance", jnum(row.distance)),
        ("point", jvec3(row.point)),
        ("normal", jvec3(row.normal)),
    ];
    if let Some(triangle) = row.triangle {
        entries.push((
            "triangle",
            JsonValue::Array(triangle.into_iter().map(jvec3).collect()),
        ));
    }
    jmap(entries)
}

/// The `parse_world_dynamic_state_tsv` mapping embedded in a moving-contact
/// candidate (scripts/kart_track_world.py:447-471).
fn dynamic_state_json(state: &DynamicStateRow) -> BTreeMap<String, JsonValue> {
    jmap(vec![
        ("track_id", jstr(state.track_id.clone())),
        ("node_index", jint(state.node_index)),
        ("node", jstr(state.node.clone())),
        ("time_ms", jint(state.time_ms)),
        ("delta_ms", jint(state.delta_ms)),
        ("position", jvec3(state.position)),
        ("matrix", jvec9(state.matrix)),
        ("velocity", jvec3(state.velocity)),
        ("angular_velocity", jvec3(state.angular_velocity)),
        (
            "authority",
            jstr("track_prs_parent_composed_time_sample_and_forward_difference"),
        ),
    ])
}

// ---------------------------------------------------------------------------
// TrackWorld: typed world adapter.
// ---------------------------------------------------------------------------

/// P2 disposition for one decoded `ToMovableObject`.
///
/// Event objects are visible to the world adapter even when their physical
/// behavior is not eligible for a query surface.  This keeps the fail-safe
/// decision explicit: an unverified event cannot silently enter the static
/// collision soup or the MOVING-CONTACT selector merely because kart-fmt can
/// decode and render its subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventPhysicsDisposition {
    /// P1-closed dynamic GoItemObstacle response routed through CONTACT.
    ContactDynamicObstacle,
    /// No per-class P1 physical adjudication exists.
    VisualOnlyUnadjudicated,
}

impl EventPhysicsDisposition {
    pub fn physics_enabled(self) -> bool {
        matches!(self, Self::ContactDynamicObstacle)
    }
}

/// One decoded event object projected at the P2 world boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct EventWorldObject {
    pub event_index: i16,
    pub class_name: &'static str,
    pub name: String,
    pub object_type: Option<String>,
    pub position: [f32; 3],
    pub disposition: EventPhysicsDisposition,
}

fn event_world_objects(track: &kart_fmt::kart_object::Track) -> Vec<EventWorldObject> {
    track
        .runtime_event_roots()
        .filter_map(|root| {
            if root.class_name != Some("ToMovableObject") {
                return None;
            }
            let placement = root.placement.as_ref()?;
            let big_gear = root.index == 1743
                && placement.name == "mo_dummy01-톱니바퀴a"
                && placement.position == FACTORY_BIG_GEAR_POSITION;
            Some(EventWorldObject {
                event_index: root.index,
                class_name: "ToMovableObject",
                name: placement.name.clone(),
                object_type: tag_attr(placement.property.as_ref(), "object", "type"),
                position: placement.position,
                disposition: if big_gear {
                    EventPhysicsDisposition::ContactDynamicObstacle
                } else {
                    EventPhysicsDisposition::VisualOnlyUnadjudicated
                },
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EventObstacleModel {
    event_index: i16,
    object_id: String,
    subtree_root: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BananaItemModel {
    event_index: i16,
    object_id: String,
    subtree_root: usize,
}

fn banana_item_models(track: &kart_fmt::kart_object::Track) -> Vec<BananaItemModel> {
    [
        (1762i16, "mo_banana04+", "banana:0xF0000039"),
        (1818i16, "mo_banana07+", "banana:0xF000003B"),
        (2150i16, "mo_banana08+", "banana:0xF0000045"),
    ]
    .into_iter()
    .filter_map(|(event_index, name, object_id)| {
        track.runtime_event_roots().find(|root| {
            root.index == event_index
                && root
                    .placement
                    .as_ref()
                    .is_some_and(|placement| placement.name == name)
        })?;
        let wanted = format!("#{name}").to_lowercase();
        let subtree_root = track
            .runtime_event_roots()
            .filter_map(|root| root.node)
            .find(|&index| track.nodes[index].name.to_lowercase() == wanted)?;
        Some(BananaItemModel {
            event_index,
            object_id: object_id.to_string(),
            subtree_root,
        })
    })
    .collect()
}

fn event_obstacle_models(
    track: &kart_fmt::kart_object::Track,
    objects: &[EventWorldObject],
) -> Vec<EventObstacleModel> {
    objects
        .iter()
        .filter(|object| object.disposition.physics_enabled())
        .filter_map(|object| {
            let wanted = format!("#{}", object.name).to_lowercase();
            let subtree_root = track
                .runtime_event_roots()
                .filter_map(|root| root.node)
                .find(|&index| track.nodes[index].name.to_lowercase() == wanted)?;
            Some(EventObstacleModel {
                event_index: object.event_index,
                object_id: format!("event:{}", object.event_index),
                subtree_root,
            })
        })
        .collect()
}

/// Typed track.1s world: the Rust-native equivalent of the Python
/// `KartTrackWorld` subprocess client. Loads through the same asset layer
/// (cached by [`crate::track::load_static_track`]) and answers
/// `kart_physics::world::World` queries with the exact values the Python
/// client would produce through the CLI/TSV path.
///
/// Error-channel note: the Python client surfaces failures (bad track file,
/// missing embedded course, invalid arguments) as exceptions. The `World`
/// trait has no error channel, so this adapter maps them to the nearest
/// defined absence: render failures become `None` (optional capabilities)
/// or `WorldGroundResult::default()` (`query_ground`, mirroring the FFI
/// adapter precedent at kart-physics/src/ffi.rs:2739-2742), and Python
/// `ValueError` argument validation becomes the inner `None`.
pub struct TrackWorld {
    track_path: PathBuf,
    /// `clean(track_id(track_path))`: the exact `track_id` column value the
    /// TSV path produced for this world.
    track_id: String,
    index: Arc<TrackQueryIndex>,
    track: Arc<kart_fmt::kart_object::Track>,
    event_objects: Vec<EventWorldObject>,
    event_obstacles: Vec<EventObstacleModel>,
    banana_items: Vec<BananaItemModel>,
    kap_paths: Vec<PathBuf>,
    checkpoint_markers: OnceLock<Vec<CheckpointMarker>>,
}

impl TrackWorld {
    /// Load a `.track.1s` world from a path. The track is parsed eagerly so
    /// a bad path fails here instead of inside a query.
    pub fn from_path(track_path: impl Into<PathBuf>) -> Result<Self, String> {
        let track_path = track_path.into();
        let index = query::load_track_query_index(&track_path)?;
        let track = crate::track::load_static_track(&track_path)?;
        let track_id = crate::track::clean(&crate::track::track_id(&track_path));
        let event_objects = event_world_objects(&track);
        let event_obstacles = event_obstacle_models(&track, &event_objects);
        let banana_items = banana_item_models(&track);
        Ok(Self {
            track_path,
            track_id,
            index,
            track,
            event_objects,
            event_obstacles,
            banana_items,
            kap_paths: Vec::new(),
            checkpoint_markers: OnceLock::new(),
        })
    }

    /// Load a `.track.1s` world from caller-supplied bytes (W3: the browser
    /// passes the fetched ArrayBuffer; there is no filesystem). `track_path`
    /// is the identity path the bytes are registered under — the TSV render
    /// pipeline is keyed by path, so this must be the same string a native
    /// [`TrackWorld::from_path`] run would use for responses (e.g.
    /// `track_id` columns) to stay bit-identical.
    pub fn from_bytes(track_path: impl Into<PathBuf>, bytes: &[u8]) -> Result<Self, String> {
        let track_path = track_path.into();
        crate::track::register_virtual_bytes(&track_path, bytes)?;
        Self::from_path(track_path)
    }

    /// Attach KAP event-marker paths backing the checkpoint selectors
    /// (`KartTrackWorld(kap_paths=...)`, scripts/kart_track_world.py:529-537).
    pub fn with_kap_paths(mut self, kap_paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.kap_paths = kap_paths.into_iter().collect();
        self
    }

    /// The bytes variant of [`TrackWorld::with_kap_paths`]: registers each
    /// KAP payload under its identity path (see [`TrackWorld::from_bytes`])
    /// and attaches the paths in order.
    pub fn with_kap_bytes(
        mut self,
        kap: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
    ) -> Result<Self, String> {
        let mut paths = Vec::new();
        for (path, bytes) in kap {
            crate::track::register_virtual_bytes(&path, &bytes)?;
            paths.push(path);
        }
        self.kap_paths = paths;
        Ok(self)
    }

    pub fn track_path(&self) -> &Path {
        &self.track_path
    }

    pub fn kap_paths(&self) -> &[PathBuf] {
        &self.kap_paths
    }

    /// Decoded `ToMovableObject` inventory plus its P1/P2 physical
    /// disposition.  This is diagnostic metadata, not a physics query.
    pub fn event_objects(&self) -> &[EventWorldObject] {
        &self.event_objects
    }

    /// True only after at least one decoded event class has a fully closed
    /// P1 response and is actually routed into a world query capability.
    pub fn has_event_physics_candidates(&self) -> bool {
        self.event_objects
            .iter()
            .any(|object| object.disposition.physics_enabled())
    }

    fn rail_contexts(&self) -> Option<Arc<Vec<RailContext>>> {
        rail_context_cache(&self.track_path)
    }

    fn rail_entries(&self) -> Option<Arc<Vec<RailEntry>>> {
        rail_entry_cache(&self.track_path)
    }

    /// `checkpoint_markers` (scripts/kart_track_world.py:613-656). A KAP
    /// path that fails to render is skipped (Python would raise; the trait
    /// has no error channel — recorded deviation).
    fn checkpoint_markers(&self) -> &Vec<CheckpointMarker> {
        self.checkpoint_markers.get_or_init(|| {
            let mut rows = Vec::new();
            for path in &self.kap_paths {
                let Ok(text) = cli::render_kap_markers(&[path.to_string_lossy().into_owned()])
                else {
                    continue;
                };
                rows.extend(parse_kap_marker_rows(&TsvTable::parse(&text)));
            }
            cluster_checkpoint_markers(&rows)
        })
    }

    /// `dynamic_object_state` (scripts/kart_track_world.py:700-723).
    fn dynamic_object_state(
        &self,
        node_index: i64,
        time_ms: i64,
        delta_ms: i64,
    ) -> Option<DynamicStateRow> {
        if node_index < 0 || delta_ms <= 0 {
            return None;
        }
        let time_ms = i32::try_from(time_ms).ok()?;
        let delta_ms = i32::try_from(delta_ms).ok()?;
        let text = cli::render_world_dynamic_state(&[
            self.track_path.to_string_lossy().into_owned(),
            node_index.to_string(),
            time_ms.to_string(),
            delta_ms.to_string(),
        ])
        .ok()?;
        Some(parse_dynamic_state_row(&TsvTable::parse(&text)))
    }

    /// The full `match_rail_entry_boundary` scan of
    /// `query_rail_entry_boundary` (scripts/kart_track_world.py:832-847):
    /// substring road filter, then the smallest plane distance wins.
    fn match_entry(
        &self,
        position: [f64; 3],
        road: Option<&str>,
        plane_tolerance: f64,
    ) -> Option<RailEntryMatch> {
        let entries = self.rail_entries()?;
        let mut best: Option<RailEntryMatch> = None;
        for entry in entries.iter() {
            if let Some(road) = road {
                // Python: `if road is not None and entry["road"] not in road`.
                if !road.contains(entry.road.as_str()) {
                    continue;
                }
            }
            if let Some(matched) = match_rail_entry_boundary(position, entry, plane_tolerance, 0.05)
            {
                if best
                    .as_ref()
                    .is_none_or(|current| matched.plane_distance < current.plane_distance)
                {
                    best = Some(matched);
                }
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// In-memory selection -> typed result conversion (replaces the TSV round
// trip; every f32 field crosses through [`q9`] exactly where the old code
// crossed text).
// ---------------------------------------------------------------------------

/// One `world-overlap-obb-all` / rotating-all hit row as a typed overlap:
/// the row semantics of `parse_world_obb_overlaps_tsv` — the face-normal
/// columns are always emitted on a hit, so the authored winding is the
/// normal authority and `contact_basis_44` is always present.
fn overlap_hit_result(
    track_id: &str,
    loop_index: usize,
    entry: &query::StaticTri,
    hit: &crate::geom::ObbTriangleHit,
) -> WorldObbOverlap {
    WorldObbOverlap {
        track_id: track_id.to_string(),
        hit: true,
        loop_index: loop_index as i64,
        fraction: q9(hit.fraction),
        node: entry.node.to_string(),
        tri_index: entry.tri_index as i64,
        road_surface: entry.surface.to_string(),
        surface_id: entry.surface_id as i64,
        road_rail: entry.rail.to_string(),
        center: q93(hit.center),
        support_point: q93(hit.support_point),
        ccd_normal: q93(hit.normal),
        normal: q93(hit.face_normal),
        normal_authority: "CollisionFace_authored_triangle_winding".to_string(),
        contact_basis_44: Some(q93(entry.contact_basis_44)),
        boost_multiplier_64c: false,
    }
}

impl World for TrackWorld {
    fn capabilities(&self) -> WorldCapabilities {
        // `query_at_step_start` mirrors the Python class attribute
        // (scripts/kart_track_world.py:527). `query_network_correction` and
        // `query_distinct_water_volume` stay absent: they are
        // recorded-runtime features, not asset queries.
        WorldCapabilities {
            overlap_obb_all: true,
            query_event_obstacle_contacts: !self.event_obstacles.is_empty(),
            query_banana_trigger: !self.banana_items.is_empty(),
            query_takeoff_contact: true,
            query_landing_contact: true,
            query_rail_contact: true,
            query_rail_entry_boundary: true,
            query_moving_contact: true,
            query_checkpoint_target_candidate: true,
            query_checkpoint_respawn: true,
            sweep_obb_rotating_all: true,
            sweep_obb_rotating: true,
            sweep_point: true,
            query_at_step_start: true,
            rail_query_contexts: true,
            rail_entry_contexts: true,
            query_network_correction: false,
            query_distinct_water_volume: false,
        }
    }

    /// `query_ground` (scripts/kart_track_world.py:1046-1051).
    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult {
        let point = f32x3(position);
        let up = f32x3(up);
        let min_dot = min_dot as f32;
        match query::query_ground(&self.index, point, up, min_dot) {
            Some(selection) => WorldGroundResult {
                track_id: self.track_id.clone(),
                node: selection.entry.node.to_string(),
                tri_index: selection.entry.tri_index as i64,
                road_surface: selection.entry.surface.to_string(),
                surface_id: selection.entry.surface_id as i64,
                road_rail: selection.entry.rail.to_string(),
                distance: q9(selection.distance_sq.sqrt()),
                point: q93(selection.closest),
                normal: q93(selection.entry.normal),
                triangle: Some(selection.entry.tri.map(q93)),
                contact_basis_44: Some(q93(selection.entry.contact_basis_44)),
            },
            // FFI adapter precedent: absence degrades to the default row.
            None => WorldGroundResult::default(),
        }
    }

    /// `sweep_obb_rotating_all` (scripts/kart_track_world.py:1110-1138):
    /// static-pose shortcut to `overlap_obb_all`, otherwise raw column axes
    /// (no re-orthonormalization, unlike `sweep_obb_rotating`).
    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        if start == end && start_matrix == end_matrix {
            return self.overlap_obb_all(start, start_matrix, half_extents);
        }
        let column_axes = |matrix: [f64; 9]| -> [f64; 9] {
            [
                matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2],
                matrix[5], matrix[8],
            ]
        };
        let start_axes = column_axes(start_matrix);
        let end_axes = column_axes(end_matrix);
        let start = f32x3(start);
        let end = f32x3(end);
        let start_axes = f32x9(start_axes);
        let end_axes = f32x9(end_axes);
        let half_extents = f32x3(half_extents);
        // The renderer validates both poses before scanning; its failure
        // surfaced as `None` here (`.ok()?`).
        if validate_obb(start_axes, half_extents).is_err()
            || validate_obb(end_axes, half_extents).is_err()
        {
            return None;
        }
        let hits = query::query_sweep_obb_rotating_all(
            &self.index,
            start,
            end,
            start_axes,
            end_axes,
            half_extents,
        );
        Some(
            hits.iter()
                .enumerate()
                .map(|(loop_index, selection)| {
                    overlap_hit_result(&self.track_id, loop_index, selection.entry, &selection.hit)
                })
                .collect(),
        )
    }

    /// `sweep_obb_rotating` (scripts/kart_track_world.py:1087-1100): both
    /// matrices pass through `_orthonormal_column_axes` first.
    fn sweep_obb_rotating(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<WorldObbOverlap> {
        let start_axes = orthonormal_column_axes(start_matrix)?;
        let end_axes = orthonormal_column_axes(end_matrix)?;
        let start = f32x3(start);
        let end = f32x3(end);
        let start_axes = f32x9(start_axes);
        let end_axes = f32x9(end_axes);
        let half_extents = f32x3(half_extents);
        // Python raises on CLI failure (e.g. validate_obb rejecting
        // degenerate axes); the trait has no error channel, so the failure
        // degrades to the miss row shape (empty track_id, like the parsed
        // default row).
        if validate_obb(start_axes, half_extents).is_err()
            || validate_obb(end_axes, half_extents).is_err()
        {
            return Some(WorldObbOverlap {
                normal_authority: "legacy_CCD_separating_axis".to_string(),
                ..WorldObbOverlap::default()
            });
        }
        let best = query::query_sweep_obb_rotating(
            &self.index,
            start,
            end,
            start_axes,
            end_axes,
            half_extents,
        );
        let (
            hit,
            track_id,
            node,
            tri_index,
            surface,
            surface_id,
            rail,
            fraction,
            center,
            support,
            normal,
        ) = match &best {
            Some(selection) => (
                true,
                self.track_id.clone(),
                selection.entry.node.to_string(),
                selection.entry.tri_index as i64,
                selection.entry.surface.to_string(),
                selection.entry.surface_id as i64,
                selection.entry.rail.to_string(),
                q9(selection.hit.fraction),
                q93(selection.hit.center),
                q93(selection.hit.support_point),
                q93(selection.hit.normal),
            ),
            None => (
                false,
                self.track_id.clone(),
                String::new(),
                0,
                String::new(),
                0,
                String::new(),
                0.0,
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
            ),
        };
        Some(WorldObbOverlap {
            track_id,
            hit,
            loop_index: 0,
            fraction,
            node,
            tri_index,
            road_surface: surface,
            surface_id,
            road_rail: rail,
            center,
            support_point: support,
            ccd_normal: normal,
            normal,
            normal_authority: "legacy_CCD_separating_axis".to_string(),
            contact_basis_44: None,
            boost_multiplier_64c: false,
        })
    }

    /// `sweep_point` (scripts/kart_track_world.py:1053-1061).
    fn sweep_point(&self, start: [f64; 3], end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        let start = f32x3(start);
        let end = f32x3(end);
        let best = query::query_sweep(&self.index, start, end);
        let mut entries = vec![
            ("track_id", jstr(self.track_id.clone())),
            ("hit", jbool(best.is_some())),
        ];
        if let Some(best) = best {
            entries.extend([
                ("fraction", jnum(q9(best.fraction))),
                ("node", jstr(best.entry.node.to_string())),
                ("tri_index", jint(best.entry.tri_index as i64)),
                ("road_surface", jstr(best.entry.surface.to_string())),
                ("surface_id", jint(best.entry.surface_id as i64)),
                ("road_rail", jstr(best.entry.rail.to_string())),
                ("point", jvec3(q93(best.point))),
                ("normal", jvec3(q93(best.entry.normal))),
            ]);
        }
        Some(jmap(entries))
    }

    /// `overlap_obb_all` (scripts/kart_track_world.py:1102-1108).
    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let axes = orthonormal_column_axes(matrix)?;
        let center = f32x3(center);
        let axes = f32x9(axes);
        let half_extents = f32x3(half_extents);
        // The renderer validates before scanning; its failure surfaced as
        // `None` here (`.ok()?`).
        if validate_obb(axes, half_extents).is_err() {
            return None;
        }
        let hits = query::query_overlap_obb_all(&self.index, center, axes, half_extents);
        Some(
            hits.iter()
                .enumerate()
                .map(|(loop_index, selection)| {
                    overlap_hit_result(&self.track_id, loop_index, selection.entry, &selection.hit)
                })
                .collect(),
        )
    }

    fn query_event_obstacle_contacts(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
        time_ms: i64,
        delta_ms: i64,
    ) -> Option<Vec<EventObstacleContact>> {
        if delta_ms <= 0 {
            return None;
        }
        let time_ms = i32::try_from(time_ms).ok()?;
        let delta_ms_i32 = i32::try_from(delta_ms).ok()?;
        let end_time_ms = time_ms.checked_add(delta_ms_i32)?;
        let axes = orthonormal_column_axes(matrix)?;
        let center = f32x3(center);
        let axes = f32x9(axes);
        let half_extents = f32x3(half_extents);
        validate_obb(axes, half_extents).ok()?;
        let dt = delta_ms_i32 as f32 / 1000.0;
        let mut contacts = Vec::new();
        let mut loop_index = 0i64;
        for obstacle in &self.event_obstacles {
            let start = self
                .track
                .flatten_subtree_at(obstacle.subtree_root, time_ms);
            let end = self
                .track
                .flatten_subtree_at(obstacle.subtree_root, end_time_ms)
                .into_iter()
                .collect::<BTreeMap<_, _>>();
            let obstacle_position = start
                .first()
                .map(|(_, world)| [world[12], world[13], world[14]])
                .unwrap_or([0.0; 3]);
            for (node_index, start_world) in start {
                let node = &self.track.nodes[node_index];
                let Some(geom) = &node.geom else {
                    continue;
                };
                let Some(end_world) = end.get(&node_index) else {
                    continue;
                };
                for (tri_index, indices) in geom_triangles(geom).iter().enumerate() {
                    if indices
                        .iter()
                        .any(|&index| index as usize >= geom.vertices.len())
                    {
                        continue;
                    }
                    let local = indices.map(|index| geom.vertices[index as usize]);
                    let triangle = local.map(|point| xform(&start_world, point));
                    let this_loop_index = loop_index;
                    loop_index += 1;
                    let Some(hit) = swept_obb_triangle_intersection(
                        center,
                        center,
                        axes,
                        half_extents,
                        triangle,
                    ) else {
                        continue;
                    };
                    let start_centroid = [
                        (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
                        (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
                        (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0,
                    ];
                    let end_triangle = local.map(|point| xform(end_world, point));
                    let end_centroid = [
                        (end_triangle[0][0] + end_triangle[1][0] + end_triangle[2][0]) / 3.0,
                        (end_triangle[0][1] + end_triangle[1][1] + end_triangle[2][1]) / 3.0,
                        (end_triangle[0][2] + end_triangle[1][2] + end_triangle[2][2]) / 3.0,
                    ];
                    let motion = [
                        (end_centroid[0] - start_centroid[0]) / dt,
                        (end_centroid[1] - start_centroid[1]) / dt,
                        (end_centroid[2] - start_centroid[2]) / dt,
                    ];
                    contacts.push(EventObstacleContact {
                        event_index: i64::from(obstacle.event_index),
                        object_id: obstacle.object_id.clone(),
                        node: node.name.clone(),
                        tri_index: tri_index as i64,
                        loop_index: this_loop_index,
                        centroid: q93(start_centroid),
                        normal: q93(hit.face_normal),
                        motion: q93(motion),
                        obstacle_position: q93(obstacle_position),
                    });
                }
            }
        }
        Some(contacts)
    }

    fn query_banana_trigger(&self, position: [f64; 3], time_ms: i64) -> Option<BananaTrigger> {
        let time_ms = i32::try_from(time_ms).ok()?;
        let position = f32x3(position);
        let mut best: Option<(f32, &BananaItemModel, P3)> = None;
        for item in &self.banana_items {
            let item_position = self
                .track
                .flatten_subtree_at(item.subtree_root, time_ms)
                .first()
                .map(|(_, world)| [world[12], world[13], world[14]])?;
            let dx = position[0] - item_position[0];
            let dy = position[1] - item_position[1];
            let dz = position[2] - item_position[2];
            let distance = ((dx * dx + dy * dy) + dz * dz).sqrt();
            if distance <= 2.0 && best.as_ref().is_none_or(|(prior, _, _)| distance < *prior) {
                best = Some((distance, item, item_position));
            }
        }
        let (distance, item, item_position) = best?;
        Some(BananaTrigger {
            object_id: item.object_id.clone(),
            event_index: i64::from(item.event_index),
            position: q93(item_position),
            distance: q9(distance),
            radius: 2.0,
            authority: "factory_ToMovableObject_PRS_plus_GoItemBanana_2m_gate".to_string(),
        })
    }

    /// Asset-backed takeoff selector. Mirrors
    /// `KartTrackWorld.query_takeoff_contact`: authored jump metadata is the
    /// general path; the only metadata-less exception is the exact
    /// desert_I08 face retained by two accepted original-game AIR cycles.
    /// A generic hard overlap is deliberately not a takeoff selector because
    /// factory_I02 uses the same collision writer while remaining supported.
    fn query_takeoff_contact(
        &self,
        _position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let mut best: Option<(f64, [f64; 3], &str, i64, &str)> = None;
        for (index, active) in active_slots.iter().enumerate() {
            if *active == 0 {
                continue;
            }
            let Some(plane) = support_planes.get(index) else {
                continue;
            };
            let proven_unlabelled = plane.node == DESERT_I08_UNLABELLED_TAKEOFF_NODE
                && ((plane.tri_index == DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRI
                    && plane.normal == DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_NORMAL
                    && plane.triangle == Some(DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRIANGLE))
                    || (plane.tri_index == DESERT_I08_UNLABELLED_TAKEOFF_FACE_TRI
                        && plane.normal == DESERT_I08_UNLABELLED_TAKEOFF_FACE_NORMAL
                        && plane.triangle == Some(DESERT_I08_UNLABELLED_TAKEOFF_FACE_TRIANGLE)));
            let token = plane.road_surface.trim().to_lowercase();
            if token != "jump" && token != "점프" && !proven_unlabelled {
                continue;
            }
            let (normal, node, tri_index, source) = if proven_unlabelled {
                (
                    DESERT_I08_UNLABELLED_TAKEOFF_FACE_NORMAL,
                    DESERT_I08_UNLABELLED_TAKEOFF_NODE,
                    DESERT_I08_UNLABELLED_TAKEOFF_FACE_TRI,
                    "track_world_runtime_proven_desert_i08_unlabelled_takeoff_face",
                )
            } else {
                (
                    plane.normal,
                    plane.node.as_str(),
                    plane.tri_index,
                    "track_world_active_support_jump_face",
                )
            };
            let approach = dot3_sum(velocity, normal);
            if approach < 0.0 && best.as_ref().is_none_or(|current| approach < current.0) {
                best = Some((approach, normal, node, tri_index, source));
            }
        }
        Some(best.map(
            |(_, normal, node, tri_index, source)| TakeoffLandingContact {
                selected: Some(true),
                hit: None,
                normal: Some(normal),
                face_id: Some(format!("{node}:{tri_index}")),
                source: Some(source.to_string()),
            },
        ))
    }

    /// Asset-backed landing selector. Same provenance caveat as
    /// `query_takeoff_contact`: no authoritative Python implementation
    /// exists. Composition: among the active support planes pick the normal
    /// most opposed to the velocity (the face being landed on); with no
    /// active planes, fall back to the nearest upward ground face. The
    /// engine only takes the normalized `normal` plus optional `face_id`
    /// (kartrider_vehicle_engine.py:6592-6633).
    fn query_landing_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let mut best: Option<(f64, &WorldGroundResult)> = None;
        for slot in active_slots {
            let Ok(index) = usize::try_from(*slot) else {
                continue;
            };
            let Some(plane) = support_planes.get(index) else {
                continue;
            };
            let opposition = dot3_sum(velocity, plane.normal);
            if best.as_ref().is_none_or(|current| opposition < current.0) {
                best = Some((opposition, plane));
            }
        }
        if let Some((_, plane)) = best {
            return Some(Some(TakeoffLandingContact {
                selected: None,
                hit: Some(true),
                normal: Some(plane.normal),
                face_id: Some(format!("{}:{}", plane.node, plane.tri_index)),
                source: Some("track_world_most_opposed_active_support_normal".to_string()),
            }));
        }
        let ground = self.query_ground(position, [0.0, 0.0, 1.0], 0.0);
        if ground.track_id.is_empty() {
            return Some(None);
        }
        Some(Some(TakeoffLandingContact {
            selected: None,
            hit: Some(true),
            normal: Some(ground.normal),
            face_id: Some(format!("{}:{}", ground.node, ground.tri_index)),
            source: Some("track_world_nearest_upward_ground_face".to_string()),
        }))
    }

    /// `query_rail_contact` (scripts/kart_track_world.py:937-1044): the
    /// scoped authored `road.rail=highSpeed` selector.
    fn query_rail_contact(
        &self,
        position: [f64; 3],
        previous_rail_state: i64,
        owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        if previous_rail_state < 0 {
            // Python raises ValueError; surfaced as absence here.
            return Some(None);
        }
        let point = f32x3(position);
        let best = query::query_rail(&self.index, point);
        let hit = best.is_some();
        let candidate = match &best {
            Some(selection) => GroundRow {
                track_id: self.track_id.clone(),
                node: selection.entry.node.to_string(),
                tri_index: selection.entry.tri_index as i64,
                road_surface: selection.entry.surface.to_string(),
                surface_id: selection.entry.surface_id as i64,
                road_rail: selection.entry.rail.to_string(),
                distance: q9(selection.distance_sq.sqrt()),
                point: q93(selection.closest),
                normal: q93(selection.entry.normal),
                triangle: Some(selection.entry.tri.map(q93)),
                // The rail TSV carries no contact_basis_44 columns.
                contact_basis_44: None,
            },
            None => GroundRow {
                track_id: self.track_id.clone(),
                ..GroundRow::default()
            },
        };
        let entry_gate = if hit && previous_rail_state == 0 {
            self.match_entry(position, Some(&candidate.node), 0.05)
        } else {
            None
        };
        let entry_forward_crossed = entry_gate
            .as_ref()
            .is_some_and(|gate| gate.tangent_progress >= 0.0);
        let contact_distance_limit = if previous_rail_state == 0 {
            0.3203125
        } else {
            4.00
        };
        if !hit
            || candidate.road_rail != "highSpeed"
            || (previous_rail_state != 0 && candidate.distance > contact_distance_limit)
            || (previous_rail_state == 0 && !entry_forward_crossed)
        {
            return Some(None);
        }
        let triangle = candidate.triangle?;
        let rail_dir = rail_triangle_direction(triangle)?;
        let rail_state = if previous_rail_state == 0 {
            1
        } else {
            previous_rail_state
        };
        let mut selector = jmap(vec![
            ("contact_active", jbool(true)),
            ("rail_state", jint(rail_state)),
            (
                "path_id",
                jstr(format!(
                    "{}:{}:{}",
                    candidate.track_id, candidate.node, candidate.road_rail
                )),
            ),
            ("owner_id", owner_id.map_or(JsonValue::Null, jstr)),
            (
                "runtime",
                JsonValue::Object(jmap(vec![
                    ("rail_dir", jvec3(rail_dir)),
                    ("accel_be0", jnum(2.0)),
                    ("obf_6e8", jnum(2304.0)),
                    ("grip_720", jnum(2070.0)),
                    ("gravity_be8", jnum(0.5)),
                    ("obf_808", jnum(1.84949994)),
                    ("obf_814", jnum(1.49399996)),
                    ("min_velocity", jnum(83.3333359)),
                    ("max_velocity", jnum(97.2222214)),
                ])),
            ),
            (
                "geometry",
                JsonValue::Object(rail_candidate_json(&candidate)),
            ),
            (
                "entry_gate",
                entry_gate
                    .as_ref()
                    .map(|gate| JsonValue::Object(rail_entry_match_json(gate)))
                    .unwrap_or(JsonValue::Null),
            ),
            ("entry_forward_crossed", jbool(entry_forward_crossed)),
            ("contact_distance_limit", jnum(contact_distance_limit)),
            (
                "authority",
                jstr(
                    "track_ToRoad_entry_boundary_forward_crossing_then_\
                     road_rail_highSpeed_nearest_triangle_and_factory_I02_\
                     runtime_profile",
                ),
            ),
            (
                "scope",
                jstr(
                    "factory_I02 highSpeed initial capture on the raw authored \
                     ToRoad entry boundary and retained-owner sustain within the \
                     decoded 4.00 m path metric threshold",
                ),
            ),
        ]);
        if (1..=3).contains(&rail_state) {
            let contexts = self
                .rail_contexts()
                .map(|contexts| {
                    JsonValue::Array(
                        contexts
                            .iter()
                            .map(|context| JsonValue::Object(rail_context_json(context)))
                            .collect(),
                    )
                })
                .unwrap_or(JsonValue::Null);
            selector.extend(jmap(vec![
                ("registered_query_contexts", contexts),
                ("contact_exists", jbool(true)),
                ("contact_identifier_nonempty", jbool(true)),
                ("metric_threshold", jnum(4.0)),
                ("height_bias", jnum(1.0)),
                ("alignment_primary_scale_ms", jnum(10.0)),
                ("alignment_stable_initial_ms", jnum(100.0)),
            ]));
        }
        Some(Some(selector))
    }

    /// `query_rail_entry_boundary` (scripts/kart_track_world.py:832-847).
    fn query_rail_entry_boundary(
        &self,
        position: [f64; 3],
        road: Option<&str>,
        plane_tolerance: f64,
    ) -> Option<Option<RailEntryBoundaryMatch>> {
        let matched = self.match_entry(position, road, plane_tolerance);
        Some(matched.map(|matched| RailEntryBoundaryMatch {
            context_id: jint(matched.context_id),
            context_index: matched.context_index,
            road: matched.road,
            element: jint(matched.element),
            marker: jstr(matched.marker),
            signed_plane_distance: matched.signed_plane_distance,
            plane_distance: matched.plane_distance,
            projected_coordinates: matched.projected_coordinates,
            path_start: matched.path_start,
            tangent: matched.tangent,
        }))
    }

    /// `rail_query_contexts` (scripts/kart_track_world.py:800-815).
    fn rail_query_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        let contexts = self.rail_contexts()?;
        Some(contexts.iter().map(rail_context_json).collect())
    }

    /// `rail_entry_contexts` (scripts/kart_track_world.py:817-830).
    fn rail_entry_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        let entries = self.rail_entries()?;
        Some(entries.iter().map(rail_entry_json).collect())
    }

    /// `query_moving_contact` (scripts/kart_track_world.py:725-798): nearest
    /// time-sampled dynamic road plus its forward-difference rigid state.
    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        if delta_ms <= 0 || max_distance < 0.0 {
            // Python raises ValueError; surfaced as absence here.
            return Some(None);
        }
        let owner_index = match object_id {
            Some(object_id) => {
                let mut parts = object_id.splitn(3, ':');
                parts.next();
                let Some(index) = parts.next() else {
                    return Some(None);
                };
                match index.trim().parse::<i64>() {
                    Ok(index) if index >= 0 => Some(index),
                    _ => return Some(None),
                }
            }
            None => None,
        };
        let time_ms_i32 = i32::try_from(time_ms).ok()?;
        if !self.index.has_moving_road_candidates {
            // No animated-owned road geometry in this track: the renderer
            // provably emits the miss row, whose parse short-circuits below
            // (`!candidate.hit`). Skip the time-sampled flatten entirely.
            return Some(None);
        }
        let mut args = vec![
            self.track_path.to_string_lossy().into_owned(),
            time_ms_i32.to_string(),
        ];
        args.extend(position.iter().map(|value| format!("{}", *value as f32)));
        if let Some(owner_index) = owner_index {
            args.push(owner_index.to_string());
        }
        let text = cli::render_world_query_moving(&args).ok()?;
        let candidate = parse_moving_row(&TsvTable::parse(&text));
        if !candidate.hit || candidate.distance > max_distance {
            return Some(None);
        }
        let owner = candidate.owner_index;
        let state = self.dynamic_object_state(owner, time_ms, delta_ms)?;
        let state_after = self.dynamic_object_state(owner, time_ms + delta_ms, delta_ms)?;
        let candidate_json = jmap(vec![
            ("track_id", jstr(candidate.track_id.clone())),
            ("hit", jbool(true)),
            ("time_ms", jint(candidate.time_ms)),
            ("owner_index", jint(candidate.owner_index)),
            ("owner_node", jstr(candidate.owner_node.clone())),
            ("node", jstr(candidate.node.clone())),
            ("tri_index", jint(candidate.tri_index)),
            ("road_surface", jstr(candidate.road_surface.clone())),
            ("surface_id", jint(candidate.surface_id)),
            ("road_rail", jstr(candidate.road_rail.clone())),
            ("distance", jnum(candidate.distance)),
            ("point", jvec3(candidate.point)),
            ("normal", jvec3(candidate.normal)),
            (
                "triangle",
                JsonValue::Array(candidate.triangle.into_iter().map(jvec3).collect()),
            ),
            (
                "object_state",
                JsonValue::Object(dynamic_state_json(&state)),
            ),
            (
                "object_state_after",
                JsonValue::Object(dynamic_state_json(&state_after)),
            ),
            (
                "object_id",
                jstr(format!(
                    "{}:{}:{}",
                    candidate.track_id, candidate.owner_index, candidate.owner_node
                )),
            ),
            (
                "authority",
                jstr("time_sampled_track_prs_dynamic_road_geometry_and_rigid_state"),
            ),
            ("contact_active", jbool(true)),
            ("object_position", jvec3(state.position)),
            ("object_velocity", jvec3(state.velocity)),
            ("object_matrix", jvec9(state.matrix)),
            ("object_angular_velocity", jvec3(state.angular_velocity)),
            ("contact_normal_world", jvec3(candidate.normal)),
        ]);
        Some(Some(candidate_json))
    }

    /// `query_checkpoint_target_candidate`
    /// (scripts/kart_track_world.py:658-680).
    fn query_checkpoint_target_candidate(
        &self,
        position: [f64; 3],
        radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        if radius < 0.0 {
            // Python raises ValueError; surfaced as absence here.
            return Some(None);
        }
        let mut best: Option<(f64, &CheckpointMarker)> = None;
        for marker in self.checkpoint_markers() {
            let distance_sq = sum_f64(&[
                (position[0] - marker.position[0]).powi(2),
                (position[1] - marker.position[1]).powi(2),
                (position[2] - marker.position[2]).powi(2),
            ]);
            if distance_sq <= radius * radius
                && best.as_ref().is_none_or(|current| distance_sq < current.0)
            {
                best = Some((distance_sq, marker));
            }
        }
        Some(best.map(|(distance_sq, marker)| CheckpointTargetCandidate {
            marker_id: marker.marker_id.clone(),
            marker_ordinal: marker.marker_ordinal,
            position: marker.position,
            distance: distance_sq.sqrt(),
            authority: "decoded_KAP_event_marker_proximity_candidate".to_string(),
        }))
    }

    /// `query_checkpoint_respawn` (scripts/kart_track_world.py:849-926):
    /// nearest authored course segment, one metre behind the target along
    /// the segment tangent.
    fn query_checkpoint_respawn(
        &self,
        _position: [f64; 3],
        checkpoint_target: [f64; 3],
        checkpoint_gate: bool,
        mode_family: &str,
        max_course_distance: f64,
    ) -> Option<Option<CheckpointRespawnCandidate>> {
        if mode_family != "c016" && mode_family != "c72c" {
            // Python raises ValueError; surfaced as absence here.
            return Some(None);
        }
        if max_course_distance < 0.0 {
            return Some(None);
        }
        if !checkpoint_gate {
            return Some(None);
        }
        let contexts = self.rail_contexts()?;
        struct SegmentBest {
            distance_sq: f64,
            context_index: i64,
            road: String,
            segment_index: i64,
            segment_fraction: f64,
            segment_length: f64,
            tangent: [f64; 3],
        }
        let mut best: Option<SegmentBest> = None;
        for context in contexts.iter() {
            for (segment_index, pair) in context.records.windows(2).enumerate() {
                let start = pair[0].position;
                let end = pair[1].position;
                let delta = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                let length_sq = dot3_sum(delta, delta);
                if length_sq <= 1.0e-12 {
                    continue;
                }
                let alpha = max2_f64(
                    0.0,
                    min2_f64(
                        1.0,
                        dot3_sum(
                            [
                                checkpoint_target[0] - start[0],
                                checkpoint_target[1] - start[1],
                                checkpoint_target[2] - start[2],
                            ],
                            delta,
                        ) / length_sq,
                    ),
                );
                let projection = [
                    start[0] + alpha * delta[0],
                    start[1] + alpha * delta[1],
                    start[2] + alpha * delta[2],
                ];
                let distance_sq = dot3_sum(
                    [
                        checkpoint_target[0] - projection[0],
                        checkpoint_target[1] - projection[1],
                        checkpoint_target[2] - projection[2],
                    ],
                    [
                        checkpoint_target[0] - projection[0],
                        checkpoint_target[1] - projection[1],
                        checkpoint_target[2] - projection[2],
                    ],
                );
                if best
                    .as_ref()
                    .is_none_or(|current| distance_sq < current.distance_sq)
                {
                    let length = length_sq.sqrt();
                    best = Some(SegmentBest {
                        distance_sq,
                        context_index: context.context_index,
                        road: context.road.clone(),
                        segment_index: segment_index as i64,
                        segment_fraction: alpha,
                        segment_length: length,
                        tangent: [delta[0] / length, delta[1] / length, delta[2] / length],
                    });
                }
            }
        }
        let Some(best) = best else {
            return Some(None);
        };
        if best.distance_sq > max_course_distance * max_course_distance {
            return Some(None);
        }
        let reference = [
            checkpoint_target[0] - best.tangent[0],
            checkpoint_target[1] - best.tangent[1],
            checkpoint_target[2] - best.tangent[2],
        ];
        Some(Some(CheckpointRespawnCandidate {
            course_reference_position: reference,
            sample_result_zero: true,
            course_context_id: jint(best.context_index),
            course_road: best.road,
            course_segment_index: best.segment_index,
            course_segment_fraction: best.segment_fraction,
            // The Python selector carries the vec3 projection point here;
            // the trait's f64 field takes the scalar projection distance
            // along the segment. The engine never reads this field
            // (kart-physics/src/engine.rs:8850-8855 consumes only
            // course_reference_position / sample_result_zero / authority).
            course_projection: best.segment_fraction * best.segment_length,
            course_distance: best.distance_sq.sqrt(),
            course_tangent: best.tangent,
            authority: Some("track_ToRoad_forward_course_nearest_segment_reference".to_string()),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_path(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../")
            .join(relative)
    }

    fn factory_world() -> TrackWorld {
        TrackWorld::from_path(repo_path("analysis/rail/factory_I02.track.1s"))
            .expect("factory_I02 track loads")
    }

    fn f64_text(text: &str) -> f64 {
        text.parse::<f64>().unwrap()
    }

    #[test]
    fn factory_big_gear_is_the_only_p1_enabled_event_obstacle() {
        let world = factory_world();
        assert_eq!(world.event_objects().len(), 5);
        let gear = world
            .event_objects()
            .iter()
            .find(|object| object.event_index == 1743)
            .expect("decoded big-gear event");
        assert_eq!(gear.class_name, "ToMovableObject");
        assert_eq!(gear.name, "mo_dummy01-톱니바퀴a");
        assert_eq!(gear.object_type.as_deref(), Some("obstacle"));
        assert_eq!(gear.position, FACTORY_BIG_GEAR_POSITION);
        assert_eq!(
            gear.disposition,
            EventPhysicsDisposition::ContactDynamicObstacle
        );
        assert!(world
            .event_objects()
            .iter()
            .filter(|object| object.event_index != 1743)
            .all(|object| {
                object.disposition == EventPhysicsDisposition::VisualOnlyUnadjudicated
            }));
        assert!(world.has_event_physics_candidates());
        assert_eq!(world.event_obstacles.len(), 1);
        assert_eq!(world.event_obstacles[0].event_index, 1743);
    }

    #[test]
    fn factory_banana_query_uses_authored_prs_and_exact_two_metre_gate() {
        let world = factory_world();
        assert_eq!(world.banana_items.len(), 3);
        assert_eq!(world.banana_items[2].event_index, 2150);
        assert_eq!(world.banana_items[2].object_id, "banana:0xF0000045");
        for expected in [
            (0usize, 1762i64, "banana:0xF0000039"),
            (1usize, 1818i64, "banana:0xF000003B"),
            (2usize, 2150i64, "banana:0xF0000045"),
        ] {
            let model = &world.banana_items[expected.0];
            let matrix = world
                .track
                .flatten_subtree_at(model.subtree_root, 1400)
                .first()
                .expect("banana root sample")
                .1;
            let hit = world
                .query_banana_trigger(
                    [
                        f64::from(matrix[12]),
                        f64::from(matrix[13]),
                        f64::from(matrix[14]),
                    ],
                    1400,
                )
                .expect("each authored banana center is inside its gate");
            assert_eq!(hit.event_index, expected.1);
            assert_eq!(hit.object_id, expected.2);
        }
        let item = &world.banana_items[0];
        let world_matrix = world
            .track
            .flatten_subtree_at(item.subtree_root, 1400)
            .first()
            .expect("banana root sample")
            .1;
        let position = [
            f64::from(world_matrix[12]),
            f64::from(world_matrix[13]),
            f64::from(world_matrix[14]),
        ];
        let hit = world
            .query_banana_trigger(position, 1400)
            .expect("root center is inside gate");
        assert_eq!(hit.object_id, "banana:0xF0000039");
        assert_eq!(hit.event_index, 1762);
        assert_eq!(hit.distance, 0.0);
        assert_eq!(hit.radius, 2.0);
        assert!(world
            .query_banana_trigger([position[0] + 2.01, position[1], position[2]], 1400)
            .is_none());
    }

    #[test]
    fn factory_big_gear_query_reaches_run16_primary_face() {
        let world = factory_world();
        let contacts = world
            .query_event_obstacle_contacts(
                [162.638_184, 844.211_121, 68.182_640_1],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
                421,
                1,
            )
            .expect("event-obstacle query capability");
        assert_eq!(contacts.len(), 1);
        let contact = &contacts[0];
        assert_eq!(contact.event_index, 1743);
        assert_eq!(contact.object_id, "event:1743");
        assert_eq!(contact.node, "Object12_Geom");
        assert_eq!(contact.tri_index, 2);
        assert_eq!(
            contact.centroid,
            [162.639_816_284, 844.219_543_457, 68.183_319_092]
        );
        assert_eq!(
            contact.normal,
            [-0.904_148_221, 0.427_219_033, 0.000_005_809]
        );
        assert_eq!(contact.motion, [-12.084_959_984, 3.906_249_762, 0.0]);

        // run16 seq901418 is the same authored face at a runtime phase
        // whose clock origin differs from the track PRS loop.  The matched
        // phase proves that decoded event geometry supplies all three
        // upstream formula inputs, including rotating-surface motion.
        let observed_centroid = [162.638_184, 844.211_121, 68.182_640_1];
        let observed_normal = [-0.904_148_161, 0.427_219_033, 0.000_005_808_697_1];
        let observed_motion = [-12.069_702_1, 4.006_778_72, 0.0];
        assert!(contact
            .centroid
            .iter()
            .zip(observed_centroid)
            .all(|(actual, expected)| (actual - expected).abs() < 0.01));
        assert!(contact
            .normal
            .iter()
            .zip(observed_normal)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-6));
        assert!(contact
            .motion
            .iter()
            .zip(observed_motion)
            .all(|(actual, expected)| (actual - expected).abs() < 0.11));
    }

    #[test]
    fn orthonormal_column_axes_identity_roundtrip() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(orthonormal_column_axes(identity), Some(identity));
    }

    #[test]
    fn orthonormal_column_axes_repairs_drift() {
        // Row-major matrix with a slightly sheared first column.
        let drifted = [1.0, 0.0, 0.0, 0.001, 1.0, 0.0, 0.0, 0.0, 1.0];
        let axes = orthonormal_column_axes(drifted).expect("axes");
        let first = [axes[0], axes[1], axes[2]];
        let second = [axes[3], axes[4], axes[5]];
        let third = [axes[6], axes[7], axes[8]];
        assert!((dot3_sum(first, first) - 1.0).abs() < 1e-15);
        assert!((dot3_sum(second, second) - 1.0).abs() < 1e-15);
        assert!(dot3_sum(first, second).abs() < 1e-15);
        let cross = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ];
        assert_eq!(third, cross);
        // Right-handed: third keeps a positive component along column two.
        assert!(dot3_sum(third, [0.0, 0.0, 1.0]) > 0.0);
    }

    #[test]
    fn orthonormal_column_axes_rejects_degenerate() {
        assert_eq!(orthonormal_column_axes([0.0; 9]), None);
    }

    #[test]
    fn rail_triangle_direction_picks_longest_edge_with_positive_dominant() {
        let direction =
            rail_triangle_direction([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
                .expect("direction");
        let scale = 1.25_f64.sqrt();
        assert!((direction[0] - 1.0 / scale).abs() < 1e-15);
        assert!((direction[1] + 0.5 / scale).abs() < 1e-15);
        assert_eq!(direction[2], 0.0);
    }

    #[test]
    fn match_rail_entry_boundary_plane_and_edges() {
        // Strip-ordered (not winding-ordered) authored rectangle.
        let entry = RailEntry {
            context_index: 7,
            road: "RoadObj01".to_string(),
            element: 5,
            marker: "start".to_string(),
            boundary: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            path_start: [1.0, -1.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            normal_or_side: [0.0, 0.0, 1.0],
            ..RailEntry::default()
        };
        let inside = match_rail_entry_boundary([1.0, 1.0, 0.02], &entry, 0.05, 0.05)
            .expect("inside the plane band");
        assert!((inside.plane_distance - 0.02).abs() < 1e-15);
        assert_eq!(inside.context_index, 7);
        assert!((inside.tangent_progress - 2.0).abs() < 1e-15);
        // Outside the plane band.
        assert!(match_rail_entry_boundary([1.0, 1.0, 0.1], &entry, 0.05, 0.05).is_none());
        // Far beyond the authored span: projected coordinates reject it.
        // (With an equal-length offset set the fitted basis is diagonal, so
        // the min/max projection box is the diamond through the vertices —
        // (3,1,0) lands exactly on it; (4,1,0) is outside. This mirrors the
        // Python algorithm's acceptance region exactly.)
        assert!(match_rail_entry_boundary([4.0, 1.0, 0.0], &entry, 0.05, 0.05).is_none());
        // Inside the band and inside the edge tolerance margin.
        assert!(match_rail_entry_boundary([2.02, 1.0, 0.0], &entry, 0.05, 0.05).is_some());
    }

    #[test]
    fn checkpoint_marker_spatial_dedup() {
        let rows = vec![
            KapMarkerRow {
                track_id: "t".to_string(),
                marker_ordinal: 0,
                position: [0.0, 0.0, 0.0],
                ..KapMarkerRow::default()
            },
            KapMarkerRow {
                track_id: "t".to_string(),
                marker_ordinal: 1,
                position: [0.5, 0.0, 0.0],
                ..KapMarkerRow::default()
            },
            KapMarkerRow {
                track_id: "t".to_string(),
                marker_ordinal: 2,
                position: [5.0, 0.0, 0.0],
                ..KapMarkerRow::default()
            },
        ];
        let clusters = cluster_checkpoint_markers(&rows);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].marker_id, "t:0");
        assert_eq!(clusters[0].marker_ordinal, 0);
        assert_eq!(clusters[0].source_count, 2);
        assert_eq!(clusters[1].marker_id, "t:1");
        assert_eq!(clusters[1].marker_ordinal, 2);
    }

    #[test]
    fn takeoff_selects_approaching_jump_face() {
        let planes = vec![
            WorldGroundResult {
                node: "road".to_string(),
                tri_index: 3,
                road_surface: "jump".to_string(),
                normal: [0.0, 0.0, 1.0],
                ..WorldGroundResult::default()
            },
            WorldGroundResult {
                node: "road".to_string(),
                tri_index: 4,
                road_surface: String::new(),
                normal: [0.0, 0.0, 1.0],
                ..WorldGroundResult::default()
            },
        ];
        let world = factory_world();
        // Descending onto the jump face: selected.
        let candidate = world
            .query_takeoff_contact([0.0; 3], [0.0, 0.0, -2.0], &planes, &[1, 1])
            .expect("capability")
            .expect("candidate");
        assert_eq!(candidate.selected, Some(true));
        assert_eq!(candidate.normal, Some([0.0, 0.0, 1.0]));
        assert_eq!(candidate.face_id.as_deref(), Some("road:3"));
        // Receding from the jump face: no candidate.
        assert!(world
            .query_takeoff_contact([0.0; 3], [0.0, 0.0, 2.0], &planes, &[1, 1])
            .expect("capability")
            .is_none());
    }

    #[test]
    fn takeoff_contains_factory_hard_support_and_retains_desert_face() {
        let factory = WorldGroundResult {
            node: "RoadObj01@s2".to_string(),
            tri_index: 11,
            normal: [0.0831683353, -0.0000397030781, 0.99653548],
            ..WorldGroundResult::default()
        };
        let world = factory_world();
        assert!(world
            .query_takeoff_contact(
                [480.0, 695.0, 47.0],
                [-45.2024803, 0.967220724, 1.04835331],
                &[factory.clone(), factory],
                &[1, 1],
            )
            .expect("capability")
            .is_none());

        let desert_anchor = WorldGroundResult {
            node: DESERT_I08_UNLABELLED_TAKEOFF_NODE.to_string(),
            tri_index: DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRI,
            normal: DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_NORMAL,
            triangle: Some(DESERT_I08_UNLABELLED_TAKEOFF_ANCHOR_TRIANGLE),
            ..WorldGroundResult::default()
        };
        let candidate = world
            .query_takeoff_contact(
                [618.351868, 695.0, 42.8869057],
                [55.43853, -0.0000388665831, 0.339421511],
                &[desert_anchor.clone(), desert_anchor],
                &[1, 1],
            )
            .expect("capability")
            .expect("candidate");
        assert_eq!(
            candidate.normal,
            Some(DESERT_I08_UNLABELLED_TAKEOFF_FACE_NORMAL)
        );
        assert_eq!(candidate.face_id.as_deref(), Some("RoadObj01@s4:15"));
        assert_eq!(
            candidate.source.as_deref(),
            Some("track_world_runtime_proven_desert_i08_unlabelled_takeoff_face")
        );
    }

    #[test]
    fn landing_picks_most_opposed_active_normal() {
        let planes = vec![
            WorldGroundResult {
                node: "edge".to_string(),
                tri_index: 0,
                normal: [0.2, 0.0, 0.979795897],
                ..WorldGroundResult::default()
            },
            WorldGroundResult {
                node: "edge".to_string(),
                tri_index: 1,
                normal: [-0.2, 0.0, 0.979795897],
                ..WorldGroundResult::default()
            },
        ];
        let world = factory_world();
        let candidate = world
            .query_landing_contact([0.0; 3], [0.5, 0.0, -2.0], &planes, &[0, 1])
            .expect("capability")
            .expect("candidate");
        // velocity . n1 = -0.1 - 1.96 = -2.06 < velocity . n0 = 0.1 - 1.96,
        // so the (-0.2, ...) face is the one being landed on.
        assert_eq!(candidate.normal, Some([-0.2, 0.0, 0.979795897]));
        assert_eq!(candidate.face_id.as_deref(), Some("edge:1"));
    }

    // --- Integration: typed API vs frozen TSV baseline values -------------

    #[test]
    fn factory_query_ground_matches_baseline_row() {
        let world = factory_world();
        let result = world.query_ground(
            [438.71028934642425, 726.4575481429745, 53.43141470976714],
            [
                -0.810520753818237,
                0.5857098926156636,
                0.00017123686760357748,
            ],
            0.65,
        );
        assert_eq!(result.track_id, "rail");
        assert_eq!(result.node, "Object13_Geom");
        assert_eq!(result.tri_index, 0);
        assert_eq!(result.distance, f64_text("0.499980986"));
        assert_eq!(
            result.point,
            [
                f64_text("439.115539551"),
                f64_text("726.164672852"),
                f64_text("53.431331635")
            ]
        );
        assert_eq!(
            result.normal,
            [
                f64_text("-0.810520768"),
                f64_text("0.585709929"),
                f64_text("0.000171237")
            ]
        );
        assert_eq!(
            result.contact_basis_44,
            Some([
                f64_text("0.000230898"),
                f64_text("0.000027164"),
                f64_text("1.000000000")
            ])
        );
        assert!(result.triangle.is_some());
    }

    #[test]
    fn factory_sweep_and_overlap_match_baseline_rows() {
        let world = factory_world();
        let start = [437.4945082156969, 727.336112981898, 53.43167156506855];
        let end = [440.7365912309698, 724.9932734114353, 53.43098661759813];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = 0.9950041652780258_f64;
        let s = 0.09983341664682815_f64;
        // Row-major matrix whose columns are the baseline ROT_AXES.
        let rotated = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
        let hits = world
            .sweep_obb_rotating_all(start, end, identity, rotated, [0.4, 0.4, 0.4])
            .expect("capability");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].loop_index, 0);
        assert_eq!(hits[0].fraction, f64_text("0.359643996"));
        assert_eq!(hits[0].node, "Object13_Geom");
        assert_eq!(
            hits[0].normal_authority,
            "CollisionFace_authored_triangle_winding"
        );
        assert!(hits[0].contact_basis_44.is_some());

        let overlaps = world
            .overlap_obb_all(
                [439.11554972333334, 726.1646931966667, 53.43132909133334],
                identity,
                [0.4, 0.4, 0.4],
            )
            .expect("capability");
        assert_eq!(overlaps.len(), 2);
        assert_eq!(
            overlaps[0].center,
            [
                f64_text("439.115539551"),
                f64_text("726.164672852"),
                f64_text("53.431327820")
            ]
        );

        // Static-pose sweep delegates to overlap_obb_all (Python shortcut).
        let static_hits = world
            .sweep_obb_rotating_all(
                [439.11554972333334, 726.1646931966667, 53.43132909133334],
                [439.11554972333334, 726.1646931966667, 53.43132909133334],
                identity,
                identity,
                [0.4, 0.4, 0.4],
            )
            .expect("capability");
        assert_eq!(static_hits.len(), overlaps.len());
    }

    #[test]
    fn factory_rail_assets_match_baseline_counts() {
        let world = factory_world();
        let contexts = world.rail_query_contexts().expect("capability");
        let record_count: usize = contexts
            .iter()
            .map(|context| match &context["records"] {
                JsonValue::Array(records) => records.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(record_count, 236);
        let entries = world.rail_entry_contexts().expect("capability");
        let boundary_count: usize = entries
            .iter()
            .map(|entry| match &entry["boundary"] {
                JsonValue::Array(vertices) => vertices.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(boundary_count, 152);
        // A far-away point crosses no entry plane: no rail selector.
        assert!(world
            .query_rail_contact([100000.0, 100000.0, 100000.0], 0, None)
            .expect("capability")
            .is_none());
    }

    #[test]
    fn factory_checkpoint_target_and_respawn() {
        let world = factory_world().with_kap_paths([
            repo_path("analysis/factory_I02_1.kap"),
            repo_path("analysis/factory_I02_2.kap"),
            repo_path("analysis/factory_I02_3.kap"),
        ]);
        let marker = world
            .query_checkpoint_target_candidate([554.110168457, 694.975097656, 47.0], 2.0)
            .expect("capability")
            .expect("marker inside radius");
        assert_eq!(marker.marker_id, "factory_I02:0");
        assert_eq!(marker.marker_ordinal, 0);
        assert_eq!(marker.distance, 0.0);
        assert_eq!(
            marker.authority,
            "decoded_KAP_event_marker_proximity_candidate"
        );
        // Far outside the candidate radius: no marker.
        assert!(world
            .query_checkpoint_target_candidate([0.0, 0.0, 0.0], 2.0)
            .expect("capability")
            .is_none());

        let respawn = world
            .query_checkpoint_respawn(
                [0.0; 3],
                [554.110168457, 694.975097656, 47.0],
                true,
                "c016",
                20.0,
            )
            .expect("capability")
            .expect("course reference near the target");
        assert!(respawn.sample_result_zero);
        assert!(respawn.course_distance <= 20.0);
        assert_eq!(
            respawn.authority.as_deref(),
            Some("track_ToRoad_forward_course_nearest_segment_reference")
        );
        // Gate closed: no respawn selector.
        assert!(world
            .query_checkpoint_respawn(
                [0.0; 3],
                [554.110168457, 694.975097656, 47.0],
                false,
                "c016",
                20.0
            )
            .expect("capability")
            .is_none());
    }

    #[test]
    fn factory_moving_contact_misses_static_road() {
        let world = factory_world();
        // Baseline world-query-moving at the road center is hit=0 (no
        // dynamic road near), so the composition returns no candidate.
        assert!(world
            .query_moving_contact(
                [439.11554972333334, 726.1646931966667, 53.43132909133334],
                0,
                None,
                1,
                1.5,
            )
            .expect("capability")
            .is_none());
    }

    #[test]
    fn factory_capabilities_full_asset_set() {
        let world = factory_world();
        let caps = world.capabilities();
        assert!(caps.overlap_obb_all);
        assert!(caps.query_takeoff_contact);
        assert!(caps.query_landing_contact);
        assert!(caps.query_rail_contact);
        assert!(caps.query_rail_entry_boundary);
        assert!(caps.query_moving_contact);
        assert!(caps.query_checkpoint_target_candidate);
        assert!(caps.query_checkpoint_respawn);
        assert!(caps.sweep_obb_rotating_all);
        assert!(caps.sweep_obb_rotating);
        assert!(caps.sweep_point);
        assert!(caps.query_at_step_start);
        assert!(caps.rail_query_contexts);
        assert!(caps.rail_entry_contexts);
        assert!(!caps.query_network_correction);
        assert!(!caps.query_distinct_water_volume);
    }

    // -----------------------------------------------------------------------
    // P2 grid-equivalence gate: the migrated typed methods (in-memory
    // selection + q9 quantization) must be bit-identical to the legacy
    // render-TSV-then-parse path they replaced. The legacy path below is a
    // verbatim copy of the pre-P2 implementation, kept as the reference.
    // -----------------------------------------------------------------------

    fn parse_ground_row(table: &TsvTable, row: usize) -> GroundRow {
        let triangle = if (0..3).all(|i| {
            ["x", "y", "z"]
                .iter()
                .all(|axis| table.has_column(&format!("v{i}{axis}")))
        }) {
            Some([
                pf3(table, row, ["v0x", "v0y", "v0z"]),
                pf3(table, row, ["v1x", "v1y", "v1z"]),
                pf3(table, row, ["v2x", "v2y", "v2z"]),
            ])
        } else {
            None
        };
        let contact_basis_44 = if ["x", "y", "z"].iter().all(|axis| {
            !table
                .get(row, &format!("contact_basis_44_{axis}"))
                .is_empty()
        }) {
            Some(pf3(
                table,
                row,
                [
                    "contact_basis_44_x",
                    "contact_basis_44_y",
                    "contact_basis_44_z",
                ],
            ))
        } else {
            None
        };
        GroundRow {
            track_id: table.get(row, "track_id").to_string(),
            node: table.get(row, "node").to_string(),
            tri_index: pi(table.get(row, "tri_index")),
            road_surface: table.get(row, "road_surface").to_string(),
            surface_id: pi(table.get(row, "surface_id")),
            road_rail: table.get(row, "road_rail").to_string(),
            distance: pf(table.get(row, "distance")),
            point: pf3(table, row, ["qx", "qy", "qz"]),
            normal: pf3(table, row, ["nx", "ny", "nz"]),
            triangle,
            contact_basis_44,
        }
    }

    impl GroundRow {
        fn into_result(self) -> WorldGroundResult {
            WorldGroundResult {
                track_id: self.track_id,
                node: self.node,
                tri_index: self.tri_index,
                road_surface: self.road_surface,
                surface_id: self.surface_id,
                road_rail: self.road_rail,
                distance: self.distance,
                point: self.point,
                normal: self.normal,
                triangle: self.triangle,
                contact_basis_44: self.contact_basis_44,
            }
        }
    }

    /// `parse_world_sweep_tsv` (scripts/kart_track_world.py:308-323).
    #[derive(Debug, Clone, Default)]
    struct SweepRow {
        track_id: String,
        hit: bool,
        fraction: f64,
        node: String,
        tri_index: i64,
        road_surface: String,
        surface_id: i64,
        road_rail: String,
        point: [f64; 3],
        normal: [f64; 3],
    }

    fn parse_sweep_row(table: &TsvTable) -> SweepRow {
        let hit = table.get(0, "hit") == "1";
        let mut row = SweepRow {
            track_id: table.get(0, "track_id").to_string(),
            hit,
            ..SweepRow::default()
        };
        if hit {
            row.fraction = pf(table.get(0, "fraction"));
            row.node = table.get(0, "node").to_string();
            row.tri_index = pi(table.get(0, "tri_index"));
            row.road_surface = table.get(0, "road_surface").to_string();
            row.surface_id = pi(table.get(0, "surface_id"));
            row.road_rail = table.get(0, "road_rail").to_string();
            row.point = pf3(table, 0, ["hx", "hy", "hz"]);
            row.normal = pf3(table, 0, ["nx", "ny", "nz"]);
        }
        row
    }

    /// `parse_world_obb_sweep_tsv` (scripts/kart_track_world.py:345-361).
    #[derive(Debug, Clone, Default)]
    struct ObbSweepRow {
        track_id: String,
        hit: bool,
        fraction: f64,
        node: String,
        tri_index: i64,
        road_surface: String,
        surface_id: i64,
        road_rail: String,
        center: [f64; 3],
        support_point: [f64; 3],
        normal: [f64; 3],
    }

    fn parse_obb_sweep_row(table: &TsvTable) -> ObbSweepRow {
        let hit = table.get(0, "hit") == "1";
        let mut row = ObbSweepRow {
            track_id: table.get(0, "track_id").to_string(),
            hit,
            ..ObbSweepRow::default()
        };
        if hit {
            row.fraction = pf(table.get(0, "fraction"));
            row.node = table.get(0, "node").to_string();
            row.tri_index = pi(table.get(0, "tri_index"));
            row.road_surface = table.get(0, "road_surface").to_string();
            row.surface_id = pi(table.get(0, "surface_id"));
            row.road_rail = table.get(0, "road_rail").to_string();
            row.center = pf3(table, 0, ["cx", "cy", "cz"]);
            row.support_point = pf3(table, 0, ["sx", "sy", "sz"]);
            row.normal = pf3(table, 0, ["nx", "ny", "nz"]);
        }
        row
    }

    /// One `parse_world_obb_overlaps_tsv` hit (scripts/kart_track_world.py:364-401).
    #[derive(Debug, Clone, Default)]
    struct ObbOverlapRow {
        track_id: String,
        loop_index: i64,
        fraction: f64,
        node: String,
        tri_index: i64,
        road_surface: String,
        surface_id: i64,
        road_rail: String,
        center: [f64; 3],
        support_point: [f64; 3],
        ccd_normal: [f64; 3],
        normal: [f64; 3],
        normal_authority: String,
        contact_basis_44: Option<[f64; 3]>,
    }

    fn parse_obb_overlap_rows(table: &TsvTable) -> Vec<ObbOverlapRow> {
        let mut results = Vec::new();
        for row in 0..table.rows.len() {
            if table.get(row, "hit") != "1" {
                continue;
            }
            let ccd_normal = pf3(table, row, ["nx", "ny", "nz"]);
            let has_face_normal = ["x", "y", "z"]
                .iter()
                .all(|axis| !table.get(row, &format!("face_n{axis}")).is_empty());
            let (normal, normal_authority) = if has_face_normal {
                (
                    pf3(table, row, ["face_nx", "face_ny", "face_nz"]),
                    "CollisionFace_authored_triangle_winding",
                )
            } else {
                (ccd_normal, "legacy_CCD_separating_axis")
            };
            let contact_basis_44 = if ["x", "y", "z"].iter().all(|axis| {
                !table
                    .get(row, &format!("contact_basis_44_{axis}"))
                    .is_empty()
            }) {
                Some(pf3(
                    table,
                    row,
                    [
                        "contact_basis_44_x",
                        "contact_basis_44_y",
                        "contact_basis_44_z",
                    ],
                ))
            } else {
                None
            };
            results.push(ObbOverlapRow {
                track_id: table.get(row, "track_id").to_string(),
                loop_index: pi(table.get(row, "loop_index")),
                fraction: pf(table.get(row, "fraction")),
                node: table.get(row, "node").to_string(),
                tri_index: pi(table.get(row, "tri_index")),
                road_surface: table.get(row, "road_surface").to_string(),
                surface_id: pi(table.get(row, "surface_id")),
                road_rail: table.get(row, "road_rail").to_string(),
                center: pf3(table, row, ["cx", "cy", "cz"]),
                support_point: pf3(table, row, ["sx", "sy", "sz"]),
                ccd_normal,
                normal,
                normal_authority: normal_authority.to_string(),
                contact_basis_44,
            });
        }
        results
    }

    impl ObbOverlapRow {
        fn into_result(self) -> WorldObbOverlap {
            WorldObbOverlap {
                track_id: self.track_id,
                hit: true,
                loop_index: self.loop_index,
                fraction: self.fraction,
                node: self.node,
                tri_index: self.tri_index,
                road_surface: self.road_surface,
                surface_id: self.surface_id,
                road_rail: self.road_rail,
                center: self.center,
                support_point: self.support_point,
                ccd_normal: self.ccd_normal,
                normal: self.normal,
                normal_authority: self.normal_authority,
                contact_basis_44: self.contact_basis_44,
                boost_multiplier_64c: false,
            }
        }
    }

    // --- Legacy typed methods (pre-P2 bodies, verbatim) --------------------

    fn legacy_f32_args(world: &TrackWorld, values: &[f64]) -> Vec<String> {
        let mut args = Vec::with_capacity(values.len() + 1);
        args.push(world.track_path.to_string_lossy().into_owned());
        args.extend(values.iter().map(|value| format!("{}", *value as f32)));
        args
    }

    fn legacy_query_ground(
        world: &TrackWorld,
        position: [f64; 3],
        up: [f64; 3],
        min_dot: f64,
    ) -> WorldGroundResult {
        let args = legacy_f32_args(
            world,
            &[
                position[0],
                position[1],
                position[2],
                up[0],
                up[1],
                up[2],
                min_dot,
            ],
        );
        match cli::render_world_query_ground(&args) {
            Ok(text) => parse_ground_row(&TsvTable::parse(&text), 0).into_result(),
            Err(_) => WorldGroundResult::default(),
        }
    }

    fn legacy_overlap_obb_all(
        world: &TrackWorld,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let axes = orthonormal_column_axes(matrix)?;
        let args = legacy_f32_args(
            world,
            &[
                center[0],
                center[1],
                center[2],
                axes[0],
                axes[1],
                axes[2],
                axes[3],
                axes[4],
                axes[5],
                axes[6],
                axes[7],
                axes[8],
                half_extents[0],
                half_extents[1],
                half_extents[2],
            ],
        );
        let text = cli::render_world_overlap_obb_all(&args).ok()?;
        Some(
            parse_obb_overlap_rows(&TsvTable::parse(&text))
                .into_iter()
                .map(ObbOverlapRow::into_result)
                .collect(),
        )
    }

    fn legacy_sweep_obb_rotating(
        world: &TrackWorld,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<WorldObbOverlap> {
        let start_axes = orthonormal_column_axes(start_matrix)?;
        let end_axes = orthonormal_column_axes(end_matrix)?;
        let args = legacy_f32_args(
            world,
            &[
                start[0],
                start[1],
                start[2],
                end[0],
                end[1],
                end[2],
                start_axes[0],
                start_axes[1],
                start_axes[2],
                start_axes[3],
                start_axes[4],
                start_axes[5],
                start_axes[6],
                start_axes[7],
                start_axes[8],
                end_axes[0],
                end_axes[1],
                end_axes[2],
                end_axes[3],
                end_axes[4],
                end_axes[5],
                end_axes[6],
                end_axes[7],
                end_axes[8],
                half_extents[0],
                half_extents[1],
                half_extents[2],
            ],
        );
        let row = match cli::render_world_sweep_obb_rotating(&args) {
            Ok(text) => parse_obb_sweep_row(&TsvTable::parse(&text)),
            Err(_) => ObbSweepRow::default(),
        };
        Some(WorldObbOverlap {
            track_id: row.track_id,
            hit: row.hit,
            loop_index: 0,
            fraction: row.fraction,
            node: row.node,
            tri_index: row.tri_index,
            road_surface: row.road_surface,
            surface_id: row.surface_id,
            road_rail: row.road_rail,
            center: row.center,
            support_point: row.support_point,
            ccd_normal: row.normal,
            normal: row.normal,
            normal_authority: "legacy_CCD_separating_axis".to_string(),
            contact_basis_44: None,
            boost_multiplier_64c: false,
        })
    }

    fn legacy_sweep_obb_rotating_all(
        world: &TrackWorld,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        if start == end && start_matrix == end_matrix {
            return legacy_overlap_obb_all(world, start, start_matrix, half_extents);
        }
        let column_axes = |matrix: [f64; 9]| -> [f64; 9] {
            [
                matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2],
                matrix[5], matrix[8],
            ]
        };
        let start_axes = column_axes(start_matrix);
        let end_axes = column_axes(end_matrix);
        let args = legacy_f32_args(
            world,
            &[
                start[0],
                start[1],
                start[2],
                end[0],
                end[1],
                end[2],
                start_axes[0],
                start_axes[1],
                start_axes[2],
                start_axes[3],
                start_axes[4],
                start_axes[5],
                start_axes[6],
                start_axes[7],
                start_axes[8],
                end_axes[0],
                end_axes[1],
                end_axes[2],
                end_axes[3],
                end_axes[4],
                end_axes[5],
                end_axes[6],
                end_axes[7],
                end_axes[8],
                half_extents[0],
                half_extents[1],
                half_extents[2],
            ],
        );
        let text = cli::render_world_sweep_obb_rotating_all(&args).ok()?;
        Some(
            parse_obb_overlap_rows(&TsvTable::parse(&text))
                .into_iter()
                .map(ObbOverlapRow::into_result)
                .collect(),
        )
    }

    fn legacy_sweep_point(
        world: &TrackWorld,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Option<BTreeMap<String, JsonValue>> {
        let args = legacy_f32_args(
            world,
            &[start[0], start[1], start[2], end[0], end[1], end[2]],
        );
        let row = match cli::render_world_sweep(&args) {
            Ok(text) => parse_sweep_row(&TsvTable::parse(&text)),
            Err(_) => SweepRow::default(),
        };
        let mut entries = vec![
            ("track_id", jstr(row.track_id.clone())),
            ("hit", jbool(row.hit)),
        ];
        if row.hit {
            entries.extend([
                ("fraction", jnum(row.fraction)),
                ("node", jstr(row.node)),
                ("tri_index", jint(row.tri_index)),
                ("road_surface", jstr(row.road_surface)),
                ("surface_id", jint(row.surface_id)),
                ("road_rail", jstr(row.road_rail)),
                ("point", jvec3(row.point)),
                ("normal", jvec3(row.normal)),
            ]);
        }
        Some(jmap(entries))
    }

    fn legacy_rail_contact(
        world: &TrackWorld,
        position: [f64; 3],
        previous_rail_state: i64,
        owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        if previous_rail_state < 0 {
            return Some(None);
        }
        let args = legacy_f32_args(world, &[position[0], position[1], position[2]]);
        let text = cli::render_world_query_rail(&args).ok()?;
        let table = TsvTable::parse(&text);
        let hit = table.get(0, "hit") == "1";
        let candidate = if hit {
            parse_ground_row(&table, 0)
        } else {
            GroundRow {
                track_id: table.get(0, "track_id").to_string(),
                ..GroundRow::default()
            }
        };
        let entry_gate = if hit && previous_rail_state == 0 {
            world.match_entry(position, Some(&candidate.node), 0.05)
        } else {
            None
        };
        let entry_forward_crossed = entry_gate
            .as_ref()
            .is_some_and(|gate| gate.tangent_progress >= 0.0);
        let contact_distance_limit = if previous_rail_state == 0 {
            0.3203125
        } else {
            4.00
        };
        if !hit
            || candidate.road_rail != "highSpeed"
            || (previous_rail_state != 0 && candidate.distance > contact_distance_limit)
            || (previous_rail_state == 0 && !entry_forward_crossed)
        {
            return Some(None);
        }
        let triangle = candidate.triangle?;
        let rail_dir = rail_triangle_direction(triangle)?;
        let rail_state = if previous_rail_state == 0 {
            1
        } else {
            previous_rail_state
        };
        let mut selector = jmap(vec![
            ("contact_active", jbool(true)),
            ("rail_state", jint(rail_state)),
            (
                "path_id",
                jstr(format!(
                    "{}:{}:{}",
                    candidate.track_id, candidate.node, candidate.road_rail
                )),
            ),
            ("owner_id", owner_id.map_or(JsonValue::Null, jstr)),
            (
                "runtime",
                JsonValue::Object(jmap(vec![
                    ("rail_dir", jvec3(rail_dir)),
                    ("accel_be0", jnum(2.0)),
                    ("obf_6e8", jnum(2304.0)),
                    ("grip_720", jnum(2070.0)),
                    ("gravity_be8", jnum(0.5)),
                    ("obf_808", jnum(1.84949994)),
                    ("obf_814", jnum(1.49399996)),
                    ("min_velocity", jnum(83.3333359)),
                    ("max_velocity", jnum(97.2222214)),
                ])),
            ),
            (
                "geometry",
                JsonValue::Object(rail_candidate_json(&candidate)),
            ),
            (
                "entry_gate",
                entry_gate
                    .as_ref()
                    .map(|gate| JsonValue::Object(rail_entry_match_json(gate)))
                    .unwrap_or(JsonValue::Null),
            ),
            ("entry_forward_crossed", jbool(entry_forward_crossed)),
            ("contact_distance_limit", jnum(contact_distance_limit)),
            (
                "authority",
                jstr(
                    "track_ToRoad_entry_boundary_forward_crossing_then_\
                     road_rail_highSpeed_nearest_triangle_and_factory_I02_\
                     runtime_profile",
                ),
            ),
            (
                "scope",
                jstr(
                    "factory_I02 highSpeed initial capture on the raw authored \
                     ToRoad entry boundary and retained-owner sustain within the \
                     decoded 4.00 m path metric threshold",
                ),
            ),
        ]);
        if (1..=3).contains(&rail_state) {
            let contexts = world
                .rail_contexts()
                .map(|contexts| {
                    JsonValue::Array(
                        contexts
                            .iter()
                            .map(|context| JsonValue::Object(rail_context_json(context)))
                            .collect(),
                    )
                })
                .unwrap_or(JsonValue::Null);
            selector.extend(jmap(vec![
                ("registered_query_contexts", contexts),
                ("contact_exists", jbool(true)),
                ("contact_identifier_nonempty", jbool(true)),
                ("metric_threshold", jnum(4.0)),
                ("height_bias", jnum(1.0)),
                ("alignment_primary_scale_ms", jnum(10.0)),
                ("alignment_stable_initial_ms", jnum(100.0)),
            ]));
        }
        Some(Some(selector))
    }

    // --- Bit-exact comparators (-0.0 and NaN are semantic: compare bits) ---

    fn bits_eq(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    fn vec3_bits_eq(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| bits_eq(a[i], b[i]))
    }

    fn opt_vec3_bits_eq(a: Option<[f64; 3]>, b: Option<[f64; 3]>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => vec3_bits_eq(a, b),
            _ => false,
        }
    }

    fn json_bits_eq(a: &JsonValue, b: &JsonValue) -> bool {
        match (a, b) {
            (JsonValue::Null, JsonValue::Null) => true,
            (JsonValue::Bool(a), JsonValue::Bool(b)) => a == b,
            (JsonValue::Number(a), JsonValue::Number(b)) => bits_eq(*a, *b),
            (JsonValue::String(a), JsonValue::String(b)) => a == b,
            (JsonValue::Array(a), JsonValue::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| json_bits_eq(a, b))
            }
            (JsonValue::Object(a), JsonValue::Object(b)) => json_map_bits_eq(a, b),
            _ => false,
        }
    }

    fn json_map_bits_eq(a: &BTreeMap<String, JsonValue>, b: &BTreeMap<String, JsonValue>) -> bool {
        a.len() == b.len()
            && a.iter()
                .all(|(key, value)| b.get(key).is_some_and(|other| json_bits_eq(value, other)))
    }

    fn ground_result_eq(a: &WorldGroundResult, b: &WorldGroundResult) -> bool {
        a.track_id == b.track_id
            && a.node == b.node
            && a.tri_index == b.tri_index
            && a.road_surface == b.road_surface
            && a.surface_id == b.surface_id
            && a.road_rail == b.road_rail
            && bits_eq(a.distance, b.distance)
            && vec3_bits_eq(a.point, b.point)
            && vec3_bits_eq(a.normal, b.normal)
            && match (&a.triangle, &b.triangle) {
                (None, None) => true,
                (Some(a), Some(b)) => a.iter().zip(b).all(|(a, b)| vec3_bits_eq(*a, *b)),
                _ => false,
            }
            && opt_vec3_bits_eq(a.contact_basis_44, b.contact_basis_44)
    }

    fn overlap_eq(a: &WorldObbOverlap, b: &WorldObbOverlap) -> bool {
        a.track_id == b.track_id
            && a.hit == b.hit
            && a.loop_index == b.loop_index
            && bits_eq(a.fraction, b.fraction)
            && a.node == b.node
            && a.tri_index == b.tri_index
            && a.road_surface == b.road_surface
            && a.surface_id == b.surface_id
            && a.road_rail == b.road_rail
            && vec3_bits_eq(a.center, b.center)
            && vec3_bits_eq(a.support_point, b.support_point)
            && vec3_bits_eq(a.ccd_normal, b.ccd_normal)
            && vec3_bits_eq(a.normal, b.normal)
            && a.normal_authority == b.normal_authority
            && opt_vec3_bits_eq(a.contact_basis_44, b.contact_basis_44)
            && a.boost_multiplier_64c == b.boost_multiplier_64c
    }

    fn overlaps_eq(a: &[WorldObbOverlap], b: &[WorldObbOverlap]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| overlap_eq(a, b))
    }

    // --- The lattice gate --------------------------------------------------

    #[test]
    fn typed_path_matches_legacy_tsv_path_on_coordinate_grid() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = 0.9950041652780258_f64;
        let s = 0.09983341664682815_f64;
        let rotated = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
        let half = [0.45, 0.5, 0.4];
        let maps = [
            "analysis/rail/factory_I02.track.1s",
            "analysis/motion_topology/desert_I08.track.1s",
            "analysis/motion_topology/abyss_I03.track.1s",
        ];
        let mut comparisons = 0usize;
        for map in maps {
            let path = repo_path(map);
            if !path.is_file() {
                eprintln!("skipping {map}: track asset not present");
                continue;
            }
            let world = TrackWorld::from_path(&path).expect("track loads");
            // Road-soup bounds drive the lattice: hit regions inside, miss
            // regions outside and far away.
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for entry in &world.index.road {
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(f64::from(entry.aabb_min[axis]));
                    hi[axis] = hi[axis].max(f64::from(entry.aabb_max[axis]));
                }
            }
            assert!(lo[0].is_finite(), "{map} has road triangles");

            // One full per-point comparison battery. Rotating-OBB CCD is the
            // one expensive predicate in a debug build, so callers gate it.
            let mut check_point = |point: [f64; 3], run_rotating: bool| {
                let [x, y, z] = point;
                for (up, min_dot) in [
                    ([0.0, 0.0, 1.0], 0.65),
                    ([0.1, -0.2, 0.975], 0.5),
                    // Curb-drift miss path: a rolled kart sends its 1000 m
                    // suspension ray almost horizontally. This specifically
                    // gates the segment DDA + indexed nearest fallback.
                    ([-0.866_025_03, 0.000_368_46, 0.500_000_5], 0.65),
                ] {
                    let new = world.query_ground(point, up, min_dot);
                    let old = legacy_query_ground(&world, point, up, min_dot);
                    assert!(
                        ground_result_eq(&new, &old),
                        "query_ground diverged at {point:?} up={up:?} on {map}:\nnew={new:?}\nold={old:?}"
                    );
                    comparisons += 1;
                }
                for matrix in [identity, rotated] {
                    let new = world.overlap_obb_all(point, matrix, half);
                    let old = legacy_overlap_obb_all(&world, point, matrix, half);
                    assert_eq!(
                        new.is_some(),
                        old.is_some(),
                        "overlap presence at {point:?} on {map}"
                    );
                    if let (Some(new), Some(old)) = (&new, &old) {
                        assert!(
                            overlaps_eq(new, old),
                            "overlap_obb_all diverged at {point:?} on {map}"
                        );
                    }
                    comparisons += 1;
                }
                let seg_end = [x + 1.7, y - 2.3, z - 0.4];
                let new = world.sweep_point(point, seg_end);
                let old = legacy_sweep_point(&world, point, seg_end);
                assert_eq!(new.is_some(), old.is_some());
                if let (Some(new), Some(old)) = (&new, &old) {
                    assert!(
                        json_map_bits_eq(new, old),
                        "sweep_point diverged at {point:?} on {map}:\nnew={new:?}\nold={old:?}"
                    );
                }
                comparisons += 1;
                let obb_end = [x + 0.8, y + 0.5, z - 0.2];
                if run_rotating {
                    let new = world.sweep_obb_rotating(point, obb_end, identity, rotated, half);
                    let old =
                        legacy_sweep_obb_rotating(&world, point, obb_end, identity, rotated, half);
                    assert_eq!(new.is_some(), old.is_some());
                    if let (Some(new), Some(old)) = (&new, &old) {
                        assert!(
                            overlap_eq(new, old),
                            "sweep_obb_rotating diverged at {point:?} on {map}:\nnew={new:?}\nold={old:?}"
                        );
                    }
                    comparisons += 1;
                    let new = world.sweep_obb_rotating_all(point, obb_end, identity, rotated, half);
                    let old = legacy_sweep_obb_rotating_all(
                        &world, point, obb_end, identity, rotated, half,
                    );
                    assert_eq!(new.is_some(), old.is_some());
                    if let (Some(new), Some(old)) = (&new, &old) {
                        assert!(
                            overlaps_eq(new, old),
                            "sweep_obb_rotating_all diverged at {point:?} on {map}"
                        );
                    }
                    comparisons += 1;
                }
                // Static-pose rotating_all takes the overlap shortcut.
                let new = world.sweep_obb_rotating_all(point, point, rotated, rotated, half);
                let old =
                    legacy_sweep_obb_rotating_all(&world, point, point, rotated, rotated, half);
                assert_eq!(new.is_some(), old.is_some());
                if let (Some(new), Some(old)) = (&new, &old) {
                    assert!(
                        overlaps_eq(new, old),
                        "static-pose sweep_obb_rotating_all diverged at {point:?} on {map}"
                    );
                }
                comparisons += 1;
                for rail_state in [0, 1, 3, 5] {
                    let new = world.query_rail_contact(point, rail_state, Some("owner"));
                    let old = legacy_rail_contact(&world, point, rail_state, Some("owner"));
                    assert_eq!(
                        new.is_some(),
                        old.is_some(),
                        "rail capability at {point:?} on {map}"
                    );
                    if let (Some(new), Some(old)) = (&new, &old) {
                        match (new, old) {
                            (Some(new), Some(old)) => assert!(
                                json_map_bits_eq(new, old),
                                "query_rail_contact diverged at {point:?} state={rail_state} on {map}"
                            ),
                            (None, None) => {}
                            _ => panic!(
                                "query_rail_contact presence diverged at {point:?} state={rail_state} on {map}"
                            ),
                        }
                    }
                    comparisons += 1;
                }
            };

            // Interior lattice: 3x3x3 across the road bounds; rotating CCD
            // on every third point.
            let axis_values = |axis: usize| -> Vec<f64> {
                (0..3)
                    .map(|i| lo[axis] + (hi[axis] - lo[axis]) * (i as f64 / 2.0))
                    .collect()
            };
            let xs = axis_values(0);
            let ys = axis_values(1);
            let zs = axis_values(2);
            let mut point_ordinal = 0usize;
            for &x in &xs {
                for &y in &ys {
                    for &z in &zs {
                        point_ordinal += 1;
                        check_point([x, y, z], point_ordinal % 3 == 1);
                    }
                }
            }
            // Exterior / far-miss points: below, beyond, and very far (these
            // drive the ground fallback to full-grid coverage).
            for (i, point) in [
                [lo[0] - 200.0, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5],
                [(lo[0] + hi[0]) * 0.5, hi[1] + 500.0, (lo[2] + hi[2]) * 0.5],
                [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2] - 200.0],
                [100000.0, 100000.0, 100000.0],
            ]
            .into_iter()
            .enumerate()
            {
                check_point(point, i % 2 == 0);
            }
            // Non-finite input: both paths take the verbatim full scan.
            let nan_point = [f64::NAN, ys[0], zs[0]];
            let new = world.query_ground(nan_point, [0.0, 0.0, 1.0], 0.65);
            let old = legacy_query_ground(&world, nan_point, [0.0, 0.0, 1.0], 0.65);
            assert!(
                ground_result_eq(&new, &old),
                "query_ground(NaN) diverged on {map}:\nnew={new:?}\nold={old:?}"
            );
            comparisons += 1;
            let new = world.sweep_point(nan_point, [ys[0]; 3]);
            let old = legacy_sweep_point(&world, nan_point, [ys[0]; 3]);
            assert_eq!(new.is_some(), old.is_some());
            if let (Some(new), Some(old)) = (&new, &old) {
                assert!(
                    json_map_bits_eq(new, old),
                    "sweep_point(NaN) diverged on {map}"
                );
            }
            comparisons += 1;
        }
        eprintln!("grid equivalence: {comparisons} bit-exact comparisons passed");
        assert!(comparisons > 0);
    }
}
