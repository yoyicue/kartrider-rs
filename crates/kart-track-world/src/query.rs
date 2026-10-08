//! Pre-derived static-geometry query core (P2 of
//! `docs/kart-physics-performance-handoff-2026-07-30.md`).
//!
//! The TSV renderers used to redo the whole load pipeline per query — track
//! parse, tree flatten, strip expansion, world transforms, tag reads,
//! per-triangle normals — before scanning every road triangle. That work is
//! done ONCE per track here: a world-space triangle soup (road-tagged and
//! rail subsets, in the renderer's exact `flatten()` iteration order) plus a
//! uniform-grid candidate index over the road soup.
//!
//! Bit-exactness contract:
//!
//! - The soup stores the exact f32 values the per-query pipeline computed
//!   (same `xform` / `tri_normal` / `shortest_directed_triangle_edge` calls
//!   on the same world triangles), so precomputing changes nothing.
//! - The grid only PROPOSES candidates; the original predicates decide every
//!   result. Candidates are iterated in ascending soup order — the original
//!   scan order — so strict-`<` tie-breaking and `_all` row ordering are
//!   preserved. Registration and lookup both widen by one cell, and lookups
//!   clamp to the registered bounds, so coverage is conservative.
//! - Bounded queries (sweeps / overlaps) gather candidates from the query
//!   volume's AABB; any triangle the predicate could accept overlaps that
//!   AABB, hence sits in a gathered cell.
//! - `query_ground` keeps the renderer's (ray-hit, distance) selection:
//!   pass 1 finds the nearest ray hit among segment-AABB candidates (a
//!   segment/triangle intersection implies AABB overlap, so coverage is
//!   complete); only when no ray hit exists anywhere does pass 2 run the
//!   nearest-point fallback through a normal-partitioned spatial BVH. Its
//!   bounds only prune candidates; leaves still run the original float32
//!   normal and closest-point predicates, with soup-order tie-breaking.
//! - Non-finite inputs (NaN/inf) bypass the grid entirely and run a
//!   verbatim full scan, matching the original loop exactly.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::geom::*;
use crate::track::{
    clean, has_tag, load_static_track, static_track_cache_key, surface_id, tag_attr,
};

/// Grid cell edge. The kart OBB full extent is ~1-2 m, so 4 m keeps every
/// per-step query volume inside a 2-3 cell span per axis while road
/// triangles (a few m each) land in a handful of cells.
const GRID_CELL: f32 = 4.0;

/// One world-space static triangle plus the row metadata the TSV contract
/// carries. Strings are pre-`clean`ed (clean is idempotent, so the
/// renderers' `clean(...)` at format time is a no-op on them) and
/// `surface_id` is precomputed from the RAW surface text exactly like the
/// renderers' `if surface.is_empty() { 0 } else { surface_id(&surface) }`.
#[derive(Clone, Debug)]
pub struct StaticTri {
    pub tri: [P3; 3],
    pub normal: P3,
    pub contact_basis_44: P3,
    pub aabb_min: P3,
    pub aabb_max: P3,
    pub node: Arc<str>,
    pub surface: Arc<str>,
    pub surface_id: u32,
    pub rail: Arc<str>,
    pub tri_index: usize,
    /// The historical kart-autoprobe NavMesh support predicate: road-tagged,
    /// drivable surface, finite vertices, and authored-facing normal z >= .35.
    pub static_support: bool,
}

/// Uniform grid over road-triangle AABBs: sparse cell map, per-cell triangle
/// indices in ascending soup order. Only ever a candidate generator.
struct UniformGrid {
    cell: f32,
    /// Registered cell-coordinate bounds (inclusive), used to clamp lookups.
    bounds_lo: [i32; 3],
    bounds_hi: [i32; 3],
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
    /// Triangles spanning more than [`GRID_MAX_SPAN`] cells on an axis:
    /// registered nowhere, then conservatively filtered by their AABB per
    /// lookup (keeps a track-sized triangle from exploding the cell map).
    overflow: Vec<(u32, P3, P3)>,
}

/// Per-axis cell span above which a triangle goes to the overflow list.
const GRID_MAX_SPAN: i32 = 16;

struct FallbackNode {
    aabb_min: P3,
    aabb_max: P3,
    normal_min: P3,
    normal_max: P3,
    children: Option<(usize, usize)>,
    indices: Vec<u32>,
}

fn build_fallback_nodes(road: &[StaticTri]) -> (Vec<FallbackNode>, Vec<usize>) {
    fn build(road: &[StaticTri], nodes: &mut Vec<FallbackNode>, mut indices: Vec<u32>) -> usize {
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        let mut normal_min = [f32::INFINITY; 3];
        let mut normal_max = [f32::NEG_INFINITY; 3];
        for &index in &indices {
            let entry = &road[index as usize];
            for axis in 0..3 {
                aabb_min[axis] = aabb_min[axis].min(entry.aabb_min[axis]);
                aabb_max[axis] = aabb_max[axis].max(entry.aabb_max[axis]);
                normal_min[axis] = normal_min[axis].min(entry.normal[axis]);
                normal_max[axis] = normal_max[axis].max(entry.normal[axis]);
            }
        }
        let node_index = nodes.len();
        nodes.push(FallbackNode {
            aabb_min,
            aabb_max,
            normal_min,
            normal_max,
            children: None,
            indices: Vec::new(),
        });
        if indices.len() <= 12 {
            indices.sort_unstable();
            nodes[node_index].indices = indices;
            return node_index;
        }
        let spatial_axis = (0..3)
            .max_by(|&left, &right| {
                (aabb_max[left] - aabb_min[left]).total_cmp(&(aabb_max[right] - aabb_min[right]))
            })
            .expect("three axes");
        indices.sort_unstable_by(|&left, &right| {
            let lhs = road[left as usize].aabb_min[spatial_axis]
                + road[left as usize].aabb_max[spatial_axis];
            let rhs = road[right as usize].aabb_min[spatial_axis]
                + road[right as usize].aabb_max[spatial_axis];
            lhs.total_cmp(&rhs).then_with(|| left.cmp(&right))
        });
        let right_indices = indices.split_off(indices.len() / 2);
        let left = build(road, nodes, indices);
        let right = build(road, nodes, right_indices);
        nodes[node_index].children = Some((left, right));
        node_index
    }

    let mut groups: BTreeMap<(i16, i16, i16), Vec<u32>> = BTreeMap::new();
    for (index, entry) in road.iter().enumerate() {
        let bin = |value: f32| (((value + 1.0) * 16.0).floor() as i16).clamp(0, 31);
        groups
            .entry((
                bin(entry.normal[0]),
                bin(entry.normal[1]),
                bin(entry.normal[2]),
            ))
            .or_default()
            .push(index as u32);
    }
    let mut nodes = Vec::new();
    let roots = groups
        .into_values()
        .map(|indices| build(road, &mut nodes, indices))
        .collect();
    (nodes, roots)
}

fn segment_aabb_intersection(start: P3, end: P3, aabb_min: P3, aabb_max: P3) -> bool {
    let mut t_min = 0.0f64;
    let mut t_max = 1.0f64;
    for axis in 0..3 {
        let origin = f64::from(start[axis]);
        let delta = f64::from(end[axis]) - origin;
        let lo = f64::from(aabb_min[axis]) - 1.0e-6;
        let hi = f64::from(aabb_max[axis]) + 1.0e-6;
        if delta == 0.0 {
            if origin < lo || origin > hi {
                return false;
            }
            continue;
        }
        let first = (lo - origin) / delta;
        let second = (hi - origin) / delta;
        t_min = t_min.max(first.min(second));
        t_max = t_max.min(first.max(second));
        if t_min > t_max {
            return false;
        }
    }
    true
}

impl UniformGrid {
    fn build(tris: &[StaticTri], cell: f32) -> Self {
        let coord = |value: f32| (value / cell).floor() as i32;
        let mut bounds_lo = [i32::MAX; 3];
        let mut bounds_hi = [i32::MIN; 3];
        let mut cells: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        let mut overflow = Vec::new();
        for (index, entry) in tris.iter().enumerate() {
            let lo = [
                coord(entry.aabb_min[0]) - 1,
                coord(entry.aabb_min[1]) - 1,
                coord(entry.aabb_min[2]) - 1,
            ];
            let hi = [
                coord(entry.aabb_max[0]) + 1,
                coord(entry.aabb_max[1]) + 1,
                coord(entry.aabb_max[2]) + 1,
            ];
            for axis in 0..3 {
                bounds_lo[axis] = bounds_lo[axis].min(lo[axis]);
                bounds_hi[axis] = bounds_hi[axis].max(hi[axis]);
            }
            if (0..3).any(|axis| hi[axis] - lo[axis] > GRID_MAX_SPAN) {
                overflow.push((index as u32, entry.aabb_min, entry.aabb_max));
                continue;
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        cells.entry((x, y, z)).or_default().push(index as u32);
                    }
                }
            }
        }
        if tris.is_empty() {
            bounds_lo = [0; 3];
            bounds_hi = [-1; 3];
        }
        Self {
            cell,
            bounds_lo,
            bounds_hi,
            cells,
            overflow,
        }
    }

    /// Ascending, deduplicated soup indices whose cells overlap the AABB
    /// (lookup widened by one cell and clamped to the registered bounds).
    /// Returns true when the (unclamped) lookup range covered every
    /// registered cell.
    fn gather(&self, aabb_min: P3, aabb_max: P3, out: &mut Vec<u32>) -> bool {
        out.clear();
        let coord = |value: f32| (value / self.cell).floor() as i32;
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        let mut empty = false;
        let mut covered = true;
        for axis in 0..3 {
            let raw_lo = coord(aabb_min[axis]) - 1;
            let raw_hi = coord(aabb_max[axis]) + 1;
            if raw_lo > self.bounds_lo[axis] || raw_hi < self.bounds_hi[axis] {
                covered = false;
            }
            lo[axis] = raw_lo.max(self.bounds_lo[axis]);
            hi[axis] = raw_hi.min(self.bounds_hi[axis]);
            if lo[axis] > hi[axis] {
                empty = true;
            }
        }
        // An empty cell map covers everything by definition (all triangles,
        // if any, are overflow and proposed for every lookup).
        if self.cells.is_empty() {
            covered = true;
        }
        if !empty {
            // Probe the coordinate cube, or scan the map and filter when the
            // cube dwarfs the cell map; whichever touches fewer cells.
            let volume = (0..3)
                .map(|axis| i64::from(hi[axis] - lo[axis] + 1))
                .product::<i64>();
            if volume > self.cells.len() as i64 * 4 {
                for (&(x, y, z), cell) in &self.cells {
                    if x >= lo[0]
                        && x <= hi[0]
                        && y >= lo[1]
                        && y <= hi[1]
                        && z >= lo[2]
                        && z <= hi[2]
                    {
                        out.extend_from_slice(cell);
                    }
                }
            } else {
                for x in lo[0]..=hi[0] {
                    for y in lo[1]..=hi[1] {
                        for z in lo[2]..=hi[2] {
                            if let Some(cell) = self.cells.get(&(x, y, z)) {
                                out.extend_from_slice(cell);
                            }
                        }
                    }
                }
            }
        }
        for &(index, tri_min, tri_max) in &self.overflow {
            if (0..3).all(|axis| tri_max[axis] >= aabb_min[axis] && tri_min[axis] <= aabb_max[axis])
            {
                out.push(index);
            }
        }
        out.sort_unstable();
        out.dedup();
        covered
    }

    /// Ascending, deduplicated soup indices from cells crossed by a segment.
    ///
    /// Triangle AABBs are registered one cell wider on every axis, so a
    /// genuine segment/triangle hit is proposed even when the intersection
    /// lies exactly on a cell face or edge. This only narrows candidate
    /// generation; the original float32 predicate and soup-order tie break
    /// remain in `ground_ray_best`.
    fn gather_segment(&self, start: P3, end: P3, out: &mut Vec<u32>) {
        out.clear();
        let coord = |value: f32| (value / self.cell).floor() as i32;
        let mut current = [coord(start[0]), coord(start[1]), coord(start[2])];
        let target = [coord(end[0]), coord(end[1]), coord(end[2])];
        let delta = [
            f64::from(end[0]) - f64::from(start[0]),
            f64::from(end[1]) - f64::from(start[1]),
            f64::from(end[2]) - f64::from(start[2]),
        ];
        let mut step = [0i32; 3];
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for axis in 0..3 {
            if delta[axis] > 0.0 {
                step[axis] = 1;
                let boundary = f64::from((current[axis] + 1) as f32 * self.cell);
                t_max[axis] = (boundary - f64::from(start[axis])) / delta[axis];
                t_delta[axis] = f64::from(self.cell) / delta[axis];
            } else if delta[axis] < 0.0 {
                step[axis] = -1;
                let boundary = f64::from(current[axis] as f32 * self.cell);
                t_max[axis] = (boundary - f64::from(start[axis])) / delta[axis];
                t_delta[axis] = -f64::from(self.cell) / delta[axis];
            }
        }

        let max_steps = (0..3)
            .map(|axis| current[axis].abs_diff(target[axis]) as usize)
            .sum::<usize>()
            + 1;
        for _ in 0..=max_steps {
            if current[0] >= self.bounds_lo[0]
                && current[0] <= self.bounds_hi[0]
                && current[1] >= self.bounds_lo[1]
                && current[1] <= self.bounds_hi[1]
                && current[2] >= self.bounds_lo[2]
                && current[2] <= self.bounds_hi[2]
            {
                if let Some(cell) = self.cells.get(&(current[0], current[1], current[2])) {
                    out.extend_from_slice(cell);
                }
            }
            if current == target {
                break;
            }
            let next_t = t_max[0].min(t_max[1]).min(t_max[2]);
            for axis in 0..3 {
                if t_max[axis] == next_t {
                    current[axis] += step[axis];
                    t_max[axis] += t_delta[axis];
                }
            }
        }
        for &(index, tri_min, tri_max) in &self.overflow {
            if segment_aabb_intersection(start, end, tri_min, tri_max) {
                out.push(index);
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

/// Per-track derived query data, built once and shared process-wide via
/// [`load_track_query_index`].
pub struct TrackQueryIndex {
    /// Road-tagged world triangles in `flatten()` order (the scan order of
    /// the ground/query/sweep/overlap renderers).
    pub road: Vec<StaticTri>,
    /// Triangles of nodes with a non-empty `road.rail` attr, in `flatten()`
    /// order (the `world-query-rail` scan order).
    pub rail: Vec<StaticTri>,
    grid: UniformGrid,
    fallback_nodes: Vec<FallbackNode>,
    fallback_roots: Vec<usize>,
    /// Per-node strip expansion cache (`geom_triangles`), shared with the
    /// animated/moving query path.
    pub node_triangles: Vec<Option<Vec<[u32; 3]>>>,
    /// Dynamic-owner assignment (`assign_dynamic_owners`): the nearest
    /// animated ancestor-or-self per node.
    pub dynamic_owners: Vec<Option<usize>>,
    /// True when any animated-owned node carries road-tagged geometry, i.e.
    /// `world-query-moving` could ever hit. False short-circuits to the miss
    /// row, which is byte-identical to scanning and finding nothing.
    pub has_moving_road_candidates: bool,
    /// Same for `world-query-dynamic` (any animated-owned geometry).
    pub has_dynamic_geom_candidates: bool,
}

fn assign_dynamic_owners(
    track: &kart_fmt::kart_object::Track,
    node_index: usize,
    inherited_owner: Option<usize>,
    owners: &mut [Option<usize>],
) {
    let node = &track.nodes[node_index];
    let owner = if node.prs.as_ref().is_some_and(|prs| prs.has_anim()) {
        Some(node_index)
    } else {
        inherited_owner
    };
    owners[node_index] = owner;
    for &child_index in &node.children {
        assign_dynamic_owners(track, child_index, owner, owners);
    }
}

impl TrackQueryIndex {
    pub fn build(track: &kart_fmt::kart_object::Track) -> Self {
        let node_triangles: Vec<Option<Vec<[u32; 3]>>> = track
            .nodes
            .iter()
            .map(|node| node.geom.as_ref().map(geom_triangles))
            .collect();
        let mut dynamic_owners = vec![None; track.nodes.len()];
        if let Some(root) = track.root {
            assign_dynamic_owners(track, root, None, &mut dynamic_owners);
        }
        let mut has_moving_road_candidates = false;
        let mut has_dynamic_geom_candidates = false;
        for (node_index, owner) in dynamic_owners.iter().enumerate() {
            if owner.is_none() {
                continue;
            }
            let node = &track.nodes[node_index];
            if node.geom.is_none() {
                continue;
            }
            has_dynamic_geom_candidates = true;
            if has_tag(node.extra_tag.as_ref(), "road") {
                has_moving_road_candidates = true;
            }
        }

        let mut road: Vec<StaticTri> = Vec::new();
        let mut rail: Vec<StaticTri> = Vec::new();
        for (node_index, world) in track.flatten() {
            let node = &track.nodes[node_index];
            let is_road = has_tag(node.extra_tag.as_ref(), "road");
            let rail_raw = tag_attr(node.extra_tag.as_ref(), "road", "rail").unwrap_or_default();
            if !is_road && rail_raw.is_empty() {
                continue;
            }
            let Some(geom) = &node.geom else { continue };
            let surface_raw =
                tag_attr(node.extra_tag.as_ref(), "road", "surface").unwrap_or_default();
            let node_name: Arc<str> = Arc::from(clean(&node.name).as_str());
            let surface: Arc<str> = Arc::from(clean(&surface_raw).as_str());
            let rail_text: Arc<str> = Arc::from(clean(&rail_raw).as_str());
            let sid = if surface_raw.is_empty() {
                0
            } else {
                surface_id(&surface_raw)
            };
            let triangles = node_triangles[node_index].as_ref().expect("geom present");
            for (tri_index, indices) in triangles.iter().enumerate() {
                if indices
                    .iter()
                    .any(|&index| index as usize >= geom.vertices.len())
                {
                    continue;
                }
                let tri = [
                    xform(&world, geom.vertices[indices[0] as usize]),
                    xform(&world, geom.vertices[indices[1] as usize]),
                    xform(&world, geom.vertices[indices[2] as usize]),
                ];
                let static_support = is_road
                    && !is_non_drivable_surface(Some(&surface_raw))
                    && tri
                        .iter()
                        .all(|vertex| vertex.iter().all(|coord| coord.is_finite()))
                    && oriented_normal_z(&world, tri) >= 0.35;
                let entry = StaticTri {
                    tri,
                    normal: tri_normal(tri),
                    contact_basis_44: shortest_directed_triangle_edge(tri),
                    aabb_min: [
                        tri[0][0].min(tri[1][0]).min(tri[2][0]),
                        tri[0][1].min(tri[1][1]).min(tri[2][1]),
                        tri[0][2].min(tri[1][2]).min(tri[2][2]),
                    ],
                    aabb_max: [
                        tri[0][0].max(tri[1][0]).max(tri[2][0]),
                        tri[0][1].max(tri[1][1]).max(tri[2][1]),
                        tri[0][2].max(tri[1][2]).max(tri[2][2]),
                    ],
                    node: Arc::clone(&node_name),
                    surface: Arc::clone(&surface),
                    surface_id: sid,
                    rail: Arc::clone(&rail_text),
                    tri_index,
                    static_support,
                };
                if is_road {
                    road.push(entry.clone());
                }
                if !rail_raw.is_empty() {
                    rail.push(entry);
                }
            }
        }

        let grid = UniformGrid::build(&road, GRID_CELL);
        let (fallback_nodes, fallback_roots) = build_fallback_nodes(&road);
        Self {
            road,
            rail,
            grid,
            fallback_nodes,
            fallback_roots,
            node_triangles,
            dynamic_owners,
            has_moving_road_candidates,
            has_dynamic_geom_candidates,
        }
    }

    /// Ascending road-soup candidate indices overlapping the AABB (finite
    /// inputs only; callers guard non-finite values onto the full scan).
    /// Returns true when the lookup covered every registered grid cell.
    fn gather(&self, aabb_min: P3, aabb_max: P3, out: &mut Vec<u32>) -> bool {
        self.grid.gather(aabb_min, aabb_max, out)
    }

    fn gather_segment(&self, start: P3, end: P3, out: &mut Vec<u32>) {
        self.grid.gather_segment(start, end, out);
    }

    /// Exact full-scan support distance used by the historical
    /// `kap-topology` command. This intentionally keeps the old scan and
    /// `min_by` order instead of using the candidate grid.
    pub fn static_support_distance(&self, point: P3) -> Option<f32> {
        self.road
            .iter()
            .filter(|entry| entry.static_support)
            .map(|entry| {
                let closest = closest_point_on_triangle(point, entry.tri);
                squared_distance(point, closest).sqrt()
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

fn is_non_drivable_surface(surface: Option<&str>) -> bool {
    matches!(
        surface,
        Some("리셋" | "reset" | "pit" | "transparent" | "transparency")
    )
}

/// Up-component of the triangle normal corrected for a mirrored world
/// transform, matching kart-autoprobe's former NavMesh construction.
fn oriented_normal_z(world: &[f32; 16], tri: [P3; 3]) -> f32 {
    let normal_z = tri_normal(tri)[2];
    if linear_determinant(world) < 0.0 {
        -normal_z
    } else {
        normal_z
    }
}

fn linear_determinant(m: &[f32; 16]) -> f32 {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
        + m[8] * (m[1] * m[6] - m[5] * m[2])
}

// ---------------------------------------------------------------------------
// Process-level cache, keyed exactly like the static-track parse cache.
// ---------------------------------------------------------------------------

static QUERY_INDEX_CACHE: OnceLock<Mutex<BTreeMap<PathBuf, Arc<TrackQueryIndex>>>> =
    OnceLock::new();

/// Loads (or reuses) the derived query index for `track_path`. The track
/// itself comes from [`load_static_track`], so the index shares that cache's
/// identity and error behavior.
pub fn load_track_query_index(track_path: &Path) -> Result<Arc<TrackQueryIndex>, String> {
    let key = static_track_cache_key(track_path);
    let cache = QUERY_INDEX_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(index) = cache
        .lock()
        .map_err(|_| "query index cache lock poisoned".to_string())?
        .get(&key)
        .cloned()
    {
        return Ok(index);
    }
    let track = load_static_track(track_path)?;
    let index = Arc::new(TrackQueryIndex::build(&track));
    cache
        .lock()
        .map_err(|_| "query index cache lock poisoned".to_string())?
        .insert(key, Arc::clone(&index));
    Ok(index)
}

/// Drops the cached index for `path` (virtual-bytes re-registration hook).
pub fn invalidate_query_index_cache(path: &Path) {
    if let Some(cache) = QUERY_INDEX_CACHE.get() {
        if let Ok(mut cache) = cache.lock() {
            cache.remove(path);
            cache.remove(&static_track_cache_key(path));
        }
    }
}

// ---------------------------------------------------------------------------
// Selection functions: the ONE implementation of triangle selection and
// ordering, shared by the CLI renderers (which format rows) and the typed
// adapter (which quantizes fields).
// ---------------------------------------------------------------------------

fn finite3(value: P3) -> bool {
    value.iter().all(|component| component.is_finite())
}

/// `world-query-ground` selection. `distance_sq` is kept unsquared-rooted so
/// the renderer's `distance_sq.sqrt()` happens at the same place as before.
pub struct GroundSelection<'a> {
    pub entry: &'a StaticTri,
    pub distance_sq: f32,
    pub closest: P3,
}

fn ground_ray_best<'a>(
    index: &'a TrackQueryIndex,
    point: P3,
    ray_end: P3,
    up: P3,
    min_dot: f32,
    candidates: &[u32],
) -> Option<GroundSelection<'a>> {
    let mut best: Option<GroundSelection<'a>> = None;
    for &tri_ref in candidates {
        let entry = &index.road[tri_ref as usize];
        if entry.normal[0] * up[0] + entry.normal[1] * up[1] + entry.normal[2] * up[2] < min_dot {
            continue;
        }
        let Some((_, hit)) = segment_triangle_intersection(point, ray_end, entry.tri) else {
            continue;
        };
        let distance_sq = squared_distance(point, hit);
        if best
            .as_ref()
            .is_none_or(|current| distance_sq < current.distance_sq)
        {
            best = Some(GroundSelection {
                entry,
                distance_sq,
                closest: hit,
            });
        }
    }
    best
}

fn fallback_node_lower_distance(node: &FallbackNode, point: P3) -> f64 {
    (0..3)
        .map(|axis| {
            let value = f64::from(point[axis]);
            let delta = if value < f64::from(node.aabb_min[axis]) {
                f64::from(node.aabb_min[axis]) - value
            } else if value > f64::from(node.aabb_max[axis]) {
                value - f64::from(node.aabb_max[axis])
            } else {
                0.0
            };
            delta * delta
        })
        .sum()
}

fn ground_fallback_indexed(
    index: &TrackQueryIndex,
    point: P3,
    up: P3,
    min_dot: f32,
) -> Option<GroundSelection<'_>> {
    if index.fallback_roots.is_empty() {
        return None;
    }
    let mut best: Option<(u32, GroundSelection<'_>)> = None;
    let mut stack = index.fallback_roots.clone();
    while let Some(node_index) = stack.pop() {
        let node = &index.fallback_nodes[node_index];
        let normal_upper = (0..3)
            .map(|axis| {
                let bound = if up[axis] >= 0.0 {
                    node.normal_max[axis]
                } else {
                    node.normal_min[axis]
                };
                f64::from(bound) * f64::from(up[axis])
            })
            .sum::<f64>();
        if normal_upper + 1.0e-6 < f64::from(min_dot) {
            continue;
        }
        let lower_sq = fallback_node_lower_distance(node, point);
        if let Some((_, current)) = &best {
            // AABB distance is exact in f64 for stored f32 coordinates. Keep
            // a deliberately generous f32-rounding envelope before pruning;
            // near ties remain evaluated and resolve by soup index below.
            let scale = point
                .iter()
                .chain(node.aabb_min.iter())
                .chain(node.aabb_max.iter())
                .map(|value| f64::from(*value).abs())
                .fold(1.0, f64::max);
            let limit = f64::from(current.distance_sq).sqrt() + scale * 1.0e-5 + 1.0e-4;
            if lower_sq > limit * limit {
                continue;
            }
        }
        if let Some((left, right)) = node.children {
            let left_distance = fallback_node_lower_distance(&index.fallback_nodes[left], point);
            let right_distance = fallback_node_lower_distance(&index.fallback_nodes[right], point);
            if left_distance <= right_distance {
                stack.push(right);
                stack.push(left);
            } else {
                stack.push(left);
                stack.push(right);
            }
            continue;
        }
        for &tri_ref in &node.indices {
            let entry = &index.road[tri_ref as usize];
            if entry.normal[0] * up[0] + entry.normal[1] * up[1] + entry.normal[2] * up[2] < min_dot
            {
                continue;
            }
            let closest = closest_point_on_triangle(point, entry.tri);
            let distance_sq = squared_distance(point, closest);
            if best.as_ref().is_none_or(|(prior_index, current)| {
                distance_sq < current.distance_sq
                    || (distance_sq == current.distance_sq && tri_ref < *prior_index)
            }) {
                best = Some((
                    tri_ref,
                    GroundSelection {
                        entry,
                        distance_sq,
                        closest,
                    },
                ));
            }
        }
    }
    best.map(|(_, selection)| selection)
}

/// The verbatim full scan of the original renderer: interleaved ray-hit /
/// nearest-point selection with the `(is_ray, distance_sq)` ordering.
fn query_ground_full(
    index: &TrackQueryIndex,
    point: P3,
    up: P3,
    min_dot: f32,
) -> Option<GroundSelection<'_>> {
    let ray_end = [
        point[0] - up[0] * 1000.0,
        point[1] - up[1] * 1000.0,
        point[2] - up[2] * 1000.0,
    ];
    let mut best: Option<(bool, GroundSelection<'_>)> = None;
    for entry in &index.road {
        if entry.normal[0] * up[0] + entry.normal[1] * up[1] + entry.normal[2] * up[2] < min_dot {
            continue;
        }
        let ray_hit = segment_triangle_intersection(point, ray_end, entry.tri).map(|(_, hit)| hit);
        let closest = ray_hit.unwrap_or_else(|| closest_point_on_triangle(point, entry.tri));
        let distance_sq = squared_distance(point, closest);
        let is_ray = ray_hit.is_some();
        if best.as_ref().is_none_or(|current| {
            (is_ray && !current.0) || (is_ray == current.0 && distance_sq < current.1.distance_sq)
        }) {
            best = Some((
                is_ray,
                GroundSelection {
                    entry,
                    distance_sq,
                    closest,
                },
            ));
        }
    }
    best.map(|(_, selection)| selection)
}

pub fn query_ground(
    index: &TrackQueryIndex,
    point: P3,
    up: P3,
    min_dot: f32,
) -> Option<GroundSelection<'_>> {
    if !finite3(point) || !finite3(up) || !min_dot.is_finite() {
        return query_ground_full(index, point, up, min_dot);
    }
    let ray_end = [
        point[0] - up[0] * 1000.0,
        point[1] - up[1] * 1000.0,
        point[2] - up[2] * 1000.0,
    ];
    // Pass 1: nearest ray hit. A segment/triangle intersection implies AABB
    // overlap, so every genuine ray hit is among these candidates; with one
    // found, the renderer's (is-ray-first, then distance) ordering resolves
    // to exactly this argmin.
    let mut candidates = Vec::new();
    index.gather_segment(point, ray_end, &mut candidates);
    if let Some(selection) = ground_ray_best(index, point, ray_end, up, min_dot, &candidates) {
        return Some(selection);
    }
    // Pass 2: no ray hit exists along the segment, so the renderer's
    // nearest-point fallback argmin applies. The normal/spatial tree is only
    // an acceleration structure: leaf predicates and soup-order ties remain
    // identical to the original full scan.
    ground_fallback_indexed(index, point, up, min_dot)
}

/// `world-query` selection (plain nearest road triangle; CLI-only cold path,
/// verbatim full scan).
pub struct NearestSelection<'a> {
    pub entry: &'a StaticTri,
    pub distance_sq: f32,
    pub closest: P3,
}

pub fn query_nearest(index: &TrackQueryIndex, point: P3) -> Option<NearestSelection<'_>> {
    let mut best: Option<NearestSelection<'_>> = None;
    for entry in &index.road {
        let closest = closest_point_on_triangle(point, entry.tri);
        let distance_sq = squared_distance(point, closest);
        if best
            .as_ref()
            .is_none_or(|current| distance_sq < current.distance_sq)
        {
            best = Some(NearestSelection {
                entry,
                distance_sq,
                closest,
            });
        }
    }
    best
}

/// `world-query-rail` selection (verbatim full scan over the rail soup).
pub struct RailSelection<'a> {
    pub entry: &'a StaticTri,
    pub distance_sq: f32,
    pub closest: P3,
}

pub fn query_rail(index: &TrackQueryIndex, point: P3) -> Option<RailSelection<'_>> {
    let mut best: Option<RailSelection<'_>> = None;
    for entry in &index.rail {
        let closest = closest_point_on_triangle(point, entry.tri);
        let distance_sq = squared_distance(point, closest);
        if best
            .as_ref()
            .is_none_or(|current| distance_sq < current.distance_sq)
        {
            best = Some(RailSelection {
                entry,
                distance_sq,
                closest,
            });
        }
    }
    best
}

/// `world-sweep` selection: earliest segment/triangle intersection.
pub struct SweepSelection<'a> {
    pub entry: &'a StaticTri,
    pub fraction: f32,
    pub point: P3,
}

fn segment_aabb(start: P3, end: P3) -> (P3, P3) {
    (
        [
            start[0].min(end[0]),
            start[1].min(end[1]),
            start[2].min(end[2]),
        ],
        [
            start[0].max(end[0]),
            start[1].max(end[1]),
            start[2].max(end[2]),
        ],
    )
}

pub fn query_sweep(index: &TrackQueryIndex, start: P3, end: P3) -> Option<SweepSelection<'_>> {
    let mut candidates = Vec::new();
    if finite3(start) && finite3(end) {
        let (lo, hi) = segment_aabb(start, end);
        index.gather(lo, hi, &mut candidates);
    } else {
        candidates.extend(0..index.road.len() as u32);
    }
    let mut best: Option<SweepSelection<'_>> = None;
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        let Some((fraction, point)) = segment_triangle_intersection(start, end, entry.tri) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| fraction < current.fraction)
        {
            best = Some(SweepSelection {
                entry,
                fraction,
                point,
            });
        }
    }
    best
}

/// `world-sweep-sphere` selection: earliest swept-sphere contact.
pub struct SphereSelection<'a> {
    pub entry: &'a StaticTri,
    pub hit: SphereTriangleHit,
}

pub fn query_sweep_sphere<'a>(
    index: &'a TrackQueryIndex,
    radius: f32,
    start: P3,
    end: P3,
) -> Option<SphereSelection<'a>> {
    let mut candidates = Vec::new();
    if finite3(start) && finite3(end) && radius.is_finite() {
        let (mut lo, mut hi) = segment_aabb(start, end);
        for axis in 0..3 {
            lo[axis] -= radius;
            hi[axis] += radius;
        }
        index.gather(lo, hi, &mut candidates);
    } else {
        candidates.extend(0..index.road.len() as u32);
    }
    let mut best: Option<SphereSelection<'a>> = None;
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        let Some(hit) = swept_sphere_triangle_intersection(start, end, radius, entry.tri) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| hit.fraction < current.hit.fraction)
        {
            best = Some(SphereSelection { entry, hit });
        }
    }
    best
}

/// `world-sweep-obb*` / `world-overlap-obb-all` selection payload.
pub struct ObbSelection<'a> {
    pub entry: &'a StaticTri,
    pub hit: ObbTriangleHit,
}

fn obb_pose_aabb(center: P3, axes: [P3; 3], half_extents: P3) -> (P3, P3) {
    let mut extent = [0.0f32; 3];
    for axis in 0..3 {
        extent[axis] = axes[0][axis].abs() * half_extents[0]
            + axes[1][axis].abs() * half_extents[1]
            + axes[2][axis].abs() * half_extents[2];
    }
    (
        [
            center[0] - extent[0],
            center[1] - extent[1],
            center[2] - extent[2],
        ],
        [
            center[0] + extent[0],
            center[1] + extent[1],
            center[2] + extent[2],
        ],
    )
}

fn gather_swept_obb(
    index: &TrackQueryIndex,
    start: P3,
    end: P3,
    axes: [P3; 3],
    half_extents: P3,
    out: &mut Vec<u32>,
) {
    if finite3(start)
        && finite3(end)
        && axes.iter().all(|axis| finite3(*axis))
        && finite3(half_extents)
    {
        let (lo0, hi0) = obb_pose_aabb(start, axes, half_extents);
        let (lo1, hi1) = obb_pose_aabb(end, axes, half_extents);
        index.gather(
            [lo0[0].min(lo1[0]), lo0[1].min(lo1[1]), lo0[2].min(lo1[2])],
            [hi0[0].max(hi1[0]), hi0[1].max(hi1[1]), hi0[2].max(hi1[2])],
            out,
        );
    } else {
        out.extend(0..index.road.len() as u32);
    }
}

/// `world-sweep-obb`: earliest fixed-orientation OBB/triangle contact.
pub fn query_sweep_obb<'a>(
    index: &'a TrackQueryIndex,
    start: P3,
    end: P3,
    axes: [P3; 3],
    half_extents: P3,
) -> Option<ObbSelection<'a>> {
    let mut candidates = Vec::new();
    gather_swept_obb(index, start, end, axes, half_extents, &mut candidates);
    let mut best: Option<ObbSelection<'a>> = None;
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        let Some(hit) = swept_obb_triangle_intersection(start, end, axes, half_extents, entry.tri)
        else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| hit.fraction < current.hit.fraction)
        {
            best = Some(ObbSelection { entry, hit });
        }
    }
    best
}

/// `world-sweep-obb-rotating`: earliest rotating-OBB contact.
pub fn query_sweep_obb_rotating<'a>(
    index: &'a TrackQueryIndex,
    start: P3,
    end: P3,
    start_axes: [P3; 3],
    end_axes: [P3; 3],
    half_extents: P3,
) -> Option<ObbSelection<'a>> {
    let mut candidates = Vec::new();
    gather_swept_rotating_obb(
        index,
        start,
        end,
        start_axes,
        end_axes,
        half_extents,
        &mut candidates,
    );
    let mut best: Option<ObbSelection<'a>> = None;
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        let Some(hit) = swept_rotating_obb_triangle_intersection(
            start,
            end,
            start_axes,
            end_axes,
            half_extents,
            entry.tri,
        ) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| hit.fraction < current.hit.fraction)
        {
            best = Some(ObbSelection { entry, hit });
        }
    }
    best
}

fn gather_swept_rotating_obb(
    index: &TrackQueryIndex,
    start: P3,
    end: P3,
    start_axes: [P3; 3],
    end_axes: [P3; 3],
    half_extents: P3,
    out: &mut Vec<u32>,
) {
    let finite = finite3(start)
        && finite3(end)
        && start_axes.iter().all(|axis| finite3(*axis))
        && end_axes.iter().all(|axis| finite3(*axis))
        && finite3(half_extents);
    if !finite {
        out.extend(0..index.road.len() as u32);
        return;
    }
    // Every point of the rotating OBB stays within the center path's segment
    // AABB expanded by the box radius (the same norm the CCD itself uses for
    // its motion bound; the +1 cell lookup margin covers its rounding).
    let box_radius = half_extents
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let (mut lo, mut hi) = segment_aabb(start, end);
    for axis in 0..3 {
        lo[axis] -= box_radius;
        hi[axis] += box_radius;
    }
    index.gather(lo, hi, out);
}

/// `world-sweep-obb-rotating-all`: every rotating-OBB contact in scan order.
pub fn query_sweep_obb_rotating_all<'a>(
    index: &'a TrackQueryIndex,
    start: P3,
    end: P3,
    start_axes: [P3; 3],
    end_axes: [P3; 3],
    half_extents: P3,
) -> Vec<ObbSelection<'a>> {
    let mut candidates = Vec::new();
    gather_swept_rotating_obb(
        index,
        start,
        end,
        start_axes,
        end_axes,
        half_extents,
        &mut candidates,
    );
    let mut hits = Vec::new();
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        if let Some(hit) = swept_rotating_obb_triangle_intersection(
            start,
            end,
            start_axes,
            end_axes,
            half_extents,
            entry.tri,
        ) {
            hits.push(ObbSelection { entry, hit });
        }
    }
    hits
}

/// `world-overlap-obb-all`: every instantaneous OBB/triangle overlap in scan
/// order (the renderer's zero-direction `swept_obb_triangle_intersection`
/// call, verbatim).
pub fn query_overlap_obb_all<'a>(
    index: &'a TrackQueryIndex,
    center: P3,
    axes: [P3; 3],
    half_extents: P3,
) -> Vec<ObbSelection<'a>> {
    let mut candidates = Vec::new();
    gather_swept_obb(index, center, center, axes, half_extents, &mut candidates);
    let mut hits = Vec::new();
    for &tri_ref in &candidates {
        let entry = &index.road[tri_ref as usize];
        if let Some(hit) =
            swept_obb_triangle_intersection(center, center, axes, half_extents, entry.tri)
        {
            hits.push(ObbSelection { entry, hit });
        }
    }
    hits
}
