//! TSV facade: byte-exact renderers for the kart-autoprobe world-query CLI
//! contract plus thin `run_*` shells that print the rendered bytes.
//!
//! Every `render_*` function produces exactly the stdout bytes the historical
//! `run_*` body emitted (including float formatting and row ordering); the
//! regression gate is `analysis/track_world/tsv_baseline_20260729`.

use std::path::PathBuf;

use kart_physics::world::World;

use crate::geom::*;
use crate::query::load_track_query_index;
use crate::topology::kap_gap_windows;
use crate::track::*;
use crate::world::{EventPhysicsDisposition, TrackWorld};

/// Appends one `println!`-equivalent line to the render buffer.
macro_rules! emitln {
    ($out:ident, $($arg:tt)*) => {{
        $out.push_str(&format!($($arg)*));
        $out.push('\n');
    }};
}

/// A render failure carrying the partial stdout bytes the original
/// `println!`-based body had already emitted before failing (e.g. the
/// `world-node-triangle` header precedes the node lookup).
pub struct RenderError {
    pub partial: String,
    pub message: String,
}

impl RenderError {
    fn new(partial: String, message: String) -> Self {
        Self { partial, message }
    }
}

impl From<String> for RenderError {
    fn from(message: String) -> Self {
        Self {
            partial: String::new(),
            message,
        }
    }
}

/// Best-candidate accumulator shape for the animated-geometry
/// nearest-triangle scan, factored out to satisfy `clippy::type_complexity`
/// (layout unchanged).
type AnimatedBest = Option<(
    f32,
    usize,
    String,
    String,
    String,
    String,
    usize,
    P3,
    P3,
    [P3; 3],
)>;

/// Iterates `--flag value` style CLI args, tracking position so callers can
/// consume the value that follows a recognized flag without hand-rolling the
/// index bookkeeping in every subcommand parser.
pub struct FlagCursor<'a> {
    pub args: &'a [String],
    pub i: usize,
}

impl<'a> FlagCursor<'a> {
    pub fn new(args: &'a [String], start: usize) -> Self {
        Self { args, i: start }
    }

    /// The current flag token, if any remain.
    pub fn flag(&self) -> Option<&'a str> {
        self.args.get(self.i).map(String::as_str)
    }

    /// Consumes and returns the value following the current flag.
    pub fn take_value(&mut self, missing: &str) -> Result<&'a str, String> {
        self.i += 1;
        self.args
            .get(self.i)
            .map(String::as_str)
            .ok_or_else(|| missing.to_string())
    }

    /// Advances past the current flag (and its value, if `take_value` was called).
    pub fn advance(&mut self) {
        self.i += 1;
    }
}

pub fn render_kap_markers(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 1 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <path.kap>".to_string(),
        ));
    }
    let path = PathBuf::from(&args[0]);
    let bytes = read_bytes(&path)?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|err| format!("decode {}: {err}", path.display()))?;
    let kap = kart_fmt::kap::parse(&clear)
        .ok_or_else(|| format!("parse {}: invalid KAP payload", path.display()))?;
    if !kap.markers_in_range() {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            format!("{} has out-of-range KAP markers", path.display()),
        ));
    }
    emitln!(
        out,
        "track_id\tslot\tmarker_ordinal\tpoint_index\tpoint_count\
\tpx\tpy\tpz\ttx\tty\ttz"
    );
    for (marker_ordinal, marker) in kap.markers.iter().copied().enumerate() {
        let point_index = marker as usize;
        let point = kap.points[point_index];
        let previous_index = point_index.saturating_sub(1);
        let next_index = (point_index + 1).min(kap.points.len() - 1);
        let mut tangent = [
            kap.points[next_index][0] - kap.points[previous_index][0],
            kap.points[next_index][1] - kap.points[previous_index][1],
            kap.points[next_index][2] - kap.points[previous_index][2],
        ];
        let length = tangent
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if length > 1.0e-9 {
            for value in &mut tangent {
                *value /= length;
            }
        }
        emitln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(kap_track_id(&path).as_deref().unwrap_or("")),
            kap_slot(&path).unwrap_or(0),
            marker_ordinal,
            point_index,
            kap.points.len(),
            point[0],
            point[1],
            point[2],
            tangent[0],
            tangent[1],
            tangent[2],
        );
    }
    Ok(out)
}

pub fn render_kap_points(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.is_empty() {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <path.kap> [--step-m <n>] [--start-s <n>] [--end-s <n>]".to_string(),
        ));
    }
    let path = PathBuf::from(&args[0]);
    let mut step_m = 25.0_f32;
    let mut start_s = 0.0_f32;
    let mut end_s = f32::INFINITY;
    let mut cursor = FlagCursor::new(args, 1);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--step-m" => {
                step_m = cursor
                    .take_value("--step-m requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--step-m must be a number".to_string())?;
            }
            "--start-s" => {
                start_s = cursor
                    .take_value("--start-s requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--start-s must be a number".to_string())?;
            }
            "--end-s" => {
                end_s = cursor
                    .take_value("--end-s requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--end-s must be a number".to_string())?;
            }
            other => {
                return Err(RenderError::new(
                    std::mem::take(&mut out),
                    format!("unknown kap-points option: {other}"),
                ))
            }
        }
        cursor.advance();
    }
    if !step_m.is_finite() || step_m <= 0.0 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "--step-m must be finite and positive".to_string(),
        ));
    }
    if !start_s.is_finite() || start_s < 0.0 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "--start-s must be finite and non-negative".to_string(),
        ));
    }
    if end_s.is_nan() || end_s < start_s {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "--end-s must be greater than or equal to --start-s".to_string(),
        ));
    }

    let bytes = read_bytes(&path)?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|err| format!("decode {}: {err}", path.display()))?;
    let kap = kart_fmt::kap::parse(&clear)
        .ok_or_else(|| format!("parse {}: invalid KAP payload", path.display()))?;
    if kap.points.is_empty() {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            format!("{} has no KAP points", path.display()),
        ));
    }

    emitln!(out, "track_id\tslot\tpoint_index\tpath_s\tx\ty\tz");
    let mut path_s = 0.0_f32;
    let mut next_s = start_s;
    let mut emitted_last = false;
    for (index, point) in kap.points.iter().copied().enumerate() {
        if index > 0 {
            let previous = kap.points[index - 1];
            path_s += ((point[0] - previous[0]).powi(2)
                + (point[1] - previous[1]).powi(2)
                + (point[2] - previous[2]).powi(2))
            .sqrt();
        }
        if path_s + 1.0e-4 >= next_s && path_s <= end_s {
            emitln!(
                out,
                "{}\t{}\t{}\t{:.6}\t{:.9}\t{:.9}\t{:.9}",
                clean(kap_track_id(&path).as_deref().unwrap_or("")),
                kap_slot(&path).unwrap_or(0),
                index,
                path_s,
                point[0],
                point[1],
                point[2]
            );
            next_s = path_s + step_m;
            emitted_last = index + 1 == kap.points.len();
        }
        if path_s > end_s {
            break;
        }
    }
    if end_s.is_infinite() && !emitted_last && kap.points.len() > 1 {
        let point = kap.points[kap.points.len() - 1];
        emitln!(
            out,
            "{}\t{}\t{}\t{:.6}\t{:.9}\t{:.9}\t{:.9}",
            clean(kap_track_id(&path).as_deref().unwrap_or("")),
            kap_slot(&path).unwrap_or(0),
            kap.points.len() - 1,
            path_s,
            point[0],
            point[1],
            point[2]
        );
    }
    Ok(out)
}

pub fn render_kap_topology(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 2 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <path.kap>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let kap_path = PathBuf::from(&args[1]);
    let bytes = read_bytes(&kap_path)?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|err| format!("decode {}: {err}", kap_path.display()))?;
    let kap = kart_fmt::kap::parse(&clear)
        .ok_or_else(|| format!("parse {}: invalid KAP payload", kap_path.display()))?;
    if kap.points.len() < 2 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            format!("{} has fewer than two KAP points", kap_path.display()),
        ));
    }

    let index = load_track_query_index(&track_path)?;
    const SUPPORT_DISTANCE_M: f32 = 2.5;
    let gaps = kap_gap_windows(&kap.points, |point| {
        index
            .static_support_distance(point)
            .map(|distance| distance <= SUPPORT_DISTANCE_M)
            .unwrap_or(false)
    });
    let output_track_id = kap_track_id(&kap_path).unwrap_or_else(|| track_id(&track_path));

    emitln!(
        out,
        "track_id\tslot\tkind\tapproach_start_s\tentry_s\texit_s\trequired_speed"
    );
    for gap in gaps {
        emitln!(
            out,
            "{}\t{}\tstatic_support_gap\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            clean(&output_track_id),
            kap_slot(&kap_path).unwrap_or(0),
            gap.approach_start_s,
            gap.entry_s,
            gap.exit_s,
            gap.required_speed
        );
    }
    Ok(out)
}

pub fn render_world_node_info(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 2 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <node-name-substring>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let wanted = &args[1];
    let track = load_static_track(&track_path)?;
    emitln!(
        out,
        "node\ttexture\thas_geom\talpha_blend\talpha_test\ttags"
    );
    for node in track.nodes.iter().take(track.scene_node_count) {
        if !node.name.contains(wanted) {
            continue;
        }
        let mut tags = Vec::new();
        if let Some(tag) = &node.extra_tag {
            flatten_tag(tag, "", &mut tags);
        }
        emitln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}",
            clean(&node.name),
            clean(node.tex.as_deref().unwrap_or("")),
            node.geom.is_some(),
            node.alpha.map(|value| value.use_blend).unwrap_or(false),
            node.alpha
                .map(|value| value.use_alpha_test)
                .unwrap_or(false),
            clean(&tags.join(";"))
        );
    }
    Ok(out)
}

pub fn render_world_node_triangle(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 3 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <exact-node-name> <triangle-index>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let wanted = &args[1];
    let triangle_index = args[2]
        .parse::<usize>()
        .map_err(|_| "triangle-index must be an unsigned integer".to_string())?;
    let track = load_static_track(&track_path)?;
    emitln!(
        out,
        concat!(
            "track_id\tnode\ttriangle_index",
            "\tv0x\tv0y\tv0z\tv1x\tv1y\tv1z\tv2x\tv2y\tv2z"
        )
    );
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        if node.name != *wanted {
            continue;
        }
        let geom = node
            .geom
            .as_ref()
            .ok_or_else(|| format!("{wanted} has no geometry"))
            .map_err(|err| RenderError::new(std::mem::take(&mut out), err))?;
        let triangles = geom_triangles(geom);
        let indices = triangles
            .get(triangle_index)
            .ok_or_else(|| {
                format!(
                    "{wanted} has {} triangles, requested {triangle_index}",
                    triangles.len()
                )
            })
            .map_err(|err| RenderError::new(std::mem::take(&mut out), err))?;
        if indices
            .iter()
            .any(|&index| index as usize >= geom.vertices.len())
        {
            return Err(RenderError::new(
                std::mem::take(&mut out),
                format!("{wanted} triangle {triangle_index} is invalid"),
            ));
        }
        let vertices = indices.map(|index| xform(&world, geom.vertices[index as usize]));
        emitln!(
            out,
            "{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)),
            clean(&node.name),
            triangle_index,
            vertices[0][0],
            vertices[0][1],
            vertices[0][2],
            vertices[1][0],
            vertices[1][1],
            vertices[1][2],
            vertices[2][0],
            vertices[2][1],
            vertices[2][2],
        );
        return Ok(out);
    }
    Err(RenderError::new(out, format!("node not found: {wanted}")))
}

pub fn render_world_dynamic_objects(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 1 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let track = load_static_track(&track_path)?;
    emitln!(
        out,
        "{}",
        concat!(
            "track_id\tnode_index\tnode\tclass_stamp\thas_geom",
            "\tdescendant_geom_count\tdescendant_road_count",
            "\tposition_key_count\trotation_key_count\tscale_key_count",
            "\tfirst_key_time\tlast_key_time\tbase_x\tbase_y\tbase_z",
            "\tcontroller_base_0\tcontroller_base_1\tcontroller_base_2",
            "\tcontroller_base_3\tcontroller_base_4\tcontroller_base_5",
            "\tcontroller_start_time\tcontroller_end_time",
            "\tcontroller_tail_0\tcontroller_tail_1\tcontroller_tail_2",
            "\tcontroller_tail_3\tcontroller_tail_4\tcontroller_tail_5\ttags"
        )
    );
    for (node_index, node) in track.nodes.iter().take(track.scene_node_count).enumerate() {
        let Some(prs) = node.prs.as_ref().filter(|prs| prs.has_anim()) else {
            continue;
        };
        let mut times = prs
            .pos
            .iter()
            .map(|(time, _)| *time)
            .chain(prs.rot.iter().map(|(time, _)| *time))
            .chain(prs.scale.iter().map(|(time, _)| *time))
            .collect::<Vec<_>>();
        times.sort_unstable();
        let mut tags = Vec::new();
        if let Some(tag) = &node.extra_tag {
            flatten_tag(tag, "", &mut tags);
        }
        let mut descendant_geom_count = 0usize;
        let mut descendant_road_count = 0usize;
        let mut pending = node.children.clone();
        while let Some(child_index) = pending.pop() {
            let child = &track.nodes[child_index];
            descendant_geom_count += usize::from(child.geom.is_some());
            descendant_road_count += usize::from(has_tag(child.extra_tag.as_ref(), "road"));
            pending.extend(child.children.iter().copied());
        }
        emitln!(
            out,
            "{}\t{}\t{}\t0x{:08X}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            clean(&track_id(&track_path)),
            node_index,
            clean(&node.name),
            node.class_stamp,
            node.geom.is_some(),
            descendant_geom_count,
            descendant_road_count,
            prs.pos.len(),
            prs.rot.len(),
            prs.scale.len(),
            times.first().copied().unwrap_or(0),
            times.last().copied().unwrap_or(0),
            node.position[0],
            node.position[1],
            node.position[2],
            prs.controller_base.map(|value| value[0]).unwrap_or(0),
            prs.controller_base.map(|value| value[1]).unwrap_or(0),
            prs.controller_base.map(|value| value[2]).unwrap_or(0),
            prs.controller_base.map(|value| value[3]).unwrap_or(0),
            prs.controller_base.map(|value| value[4]).unwrap_or(0),
            prs.controller_base.map(|value| value[5]).unwrap_or(0),
            prs.controller_base.map(|value| value[6]).unwrap_or(0),
            prs.controller_base.map(|value| value[7]).unwrap_or(0),
            prs.controller_tail.map(|value| value[0]).unwrap_or(0),
            prs.controller_tail.map(|value| value[1]).unwrap_or(0),
            prs.controller_tail.map(|value| value[2]).unwrap_or(0),
            prs.controller_tail.map(|value| value[3]).unwrap_or(0),
            prs.controller_tail.map(|value| value[4]).unwrap_or(0),
            prs.controller_tail.map(|value| value[5]).unwrap_or(0),
            clean(&tags.join(";")),
        );
    }
    Ok(out)
}

pub fn render_world_dynamic_state(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 4 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <node-index> <time-ms> <delta-ms>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let node_index = args[1]
        .parse::<usize>()
        .map_err(|_| "node-index must be an unsigned integer".to_string())?;
    let time_ms = args[2]
        .parse::<i32>()
        .map_err(|_| "time-ms must be an integer".to_string())?;
    let delta_ms = args[3]
        .parse::<i32>()
        .map_err(|_| "delta-ms must be an integer".to_string())?;
    if delta_ms <= 0 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "delta-ms must be positive".to_string(),
        ));
    }
    let track = load_static_track(&track_path)?;
    let node = track
        .nodes
        .get(node_index)
        .ok_or_else(|| format!("node-index {node_index} is out of range"))?;
    let start = track
        .flatten_at(time_ms)
        .into_iter()
        .find(|(index, _)| *index == node_index)
        .map(|(_, matrix)| matrix)
        .ok_or_else(|| format!("node-index {node_index} is unreachable from the track root"))?;
    let end = track
        .flatten_at(time_ms.saturating_add(delta_ms))
        .into_iter()
        .find(|(index, _)| *index == node_index)
        .map(|(_, matrix)| matrix)
        .ok_or_else(|| format!("node-index {node_index} is unreachable from the track root"))?;

    let position = [start[12], start[13], start[14]];
    let end_position = [end[12], end[13], end[14]];
    let dt = delta_ms as f32 / 1000.0;
    let velocity = [
        (end_position[0] - position[0]) / dt,
        (end_position[1] - position[1]) / dt,
        (end_position[2] - position[2]) / dt,
    ];
    let axes = normalized_matrix_axes(start)?;
    let end_axes = normalized_matrix_axes(end)?;
    let q0 = axes_to_quat(axes);
    let q1 = axes_to_quat(end_axes);
    let angular_velocity = angular_velocity_between(q0, q1, dt);
    let matrix = [
        axes[0][0], axes[1][0], axes[2][0], axes[0][1], axes[1][1], axes[2][1], axes[0][2],
        axes[1][2], axes[2][2],
    ];

    emitln!(
        out,
        "{}",
        concat!(
            "track_id\tnode_index\tnode\ttime_ms\tdelta_ms",
            "\tpx\tpy\tpz",
            "\tm00\tm01\tm02\tm10\tm11\tm12\tm20\tm21\tm22",
            "\tvx\tvy\tvz\twx\twy\twz"
        )
    );
    emitln!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
        clean(&track_id(&track_path)),
        node_index,
        clean(&node.name),
        time_ms,
        delta_ms,
        position[0],
        position[1],
        position[2],
        matrix[0],
        matrix[1],
        matrix[2],
        matrix[3],
        matrix[4],
        matrix[5],
        matrix[6],
        matrix[7],
        matrix[8],
        velocity[0],
        velocity[1],
        velocity[2],
        angular_velocity[0],
        angular_velocity[1],
        angular_velocity[2],
    );
    Ok(out)
}

pub fn render_world_query_moving(args: &[String]) -> Result<String, RenderError> {
    render_world_query_animated_geometry(args, true)
}

pub fn render_world_query_dynamic(args: &[String]) -> Result<String, RenderError> {
    render_world_query_animated_geometry(args, false)
}

fn render_world_query_animated_geometry(
    args: &[String],
    road_only: bool,
) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 5 && args.len() != 6 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <time-ms> <x> <y> <z> [owner-index]".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let time_ms = args[1]
        .parse::<i32>()
        .map_err(|_| "time-ms must be an integer".to_string())?;
    let parse = |index: usize, name: &str| {
        args[index]
            .parse::<f32>()
            .map_err(|_| format!("{name} must be numeric"))
    };
    let point = [parse(2, "x")?, parse(3, "y")?, parse(4, "z")?];
    let requested_owner = args
        .get(5)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "owner-index must be an unsigned integer".to_string())
        })
        .transpose()?;
    let track = load_static_track(&track_path)?;
    let index = crate::query::load_track_query_index(&track_path)?;
    let candidates_exist = if road_only {
        index.has_moving_road_candidates
    } else {
        index.has_dynamic_geom_candidates
    };
    let mut best: AnimatedBest = None;
    // With no animated-owned geometry of the queried kind the scan provably
    // finds nothing; skip the time-sampled flatten and emit the miss row
    // (byte-identical to scanning and missing).
    if candidates_exist {
        for (node_index, world) in track.flatten_at(time_ms) {
            let Some(owner_index) = index.dynamic_owners[node_index] else {
                continue;
            };
            if requested_owner.is_some_and(|requested| requested != owner_index) {
                continue;
            }
            let node = &track.nodes[node_index];
            if road_only && !has_tag(node.extra_tag.as_ref(), "road") {
                continue;
            }
            let Some(geom) = &node.geom else { continue };
            let surface = tag_attr(node.extra_tag.as_ref(), "road", "surface").unwrap_or_default();
            let rail = tag_attr(node.extra_tag.as_ref(), "road", "rail").unwrap_or_default();
            let triangles = index.node_triangles[node_index]
                .as_ref()
                .expect("geom present");
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
                let closest = closest_point_on_triangle(point, tri);
                let distance_sq = squared_distance(point, closest);
                if best.as_ref().is_none_or(|current| distance_sq < current.0) {
                    best = Some((
                        distance_sq,
                        owner_index,
                        track.nodes[owner_index].name.clone(),
                        node.name.clone(),
                        surface.clone(),
                        rail.clone(),
                        tri_index,
                        closest,
                        tri_normal(tri),
                        tri,
                    ));
                }
            }
        }
    }
    let header = concat!(
        "track_id\thit\ttime_ms\towner_index\towner_node\tnode\ttri_index",
        "\troad_surface\tsurface_id\troad_rail\tdistance",
        "\tqx\tqy\tqz\tnx\tny\tnz",
        "\tv0x\tv0y\tv0z\tv1x\tv1y\tv1z\tv2x\tv2y\tv2z"
    );
    emitln!(out, "{header}");
    let Some((
        distance_sq,
        owner_index,
        owner_node,
        node,
        surface,
        rail,
        tri_index,
        closest,
        normal,
        tri,
    )) = best
    else {
        emitln!(out, "{}\t0\t{}", clean(&track_id(&track_path)), time_ms);
        return Ok(out);
    };
    emitln!(
        out,
        "{}\t1\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
        clean(&track_id(&track_path)),
        time_ms,
        owner_index,
        clean(&owner_node),
        clean(&node),
        tri_index,
        clean(&surface),
        if surface.is_empty() { 0 } else { surface_id(&surface) },
        clean(&rail),
        distance_sq.sqrt(),
        closest[0],
        closest[1],
        closest[2],
        normal[0],
        normal[1],
        normal[2],
        tri[0][0],
        tri[0][1],
        tri[0][2],
        tri[1][0],
        tri[1][1],
        tri[1][2],
        tri[2][0],
        tri[2][1],
        tri[2][2],
    );
    Ok(out)
}

pub fn render_world_query(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 4 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <x> <y> <z>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let point = [
        args[1]
            .parse::<f32>()
            .map_err(|_| "x must be numeric".to_string())?,
        args[2]
            .parse::<f32>()
            .map_err(|_| "y must be numeric".to_string())?,
        args[3]
            .parse::<f32>()
            .map_err(|_| "z must be numeric".to_string())?,
    ];
    let index = crate::query::load_track_query_index(&track_path)?;
    let Some(best) = crate::query::query_nearest(&index, point) else {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "track has no queryable triangles".to_string(),
        ));
    };
    emitln!(out, "track_id\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tdistance\tqx\tqy\tqz\tnx\tny\tnz");
    emitln!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
        clean(&track_id(&track_path)),
        best.entry.node,
        best.entry.tri_index,
        best.entry.surface,
        best.entry.surface_id,
        best.entry.rail,
        best.distance_sq.sqrt(),
        best.closest[0],
        best.closest[1],
        best.closest[2],
        best.entry.normal[0],
        best.entry.normal[1],
        best.entry.normal[2]
    );
    Ok(out)
}

pub fn render_world_query_rail(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 4 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <x> <y> <z>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let point = [
        args[1]
            .parse::<f32>()
            .map_err(|_| "x must be numeric".to_string())?,
        args[2]
            .parse::<f32>()
            .map_err(|_| "y must be numeric".to_string())?,
        args[3]
            .parse::<f32>()
            .map_err(|_| "z must be numeric".to_string())?,
    ];
    let index = crate::query::load_track_query_index(&track_path)?;
    let Some(best) = crate::query::query_rail(&index, point) else {
        emitln!(out, "track_id\thit");
        emitln!(out, "{}\t0", clean(&track_id(&track_path)));
        return Ok(out);
    };
    let tri = best.entry.tri;
    emitln!(out,
        "track_id\thit\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tdistance\tqx\tqy\tqz\tnx\tny\tnz\tv0x\tv0y\tv0z\tv1x\tv1y\tv1z\tv2x\tv2y\tv2z"
    );
    emitln!(out,
        "{}\t1\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
        clean(&track_id(&track_path)),
        best.entry.node,
        best.entry.tri_index,
        best.entry.surface,
        best.entry.surface_id,
        best.entry.rail,
        best.distance_sq.sqrt(),
        best.closest[0],
        best.closest[1],
        best.closest[2],
        best.entry.normal[0],
        best.entry.normal[1],
        best.entry.normal[2],
        tri[0][0],
        tri[0][1],
        tri[0][2],
        tri[1][0],
        tri[1][1],
        tri[1][2],
        tri[2][0],
        tri[2][1],
        tri[2][2],
    );
    Ok(out)
}

pub fn render_world_rail_contexts(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 1 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let bytes = read_bytes(&track_path)?;
    let property = kart_fmt::track_course::find_embedded_property(&bytes)
        .ok_or_else(|| format!("no embedded course in {}", track_path.display()))?;
    let course = property
        .course(false)
        .ok_or_else(|| format!("no forward course in {}", track_path.display()))?;
    let roads = kart_fmt::track_nav::discover_to_roads(&bytes);
    let graph = kart_fmt::track_nav::assemble_course_graph(course, &roads)?;
    let contexts = &graph.contexts;
    emitln!(
        out,
        "track_id\tcontext_index\troad\telement\trecord_ordinal\trecord_count\
\ttotal_length\tsuccessor_indices\tpx\tpy\tpz\tsx\tsy\tsz\tnx\tny\tnz"
    );
    for (context_index, context) in contexts.iter().enumerate() {
        let total_length = context
            .records
            .windows(2)
            .map(|pair| {
                let delta = [
                    pair[1].position[0] - pair[0].position[0],
                    pair[1].position[1] - pair[0].position[1],
                    pair[1].position[2] - pair[0].position[2],
                ];
                (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
            })
            .sum::<f32>();
        for (record_ordinal, record) in context.records.iter().enumerate() {
            emitln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{}\t{:.9}\t{:.9}\t{:.9}\
\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
                clean(&track_id(&track_path)),
                context_index,
                clean(&context.road),
                context.element,
                record_ordinal,
                context.records.len(),
                total_length,
                graph.successors[context_index]
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                record.position[0],
                record.position[1],
                record.position[2],
                record.secondary[0],
                record.secondary[1],
                record.secondary[2],
                record.normal_or_side[0],
                record.normal_or_side[1],
                record.normal_or_side[2],
            );
        }
    }
    Ok(out)
}

pub fn render_world_rail_entries(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 1 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let bytes = read_bytes(&track_path)?;
    let property = kart_fmt::track_course::find_embedded_property(&bytes)
        .ok_or_else(|| format!("no embedded course in {}", track_path.display()))?;
    let course = property
        .course(false)
        .ok_or_else(|| format!("no forward course in {}", track_path.display()))?;
    let roads = kart_fmt::track_nav::discover_to_roads(&bytes);
    let graph = kart_fmt::track_nav::assemble_course_graph(course, &roads)?;
    emitln!(
        out,
        "track_id\tcontext_index\troad\telement\tmarker\tboundary_ordinal\
\tboundary_count\tsuccessor_indices\tbx\tby\tbz\tpx\tpy\tpz\
\tsx\tsy\tsz\tnx\tny\tnz"
    );
    for (context_index, context) in graph.contexts.iter().enumerate() {
        let road = roads
            .iter()
            .find(|road| road.name == context.road)
            .ok_or_else(|| format!("missing ToRoad {}", context.road))
            .map_err(|err| RenderError::new(std::mem::take(&mut out), err))?;
        let element = road
            .elements
            .get(context.element)
            .ok_or_else(|| {
                format!(
                    "missing ToRoad element {}:{}",
                    context.road, context.element
                )
            })
            .map_err(|err| RenderError::new(std::mem::take(&mut out), err))?;
        let first = context
            .records
            .first()
            .ok_or_else(|| format!("empty rail context {}:{}", context.road, context.element))
            .map_err(|err| RenderError::new(std::mem::take(&mut out), err))?;
        for (boundary_ordinal, boundary) in element.boundary.iter().enumerate() {
            emitln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\
\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
                clean(&track_id(&track_path)),
                context_index,
                clean(&context.road),
                context.element,
                clean(&element.marker),
                boundary_ordinal,
                element.boundary.len(),
                graph.successors[context_index]
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                boundary[0],
                boundary[1],
                boundary[2],
                first.position[0],
                first.position[1],
                first.position[2],
                first.secondary[0],
                first.secondary[1],
                first.secondary[2],
                first.normal_or_side[0],
                first.normal_or_side[1],
                first.normal_or_side[2],
            );
        }
    }
    Ok(out)
}

pub fn render_world_query_ground(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 8 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <x> <y> <z> <upx> <upy> <upz> <min-dot>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let parse = |index: usize, name: &str| {
        args[index]
            .parse::<f32>()
            .map_err(|_| format!("{name} must be numeric"))
    };
    let point = [parse(1, "x")?, parse(2, "y")?, parse(3, "z")?];
    let up = [parse(4, "upx")?, parse(5, "upy")?, parse(6, "upz")?];
    let min_dot = parse(7, "min-dot")?;
    let index = load_track_query_index(&track_path)?;
    // B28D30 probes suspension along the kart's down axis.  A generic nearest
    // triangle can select an adjacent bank/edge whose Euclidean closest point
    // is nearer even though the suspension ray actually lands on the road
    // below.  Prefer a down-axis ray hit and retain nearest-point only as a
    // compatibility fallback when no authored road triangle intersects it.
    // (The selection itself — including the ray/fallback ordering — lives in
    // `query::query_ground`, shared with the typed adapter.)
    let Some(best) = crate::query::query_ground(&index, point, up, min_dot) else {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "track has no queryable ground triangle".to_string(),
        ));
    };
    let tri = best.entry.tri;
    emitln!(out, "track_id\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tdistance\tqx\tqy\tqz\tnx\tny\tnz\tv0x\tv0y\tv0z\tv1x\tv1y\tv1z\tv2x\tv2y\tv2z\tcontact_basis_44_x\tcontact_basis_44_y\tcontact_basis_44_z");
    emitln!(out, "{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
        clean(&track_id(&track_path)), best.entry.node, best.entry.tri_index, best.entry.surface,
        best.entry.surface_id, best.entry.rail, best.distance_sq.sqrt(),
        best.closest[0], best.closest[1], best.closest[2],
        best.entry.normal[0], best.entry.normal[1], best.entry.normal[2],
        tri[0][0], tri[0][1], tri[0][2], tri[1][0], tri[1][1], tri[1][2],
        tri[2][0], tri[2][1], tri[2][2],
        best.entry.contact_basis_44[0], best.entry.contact_basis_44[1], best.entry.contact_basis_44[2]);
    Ok(out)
}

pub fn render_world_sweep(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 7 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <x0> <y0> <z0> <x1> <y1> <z1>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let mut values = [0.0f32; 6];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 1]
            .parse::<f32>()
            .map_err(|_| format!("coordinate {} must be numeric", index + 1))?;
    }
    let start = [values[0], values[1], values[2]];
    let end = [values[3], values[4], values[5]];
    let index = load_track_query_index(&track_path)?;
    let best = crate::query::query_sweep(&index, start, end);
    emitln!(out, "track_id\thit\tfraction\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\thx\thy\thz\tnx\tny\tnz");
    if let Some(best) = best {
        emitln!(
            out,
            "{}\t1\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)),
            best.fraction,
            best.entry.node,
            best.entry.tri_index,
            best.entry.surface,
            best.entry.surface_id,
            best.entry.rail,
            best.point[0],
            best.point[1],
            best.point[2],
            best.entry.normal[0],
            best.entry.normal[1],
            best.entry.normal[2]
        );
    } else {
        emitln!(
            out,
            "{}\t0\t\t\t\t\t\t\t\t\t\t\t\t",
            clean(&track_id(&track_path))
        );
    }
    Ok(out)
}

pub fn render_world_sweep_sphere(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 8 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <radius> <x0> <y0> <z0> <x1> <y1> <z1>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let radius = args[1]
        .parse::<f32>()
        .map_err(|_| "radius must be numeric".to_string())?;
    if !radius.is_finite() || radius < 0.0 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "radius must be finite and non-negative".to_string(),
        ));
    }
    let mut values = [0.0f32; 6];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 2]
            .parse::<f32>()
            .map_err(|_| format!("coordinate {} must be numeric", index + 1))?;
    }
    let start = [values[0], values[1], values[2]];
    let end = [values[3], values[4], values[5]];
    let index = load_track_query_index(&track_path)?;
    let best = crate::query::query_sweep_sphere(&index, radius, start, end);
    emitln!(out, "track_id\thit\tfraction\tradius\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tcx\tcy\tcz\thx\thy\thz\tnx\tny\tnz");
    if let Some(best) = best {
        let hit = best.hit;
        emitln!(out, "{}\t1\t{:.9}\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)), hit.fraction, radius, best.entry.node, best.entry.tri_index, best.entry.surface,
            best.entry.surface_id, best.entry.rail,
            hit.center[0], hit.center[1], hit.center[2], hit.point[0], hit.point[1], hit.point[2],
            hit.normal[0], hit.normal[1], hit.normal[2]);
    } else {
        emitln!(
            out,
            "{}\t0\t\t{:.9}\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
            clean(&track_id(&track_path)),
            radius
        );
    }
    Ok(out)
}

pub fn render_world_sweep_obb(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 19 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> start(3) end(3) axes(9) half_extents(3)".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let mut values = [0.0f32; 18];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 1]
            .parse::<f32>()
            .map_err(|_| format!("OBB value {} must be numeric", index + 1))?;
    }
    let start = [values[0], values[1], values[2]];
    let end = [values[3], values[4], values[5]];
    let axes = [
        [values[6], values[7], values[8]],
        [values[9], values[10], values[11]],
        [values[12], values[13], values[14]],
    ];
    let half_extents = [values[15], values[16], values[17]];
    validate_obb(axes, half_extents)?;
    let index = load_track_query_index(&track_path)?;
    let best = crate::query::query_sweep_obb(&index, start, end, axes, half_extents);
    emitln!(out, "track_id\thit\tfraction\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tcx\tcy\tcz\tsx\tsy\tsz\tnx\tny\tnz");
    if let Some(best) = best {
        let hit = best.hit;
        emitln!(out, "{}\t1\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)), hit.fraction, best.entry.node, best.entry.tri_index, best.entry.surface,
            best.entry.surface_id, best.entry.rail,
            hit.center[0], hit.center[1], hit.center[2], hit.support_point[0], hit.support_point[1], hit.support_point[2],
            hit.normal[0], hit.normal[1], hit.normal[2]);
    } else {
        emitln!(
            out,
            "{}\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
            clean(&track_id(&track_path))
        );
    }
    Ok(out)
}

pub fn render_world_sweep_obb_rotating(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 28 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> start(3) end(3) start_axes(9) end_axes(9) half_extents(3)"
                .to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let mut values = [0.0f32; 27];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 1]
            .parse::<f32>()
            .map_err(|_| format!("rotating OBB value {} must be numeric", index + 1))?;
    }
    let start = [values[0], values[1], values[2]];
    let end = [values[3], values[4], values[5]];
    let start_axes = [
        [values[6], values[7], values[8]],
        [values[9], values[10], values[11]],
        [values[12], values[13], values[14]],
    ];
    let end_axes = [
        [values[15], values[16], values[17]],
        [values[18], values[19], values[20]],
        [values[21], values[22], values[23]],
    ];
    let half_extents = [values[24], values[25], values[26]];
    validate_obb(start_axes, half_extents)?;
    validate_obb(end_axes, half_extents)?;
    let index = load_track_query_index(&track_path)?;
    let best = crate::query::query_sweep_obb_rotating(
        &index,
        start,
        end,
        start_axes,
        end_axes,
        half_extents,
    );
    emitln!(out, "track_id\thit\tfraction\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tcx\tcy\tcz\tsx\tsy\tsz\tnx\tny\tnz");
    if let Some(best) = best {
        let hit = best.hit;
        emitln!(out, "{}\t1\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)), hit.fraction, best.entry.node, best.entry.tri_index, best.entry.surface,
            best.entry.surface_id, best.entry.rail,
            hit.center[0], hit.center[1], hit.center[2], hit.support_point[0], hit.support_point[1], hit.support_point[2],
            hit.normal[0], hit.normal[1], hit.normal[2]);
    } else {
        emitln!(
            out,
            "{}\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
            clean(&track_id(&track_path))
        );
    }
    Ok(out)
}

pub fn render_world_sweep_obb_rotating_all(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 28 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> start(3) end(3) start_axes(9) end_axes(9) half_extents(3)"
                .to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let mut values = [0.0f32; 27];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 1]
            .parse::<f32>()
            .map_err(|_| format!("rotating OBB value {} must be numeric", index + 1))?;
    }
    let start = [values[0], values[1], values[2]];
    let end = [values[3], values[4], values[5]];
    let start_axes = [
        [values[6], values[7], values[8]],
        [values[9], values[10], values[11]],
        [values[12], values[13], values[14]],
    ];
    let end_axes = [
        [values[15], values[16], values[17]],
        [values[18], values[19], values[20]],
        [values[21], values[22], values[23]],
    ];
    let half_extents = [values[24], values[25], values[26]];
    validate_obb(start_axes, half_extents)?;
    validate_obb(end_axes, half_extents)?;
    let index = load_track_query_index(&track_path)?;
    emitln!(out, "track_id\thit\tloop_index\tfraction\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tcx\tcy\tcz\tsx\tsy\tsz\tnx\tny\tnz\tface_nx\tface_ny\tface_nz\tcontact_basis_44_x\tcontact_basis_44_y\tcontact_basis_44_z");
    let hits = crate::query::query_sweep_obb_rotating_all(
        &index,
        start,
        end,
        start_axes,
        end_axes,
        half_extents,
    );
    for (loop_index, best) in hits.iter().enumerate() {
        let hit = best.hit;
        emitln!(out, "{}\t1\t{}\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)), loop_index, hit.fraction,
            best.entry.node, best.entry.tri_index, best.entry.surface,
            best.entry.surface_id,
            best.entry.rail, hit.center[0], hit.center[1], hit.center[2],
            hit.support_point[0], hit.support_point[1], hit.support_point[2],
            hit.normal[0], hit.normal[1], hit.normal[2],
            hit.face_normal[0], hit.face_normal[1], hit.face_normal[2],
            best.entry.contact_basis_44[0], best.entry.contact_basis_44[1], best.entry.contact_basis_44[2]);
    }
    Ok(out)
}

pub fn render_world_overlap_obb_all(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 16 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> center(3) axes(9) half_extents(3)".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let mut values = [0.0f32; 15];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 1]
            .parse::<f32>()
            .map_err(|_| format!("OBB overlap value {} must be numeric", index + 1))?;
    }
    let center = [values[0], values[1], values[2]];
    let axes = [
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
        [values[9], values[10], values[11]],
    ];
    let half_extents = [values[12], values[13], values[14]];
    validate_obb(axes, half_extents)?;
    let index = load_track_query_index(&track_path)?;
    emitln!(out, "track_id\thit\tloop_index\tfraction\tnode\ttri_index\troad_surface\tsurface_id\troad_rail\tcx\tcy\tcz\tsx\tsy\tsz\tnx\tny\tnz\tface_nx\tface_ny\tface_nz\tcontact_basis_44_x\tcontact_basis_44_y\tcontact_basis_44_z");
    // The two runtime-proven china_I04 JM faces copy the normalized
    // shortest directed triangle edge into CollisionFace+0x44.  Keep
    // the field explicit so consumers can gate this scoped model;
    // it is not interchangeable with the face normal.
    let hits = crate::query::query_overlap_obb_all(&index, center, axes, half_extents);
    for (loop_index, best) in hits.iter().enumerate() {
        let hit = best.hit;
        emitln!(out, "{}\t1\t{}\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)), loop_index, hit.fraction, best.entry.node, best.entry.tri_index,
            best.entry.surface, best.entry.surface_id, best.entry.rail,
            hit.center[0], hit.center[1], hit.center[2], hit.support_point[0], hit.support_point[1],
            hit.support_point[2], hit.normal[0], hit.normal[1], hit.normal[2],
            hit.face_normal[0], hit.face_normal[1], hit.face_normal[2],
            best.entry.contact_basis_44[0], best.entry.contact_basis_44[1], best.entry.contact_basis_44[2]);
    }
    Ok(out)
}

/// Dynamic `ToMovableObject` CONTACT candidates.  This is a new P2 command,
/// so it does not alter any pre-existing TSV bytes; its field order is pinned
/// by the Python adapter parser and the focused renderer test.
pub fn render_world_event_obstacle_contacts(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 18 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <time-ms> <delta-ms> center(3) axes(9) half_extents(3)"
                .to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let time_ms = args[1]
        .parse::<i64>()
        .map_err(|_| "time-ms must be an integer".to_string())?;
    let delta_ms = args[2]
        .parse::<i64>()
        .map_err(|_| "delta-ms must be an integer".to_string())?;
    let mut values = [0.0f64; 15];
    for (index, value) in values.iter_mut().enumerate() {
        *value = args[index + 3]
            .parse::<f64>()
            .map_err(|_| format!("event OBB value {} must be numeric", index + 1))?;
    }
    let center = [values[0], values[1], values[2]];
    // CLI callers pass three vehicle axes, while the typed World boundary
    // carries the engine's row-major matrix.  Rebuild that matrix with the
    // supplied axes as columns.
    let matrix = [
        values[3], values[6], values[9], values[4], values[7], values[10], values[5], values[8],
        values[11],
    ];
    let half_extents = [values[12], values[13], values[14]];
    let world = TrackWorld::from_path(&track_path)?;
    let contacts = world
        .query_event_obstacle_contacts(center, matrix, half_extents, time_ms, delta_ms)
        .ok_or_else(|| "event-obstacle query rejected its time/delta/OBB arguments".to_string())?;
    emitln!(
        out,
        "track_id\tevent_index\tobject_id\tnode\ttri_index\tloop_index\tcx\tcy\tcz\tnx\tny\tnz\tmvx\tmvy\tmvz\topx\topy\topz"
    );
    for contact in contacts {
        emitln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            clean(&track_id(&track_path)),
            contact.event_index,
            clean(&contact.object_id),
            clean(&contact.node),
            contact.tri_index,
            contact.loop_index,
            contact.centroid[0],
            contact.centroid[1],
            contact.centroid[2],
            contact.normal[0],
            contact.normal[1],
            contact.normal[2],
            contact.motion[0],
            contact.motion[1],
            contact.motion[2],
            contact.obstacle_position[0],
            contact.obstacle_position[1],
            contact.obstacle_position[2],
        );
    }
    Ok(out)
}

/// Time-sampled GoItemBanana run-over selector. This is an item-effect
/// query, deliberately separate from event obstacle CONTACT candidates.
pub fn render_world_banana_trigger(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 5 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s> <time-ms> position(3)".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let time_ms = args[1]
        .parse::<i64>()
        .map_err(|_| "time-ms must be an integer".to_string())?;
    let position = [
        args[2]
            .parse::<f64>()
            .map_err(|_| "x must be numeric".to_string())?,
        args[3]
            .parse::<f64>()
            .map_err(|_| "y must be numeric".to_string())?,
        args[4]
            .parse::<f64>()
            .map_err(|_| "z must be numeric".to_string())?,
    ];
    let world = TrackWorld::from_path(&track_path)?;
    emitln!(
        out,
        "track_id\tevent_index\tobject_id\tpx\tpy\tpz\tdistance\tradius\tauthority"
    );
    if let Some(hit) = world.query_banana_trigger(position, time_ms) {
        emitln!(
            out,
            "{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{}",
            clean(&track_id(&track_path)),
            hit.event_index,
            clean(&hit.object_id),
            hit.position[0],
            hit.position[1],
            hit.position[2],
            hit.distance,
            hit.radius,
            clean(&hit.authority),
        );
    }
    Ok(out)
}

/// Decoded event-object inventory with the world adapter's scoped physical
/// disposition.  Python uses this one-shot query to cache the capability
/// gate before opening the hot persistent geometry-query session.
pub fn render_world_event_obstacles(args: &[String]) -> Result<String, RenderError> {
    let mut out = String::new();
    if args.len() != 1 {
        return Err(RenderError::new(
            std::mem::take(&mut out),
            "expected <track.1s>".to_string(),
        ));
    }
    let track_path = PathBuf::from(&args[0]);
    let world = TrackWorld::from_path(&track_path)?;
    emitln!(
        out,
        "track_id\tevent_index\tobject_id\tclass_name\tname\tobject_type\tpx\tpy\tpz\tdisposition\tphysics_enabled"
    );
    for object in world.event_objects() {
        let (disposition, enabled) = match object.disposition {
            EventPhysicsDisposition::ContactDynamicObstacle => ("CONTACT_DYNAMIC_OBSTACLE", 1),
            EventPhysicsDisposition::VisualOnlyUnadjudicated => ("VISUAL_ONLY_UNADJUDICATED", 0),
        };
        emitln!(
            out,
            "{}\t{}\tevent:{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}",
            clean(&track_id(&track_path)),
            object.event_index,
            object.event_index,
            object.class_name,
            clean(&object.name),
            clean(object.object_type.as_deref().unwrap_or("")),
            object.position[0],
            object.position[1],
            object.position[2],
            disposition,
            enabled
        );
    }
    Ok(out)
}

fn print_rendered(result: Result<String, RenderError>) -> Result<(), String> {
    match result {
        Ok(out) => {
            print!("{out}");
            Ok(())
        }
        Err(err) => {
            print!("{}", err.partial);
            Err(err.message)
        }
    }
}

pub fn run_kap_markers(args: &[String]) -> Result<(), String> {
    print_rendered(render_kap_markers(args))
}

pub fn run_kap_points(args: &[String]) -> Result<(), String> {
    print_rendered(render_kap_points(args))
}

pub fn run_kap_topology(args: &[String]) -> Result<(), String> {
    print_rendered(render_kap_topology(args))
}

pub fn run_world_node_info(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_node_info(args))
}

pub fn run_world_node_triangle(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_node_triangle(args))
}

pub fn run_world_dynamic_objects(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_dynamic_objects(args))
}

pub fn run_world_dynamic_state(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_dynamic_state(args))
}

pub fn run_world_query_moving(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_query_moving(args))
}

pub fn run_world_query_dynamic(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_query_dynamic(args))
}

pub fn run_world_query(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_query(args))
}

pub fn run_world_query_rail(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_query_rail(args))
}

pub fn run_world_rail_contexts(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_rail_contexts(args))
}

pub fn run_world_rail_entries(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_rail_entries(args))
}

pub fn run_world_query_ground(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_query_ground(args))
}

pub fn run_world_sweep(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_sweep(args))
}

pub fn run_world_sweep_sphere(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_sweep_sphere(args))
}

pub fn run_world_sweep_obb(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_sweep_obb(args))
}

pub fn run_world_sweep_obb_rotating(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_sweep_obb_rotating(args))
}

pub fn run_world_sweep_obb_rotating_all(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_sweep_obb_rotating_all(args))
}

pub fn run_world_overlap_obb_all(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_overlap_obb_all(args))
}

pub fn run_world_event_obstacle_contacts(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_event_obstacle_contacts(args))
}

pub fn run_world_banana_trigger(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_banana_trigger(args))
}

pub fn run_world_event_obstacles(args: &[String]) -> Result<(), String> {
    print_rendered(render_world_event_obstacles(args))
}

// ---------------------------------------------------------------------------
// Shared command dispatch (CLI + TSV regression test)
// ---------------------------------------------------------------------------

/// The query/TSV commands owned by this crate (everything except the
/// persistent `world-query-server` session, which dispatches per-request
/// through [`run_world_query_server_session`]).
pub fn is_query_command(command: &str) -> bool {
    matches!(
        command,
        "kap-markers"
            | "kap-points"
            | "kap-topology"
            | "world-node-info"
            | "world-node-triangle"
            | "world-dynamic-objects"
            | "world-dynamic-state"
            | "world-query-moving"
            | "world-query-dynamic"
            | "world-query"
            | "world-query-rail"
            | "world-rail-contexts"
            | "world-rail-entries"
            | "world-query-ground"
            | "world-sweep"
            | "world-sweep-sphere"
            | "world-sweep-obb"
            | "world-sweep-obb-rotating"
            | "world-sweep-obb-rotating-all"
            | "world-overlap-obb-all"
            | "world-event-obstacles"
            | "world-event-obstacle-contacts"
            | "world-banana-trigger"
    )
}

/// Single dispatch point for the query commands, shared by the kart-autoprobe
/// CLI and the TSV regression test (`tests/tsv_regression.rs`).
pub fn render_command(command: &str, args: &[String]) -> Result<String, RenderError> {
    match command {
        "kap-markers" => render_kap_markers(args),
        "kap-points" => render_kap_points(args),
        "kap-topology" => render_kap_topology(args),
        "world-node-info" => render_world_node_info(args),
        "world-node-triangle" => render_world_node_triangle(args),
        "world-dynamic-objects" => render_world_dynamic_objects(args),
        "world-dynamic-state" => render_world_dynamic_state(args),
        "world-query-moving" => render_world_query_moving(args),
        "world-query-dynamic" => render_world_query_dynamic(args),
        "world-query" => render_world_query(args),
        "world-query-rail" => render_world_query_rail(args),
        "world-rail-contexts" => render_world_rail_contexts(args),
        "world-rail-entries" => render_world_rail_entries(args),
        "world-query-ground" => render_world_query_ground(args),
        "world-sweep" => render_world_sweep(args),
        "world-sweep-sphere" => render_world_sweep_sphere(args),
        "world-sweep-obb" => render_world_sweep_obb(args),
        "world-sweep-obb-rotating" => render_world_sweep_obb_rotating(args),
        "world-sweep-obb-rotating-all" => render_world_sweep_obb_rotating_all(args),
        "world-overlap-obb-all" => render_world_overlap_obb_all(args),
        "world-event-obstacles" => render_world_event_obstacles(args),
        "world-event-obstacle-contacts" => render_world_event_obstacle_contacts(args),
        "world-banana-trigger" => render_world_banana_trigger(args),
        other => Err(RenderError::new(
            String::new(),
            format!("unknown query command: {other}"),
        )),
    }
}

/// Printing shell over [`render_command`] (same partial-output-on-error
/// semantics as the individual `run_*` shells).
pub fn run_command(command: &str, args: &[String]) -> Result<(), String> {
    print_rendered(render_command(command, args))
}

/// Persistent `world-query-server` session: the ready marker, then one TSV
/// payload plus `__KR_WORLD_END__` frame per tab-separated query line, until
/// `quit` or EOF. The eager parse runs before the ready marker so startup
/// failure is reported first (same byte stream as the historical loop).
pub fn run_world_query_server_session<R: std::io::BufRead, W: std::io::Write>(
    track_path: &std::path::Path,
    input: R,
    out: &mut W,
) -> Result<(), String> {
    load_static_track(track_path)?;
    writeln!(out, "__KR_WORLD_READY__").map_err(|err| format!("flush ready: {err}"))?;
    out.flush().map_err(|err| format!("flush ready: {err}"))?;
    for line in input.lines() {
        let line = line.map_err(|err| format!("read query: {err}"))?;
        let fields = line.split('\t').map(str::to_string).collect::<Vec<_>>();
        if fields.is_empty() || fields[0].is_empty() {
            continue;
        }
        if fields[0] == "quit" {
            break;
        }
        let mut query_args = vec![track_path.to_string_lossy().into_owned()];
        query_args.extend(fields.iter().skip(1).cloned());
        let result = match fields[0].as_str() {
            "world-query-ground" => render_world_query_ground(&query_args),
            "world-overlap-obb-all" => render_world_overlap_obb_all(&query_args),
            "world-event-obstacle-contacts" => render_world_event_obstacle_contacts(&query_args),
            "world-banana-trigger" => render_world_banana_trigger(&query_args),
            other => Err(RenderError::new(
                String::new(),
                format!("unsupported persistent query: {other}"),
            )),
        };
        match result {
            Ok(tsv) => {
                write!(out, "{tsv}").map_err(|err| format!("write query: {err}"))?;
                writeln!(out, "__KR_WORLD_END__\tOK")
                    .map_err(|err| format!("flush query: {err}"))?;
            }
            Err(err) => {
                write!(out, "{}", err.partial).map_err(|err| format!("write query: {err}"))?;
                writeln!(out, "__KR_WORLD_END__\tERR\t{}", clean(&err.message))
                    .map_err(|err| format!("flush query: {err}"))?;
            }
        }
        out.flush().map_err(|err| format!("flush query: {err}"))?;
    }
    Ok(())
}
