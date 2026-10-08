//! Audit an embedded track course against a captured registered-context graph.
//!
//! Usage: audit_track_course <track.1s> <registered-contexts.json> [aiPath-dir]

use kart_fmt::bml::Tag;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn tag_attr(tag: Option<&kart_fmt::kart_object::BinaryXmlTag>, child: &str, key: &str) -> String {
    tag.and_then(|root| {
        root.children
            .iter()
            .find(|item| item.name == child)
            .and_then(|item| item.attrs.iter().find(|(name, _)| name == key))
            .map(|(_, value)| value.clone())
    })
    .unwrap_or_default()
}

fn collect_roads(tag: &Tag, path: &str, out: &mut Vec<Value>) {
    let here = if path.is_empty() {
        tag.name.clone()
    } else {
        format!("{path}/{}", tag.name)
    };
    if tag.name == "road" {
        out.push(json!({
            "path": here,
            "name": tag.attr("name").unwrap_or(""),
            "start": tag.attr("start").unwrap_or(""),
            "end": tag.attr("end").unwrap_or(""),
        }));
    }
    for (index, child) in tag.children.iter().enumerate() {
        collect_roads(child, &format!("{here}[{index}]"), out);
    }
}

fn transform(matrix: &[f32; 16], point: [f32; 3]) -> [f64; 3] {
    [
        (matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12]) as f64,
        (matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13]) as f64,
        (matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14]) as f64,
    ]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn json_vec3(value: &Value) -> Option<[f32; 3]> {
    let values = value.as_array()?;
    (values.len() == 3).then_some(())?;
    Some([
        values[0].as_f64()? as f32,
        values[1].as_f64()? as f32,
        values[2].as_f64()? as f32,
    ])
}

fn context_matches_records(context: &Value, records: &[kart_fmt::track_nav::NavRecord]) -> bool {
    let Some(runtime_records) = context["records"].as_array() else {
        return false;
    };
    runtime_records.len() == records.len()
        && runtime_records.iter().zip(records).all(|(runtime, asset)| {
            json_vec3(&runtime["position"]) == Some(asset.position)
                && json_vec3(&runtime["secondary"]) == Some(asset.secondary)
                && json_vec3(&runtime["normal_or_side"]) == Some(asset.normal_or_side)
        })
}

fn context_matches_embedded_vector(
    context: &Value,
    vector: &kart_fmt::track_nav::NavRecordVector,
) -> bool {
    context_matches_records(context, &vector.records)
}

fn edge_midpoints(track: &kart_fmt::kart_object::Track) -> Vec<(String, [f64; 3])> {
    let mut out = Vec::new();
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        let Some(geom) = &node.geom else { continue };
        let triangles: Vec<[u32; 3]> = if geom.strip {
            geom.indices
                .windows(3)
                .enumerate()
                .filter_map(|(index, values)| {
                    let triangle = if index % 2 == 0 {
                        [values[0], values[1], values[2]]
                    } else {
                        [values[1], values[0], values[2]]
                    };
                    (triangle[0] != triangle[1]
                        && triangle[1] != triangle[2]
                        && triangle[0] != triangle[2])
                        .then_some(triangle)
                })
                .collect()
        } else {
            geom.indices
                .chunks_exact(3)
                .map(|values| [values[0], values[1], values[2]])
                .collect()
        };
        for triangle in triangles {
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                if a as usize >= geom.vertices.len() || b as usize >= geom.vertices.len() {
                    continue;
                }
                let pa = transform(&world, geom.vertices[a as usize]);
                let pb = transform(&world, geom.vertices[b as usize]);
                out.push((
                    node.name.clone(),
                    [
                        (pa[0] + pb[0]) * 0.5,
                        (pa[1] + pb[1]) * 0.5,
                        (pa[2] + pb[2]) * 0.5,
                    ],
                ));
            }
        }
    }
    out
}

fn shared_portal_midpoints(track: &kart_fmt::kart_object::Track) -> Vec<(String, [f64; 3])> {
    let mut out = Vec::new();
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        let Some(geom) = &node.geom else { continue };
        let triangles: Vec<[u32; 3]> = if geom.strip {
            geom.indices
                .windows(3)
                .enumerate()
                .filter_map(|(index, values)| {
                    let triangle = if index % 2 == 0 {
                        [values[0], values[1], values[2]]
                    } else {
                        [values[1], values[0], values[2]]
                    };
                    (triangle[0] != triangle[1]
                        && triangle[1] != triangle[2]
                        && triangle[0] != triangle[2])
                        .then_some(triangle)
                })
                .collect()
        } else {
            geom.indices
                .chunks_exact(3)
                .map(|values| [values[0], values[1], values[2]])
                .collect()
        };
        let mut edges = BTreeMap::<[i64; 6], ([f64; 3], [f64; 3], usize)>::new();
        for triangle in triangles {
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                if a as usize >= geom.vertices.len() || b as usize >= geom.vertices.len() {
                    continue;
                }
                let pa = transform(&world, geom.vertices[a as usize]);
                let pb = transform(&world, geom.vertices[b as usize]);
                let qa = pa.map(|value| (value * 10_000.0).round() as i64);
                let qb = pb.map(|value| (value * 10_000.0).round() as i64);
                let (left, right, qleft, qright) = if qa <= qb {
                    (pa, pb, qa, qb)
                } else {
                    (pb, pa, qb, qa)
                };
                let key = [
                    qleft[0], qleft[1], qleft[2], qright[0], qright[1], qright[2],
                ];
                edges
                    .entry(key)
                    .and_modify(|entry| entry.2 += 1)
                    .or_insert((left, right, 1));
            }
        }
        for (_, (a, b, count)) in edges {
            if count >= 2 {
                out.push((
                    node.name.clone(),
                    [
                        (a[0] + b[0]) * 0.5,
                        (a[1] + b[1]) * 0.5,
                        (a[2] + b[2]) * 0.5,
                    ],
                ));
            }
        }
    }
    out
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let track_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: audit_track_course <track.1s> <registered-contexts.json>")?;
    let snapshot_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: audit_track_course <track.1s> <registered-contexts.json>")?;
    let kap_dir = args.next().map(PathBuf::from);
    let bytes = std::fs::read(&track_path)
        .map_err(|error| format!("read {}: {error}", track_path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|error| format!("parse {}: {error}", track_path.display()))?;
    let property = kart_fmt::track_course::find_embedded_property(&bytes)
        .ok_or_else(|| format!("no property/course in {}", track_path.display()))?;
    let snapshot: Value = serde_json::from_slice(
        &std::fs::read(&snapshot_path)
            .map_err(|error| format!("read {}: {error}", snapshot_path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", snapshot_path.display()))?;
    let mut kap_points = Vec::<(String, [f64; 3])>::new();
    if let Some(dir) = &kap_dir {
        let mut paths = std::fs::read_dir(dir)
            .map_err(|error| format!("read_dir {}: {error}", dir.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("kap"))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let encoded = std::fs::read(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            let clear = kart_rho::encoded_block::decode_len_prefixed(&encoded)
                .map_err(|error| format!("decode {}: {error}", path.display()))?;
            let kap =
                kart_fmt::kap::parse(&clear).ok_or_else(|| format!("parse {}", path.display()))?;
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_string();
            kap_points.extend(
                kap.points
                    .into_iter()
                    .map(|point| (name.clone(), point.map(f64::from))),
            );
        }
    }

    let mut forward_roads = Vec::new();
    collect_roads(
        property.course(false).ok_or("missing forward course")?,
        "",
        &mut forward_roads,
    );
    let mut reverse_roads = Vec::new();
    if let Some(reverse) = property.course(true) {
        collect_roads(reverse, "", &mut reverse_roads);
    }

    let mut authored_nodes = BTreeMap::<String, Vec<Value>>::new();
    let authored_bases = forward_roads
        .iter()
        .chain(reverse_roads.iter())
        .filter_map(|road| road["name"].as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    for node in &track.nodes {
        for base in &authored_bases {
            if node.name == base.as_str() || node.name.starts_with(&format!("{base}@s")) {
                authored_nodes.entry(base.to_string()).or_default().push(json!({
                    "node": node.name,
                    "vertex_count": node.geom.as_ref().map(|geom| geom.vertices.len()).unwrap_or(0),
                    "rail": tag_attr(node.extra_tag.as_ref(), "road", "rail"),
                    "surface": tag_attr(node.extra_tag.as_ref(), "road", "surface"),
                }));
            }
        }
    }

    let midpoints = edge_midpoints(&track);
    let portal_midpoints = shared_portal_midpoints(&track);
    let contexts = snapshot["contexts"]
        .as_array()
        .ok_or("snapshot contexts is not an array")?;
    let embedded_nav_vectors = kart_fmt::track_nav::discover_record_vectors(&bytes);
    let to_roads = kart_fmt::track_nav::discover_to_roads(&bytes);
    let forward_course = property
        .course(false)
        .ok_or("embedded property has no forward course")?;
    let assembled_nav = kart_fmt::track_nav::assemble_course(forward_course, &to_roads)?;
    let exact_assembled_contexts = contexts
        .iter()
        .zip(&assembled_nav)
        .take_while(|(context, selected)| context_matches_records(context, &selected.records))
        .count();
    let exact_embedded_contexts = contexts
        .iter()
        .filter(|context| {
            embedded_nav_vectors
                .iter()
                .any(|vector| context_matches_embedded_vector(context, vector))
        })
        .count();
    let mut total_records = 0usize;
    let mut exact_records = 0usize;
    let mut exact_portal_records = 0usize;
    let mut exact_kap_records = 0usize;
    let mut nearest_kap_errors = Vec::new();
    let mut selected_context_28 = Vec::new();
    for context in contexts {
        let context_index = context["context_index"].as_u64().unwrap_or(u64::MAX);
        for record in context["records"].as_array().into_iter().flatten() {
            let values = record["position"]
                .as_array()
                .ok_or("position is not an array")?;
            let point = [
                values[0].as_f64().ok_or("invalid position x")?,
                values[1].as_f64().ok_or("invalid position y")?,
                values[2].as_f64().ok_or("invalid position z")?,
            ];
            let (node, error) = midpoints
                .iter()
                .map(|(node, midpoint)| (node, distance(point, *midpoint)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .ok_or("track has no mesh edges")?;
            total_records += 1;
            if error <= 1.0e-3 {
                exact_records += 1;
            }
            if portal_midpoints
                .iter()
                .any(|(_, midpoint)| distance(point, *midpoint) <= 1.0e-3)
            {
                exact_portal_records += 1;
            }
            if let Some(kap_error) = kap_points
                .iter()
                .map(|(_, candidate)| distance(point, *candidate))
                .min_by(f64::total_cmp)
            {
                if kap_error <= 1.0e-3 {
                    exact_kap_records += 1;
                }
                nearest_kap_errors.push(kap_error);
            }
            if context_index == 28 {
                selected_context_28.push(json!({
                    "position": point,
                    "nearest_authored_edge_midpoint_node": node,
                    "error": error,
                }));
            }
        }
    }
    nearest_kap_errors.sort_by(f64::total_cmp);
    let kap_error_percentile = |fraction: f64| {
        (!nearest_kap_errors.is_empty()).then(|| {
            nearest_kap_errors[((nearest_kap_errors.len() - 1) as f64 * fraction).round() as usize]
        })
    };

    let result = json!({
        "schema": "kartrider.track_course_asset_audit.v1",
        "track_asset": track_path,
        "runtime_snapshot": snapshot_path,
        "property_offset": property.offset,
        "property_offset_hex": format!("0x{:X}", property.offset),
        "property_encoded_len": property.encoded_len,
        "track_event_count": track.event_count,
        "forward_roads": forward_roads,
        "reverse_roads": reverse_roads,
        "authored_scene_nodes": authored_nodes,
        "runtime_context_count": contexts.len(),
        "runtime_record_count": total_records,
        "embedded_nav_vector_count": embedded_nav_vectors.len(),
        "embedded_nav_record_count": embedded_nav_vectors.iter().map(|vector| vector.records.len()).sum::<usize>(),
        "to_road_count": to_roads.len(),
        "assembled_forward_context_count": assembled_nav.len(),
        "runtime_context_order_exact_assembled_prefix": exact_assembled_contexts,
        "assembled_forward_contexts": assembled_nav.iter().map(|selected| json!({
            "road": selected.road,
            "element": selected.element,
            "record_count": selected.records.len(),
        })).collect::<Vec<_>>(),
        "runtime_context_vectors_exact_embedded": exact_embedded_contexts,
        "records_equal_authored_mesh_edge_midpoint_1e_3": exact_records,
        "records_equal_shared_triangle_portal_midpoint_1e_3": exact_portal_records,
        "kap_file_count": kap_points.iter().map(|(name, _)| name).collect::<BTreeSet<_>>().len(),
        "kap_point_count": kap_points.len(),
        "records_equal_kap_point_1e_3": exact_kap_records,
        "nearest_kap_point_error": {
            "min": nearest_kap_errors.first(),
            "p50": kap_error_percentile(0.5),
            "p90": kap_error_percentile(0.9),
            "max": nearest_kap_errors.last(),
        },
        "context_28_asset_provenance": selected_context_28,
        "closed_subset": "autonomous track.1s property/course discovery, branch/road/marker decode, scene-node resolution, embedded navigation-vector decode, and selected state1 context geometric provenance",
        "remaining_boundary": "derive and validate the state1 entry scalar against the autonomously assembled 38-context graph",
        "verdict": if contexts.len() == 38
            && total_records == 236
            && exact_embedded_contexts == contexts.len()
            && assembled_nav.len() == contexts.len()
            && exact_assembled_contexts == contexts.len()
            && selected_context_28.iter().all(|item| item["error"].as_f64().unwrap_or(f64::INFINITY) <= 1.0e-3)
        {
            "PASS_TRACK_COURSE_ASSET_PROVENANCE"
        } else {
            "FAIL_TRACK_COURSE_ASSET_PROVENANCE"
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}
