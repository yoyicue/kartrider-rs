//! Topological reconstruction of road-triangle corridors.
//!
//! KartRider's RAIL registration walks connected road triangles before it
//! emits its 36-byte path records.  This module deliberately stops at the
//! asset-owned part of that operation: expand a mesh into triangles, join
//! triangles that share an edge, and order every non-branching component from
//! one endpoint to the other.  Record splitting and junction adjustment remain
//! a separate runtime-equivalence step.

use crate::kart_object::{Geom, Vec3};
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct Portal {
    pub from_triangle: usize,
    pub to_triangle: usize,
    pub endpoints: [Vec3; 2],
    pub midpoint: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleCorridor {
    /// Triangle ordinals in deterministic endpoint-to-endpoint order.
    pub triangles: Vec<usize>,
    pub portals: Vec<Portal>,
    /// True only when the component is a path (including a single triangle).
    pub is_chain: bool,
}

/// Expand a triangle-list or triangle-strip geometry into non-degenerate
/// triangle vertex indices.
pub fn triangles(geom: &Geom) -> Vec<[u32; 3]> {
    if geom.strip {
        geom.indices
            .windows(3)
            .enumerate()
            .filter_map(|(ordinal, values)| {
                let triangle = if ordinal % 2 == 0 {
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
            .filter_map(|values| {
                let triangle = [values[0], values[1], values[2]];
                (triangle[0] != triangle[1]
                    && triangle[1] != triangle[2]
                    && triangle[0] != triangle[2])
                    .then_some(triangle)
            })
            .collect()
    }
}

fn squared_distance(a: Vec3, b: Vec3) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn shared_edge(a: [Vec3; 3], b: [Vec3; 3], tolerance: f32) -> Option<[Vec3; 2]> {
    let tolerance2 = tolerance * tolerance;
    let mut shared = Vec::with_capacity(2);
    for point_a in a {
        if b.iter()
            .any(|&point_b| squared_distance(point_a, point_b) <= tolerance2)
            && !shared
                .iter()
                .any(|&point| squared_distance(point, point_a) <= tolerance2)
        {
            shared.push(point_a);
        }
    }
    (shared.len() == 2).then(|| [shared[0], shared[1]])
}

/// Reconstruct connected triangle components and, for path-shaped components,
/// their exact traversal order and shared-edge portals.
///
/// `tolerance` is expressed in mesh-local units.  A small nonzero value is
/// needed because track meshes commonly duplicate edge vertices with a few
/// floating-point ulps of drift.
pub fn reconstruct(geom: &Geom, tolerance: f32) -> Vec<TriangleCorridor> {
    let indices = triangles(geom);
    let points = indices
        .iter()
        .map(|triangle| triangle.map(|index| geom.vertices[index as usize]))
        .collect::<Vec<_>>();
    let mut adjacency = vec![Vec::<(usize, [Vec3; 2])>::new(); points.len()];
    for left in 0..points.len() {
        for right in left + 1..points.len() {
            if let Some(edge) = shared_edge(points[left], points[right], tolerance) {
                adjacency[left].push((right, edge));
                adjacency[right].push((left, edge));
            }
        }
    }

    let mut visited = vec![false; points.len()];
    let mut corridors = Vec::new();
    for seed in 0..points.len() {
        if visited[seed] {
            continue;
        }
        let mut queue = VecDeque::from([seed]);
        let mut component = Vec::new();
        visited[seed] = true;
        while let Some(triangle) = queue.pop_front() {
            component.push(triangle);
            for &(neighbor, _) in &adjacency[triangle] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_unstable();
        let is_chain = component.len() == 1
            || (component.iter().all(|&index| adjacency[index].len() <= 2)
                && component
                    .iter()
                    .filter(|&&index| adjacency[index].len() == 1)
                    .count()
                    == 2);
        if !is_chain {
            corridors.push(TriangleCorridor {
                triangles: component,
                portals: Vec::new(),
                is_chain: false,
            });
            continue;
        }

        let start = component
            .iter()
            .copied()
            .filter(|&index| adjacency[index].len() <= 1)
            .min()
            .unwrap_or(seed);
        let mut ordered = Vec::with_capacity(component.len());
        let mut portals = Vec::with_capacity(component.len().saturating_sub(1));
        let mut previous = None;
        let mut current = Some(start);
        while let Some(triangle) = current {
            ordered.push(triangle);
            let next = adjacency[triangle]
                .iter()
                .filter(|(neighbor, _)| Some(*neighbor) != previous)
                .min_by_key(|(neighbor, _)| *neighbor);
            if let Some(&(neighbor, endpoints)) = next {
                portals.push(Portal {
                    from_triangle: triangle,
                    to_triangle: neighbor,
                    endpoints,
                    midpoint: [
                        (endpoints[0][0] + endpoints[1][0]) * 0.5,
                        (endpoints[0][1] + endpoints[1][1]) * 0.5,
                        (endpoints[0][2] + endpoints[1][2]) * 0.5,
                    ],
                });
                previous = Some(triangle);
                current = Some(neighbor);
            } else {
                current = None;
            }
        }
        corridors.push(TriangleCorridor {
            triangles: ordered,
            portals,
            is_chain: true,
        });
    }
    corridors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(vertices: Vec<Vec3>, indices: Vec<u32>) -> Geom {
        Geom {
            normals: vec![[0.0; 3]; vertices.len()],
            uvs: vec![[0.0; 2]; vertices.len()],
            vertices,
            indices,
            strip: false,
        }
    }

    #[test]
    fn orders_a_triangle_chain_and_emits_portals() {
        let mesh = geom(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            vec![0, 1, 2, 2, 1, 3, 1, 4, 3],
        );
        let corridors = reconstruct(&mesh, 1.0e-5);
        assert_eq!(corridors.len(), 1);
        assert!(corridors[0].is_chain);
        assert_eq!(corridors[0].triangles, vec![0, 1, 2]);
        assert_eq!(corridors[0].portals.len(), 2);
        assert_eq!(corridors[0].portals[0].midpoint, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn joins_duplicated_edge_vertices_within_tolerance() {
        let mesh = geom(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.000_01, 0.0, 0.0],
                [0.000_01, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 4, 3, 5],
        );
        assert_eq!(reconstruct(&mesh, 1.0e-6).len(), 2);
        let corridors = reconstruct(&mesh, 1.0e-3);
        assert_eq!(corridors.len(), 1);
        assert_eq!(corridors[0].triangles, vec![0, 1]);
    }

    #[test]
    fn reports_a_branch_without_inventing_an_order() {
        let mesh = geom(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 1, 0, 3, 2, 1, 4, 0, 2, 5],
        );
        let corridors = reconstruct(&mesh, 1.0e-5);
        assert_eq!(corridors.len(), 1);
        assert!(!corridors[0].is_chain);
        assert!(corridors[0].portals.is_empty());
    }
}
