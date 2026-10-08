use super::*;

pub(crate) struct NavMesh {
    tris: Vec<NavTri>,
    main_component: Option<usize>,
    main_center: P3,
}

#[derive(Clone, Copy)]
struct NavTri {
    center: P3,
    component: usize,
    vertices: [P3; 3],
}

impl NavMesh {
    pub(crate) fn from_track(track_path: &Path) -> Result<Self, String> {
        let bytes = fs::read(track_path)
            .map_err(|err| format!("read nav track {}: {err}", track_path.display()))?;
        let track = kart_fmt::kart_object::parse(&bytes)
            .map_err(|err| format!("parse nav track {}: {err}", track_path.display()))?;
        let mut tris = Vec::<(P3, [P3; 3])>::new();

        for (node_idx, world) in track.flatten() {
            let node = &track.nodes[node_idx];
            if !has_tag(node.extra_tag.as_ref(), "road") {
                continue;
            }
            if is_non_drivable_surface(
                tag_attr(node.extra_tag.as_ref(), "road", "surface").as_deref(),
            ) {
                continue;
            }
            let Some(geom) = &node.geom else { continue };
            for tri_idx in geom_triangles(geom) {
                if tri_idx.iter().any(|&i| i as usize >= geom.vertices.len()) {
                    continue;
                }
                let tri = [
                    xform(&world, geom.vertices[tri_idx[0] as usize]),
                    xform(&world, geom.vertices[tri_idx[1] as usize]),
                    xform(&world, geom.vertices[tri_idx[2] as usize]),
                ];
                if tri
                    .iter()
                    .any(|vertex| vertex.iter().any(|coord| !coord.is_finite()))
                {
                    continue;
                }
                if oriented_normal_z(&world, tri) < 0.35 {
                    continue;
                }
                tris.push((tri_center(tri), tri));
            }
        }

        let mut dsu = DisjointSet::new(tris.len());
        let mut by_vertex = BTreeMap::<VertexKey, Vec<usize>>::new();
        for (idx, (_, tri)) in tris.iter().enumerate() {
            for vertex in tri {
                by_vertex
                    .entry(VertexKey::from(*vertex))
                    .or_default()
                    .push(idx);
            }
        }
        for indices in by_vertex.values() {
            if let Some((&first, rest)) = indices.split_first() {
                for &other in rest {
                    dsu.union(first, other);
                }
            }
        }

        let mut nav_tris: Vec<_> = tris
            .into_iter()
            .enumerate()
            .map(|(idx, (center, vertices))| NavTri {
                center,
                component: dsu.find(idx),
                vertices,
            })
            .collect();
        let main_component = largest_component(&nav_tris);
        let main_center = component_center(&nav_tris, main_component);
        nav_tris.shrink_to_fit();

        Ok(Self {
            tris: nav_tris,
            main_component,
            main_center,
        })
    }

    pub(crate) fn nearest_component(&self, point: P3) -> Option<usize> {
        self.nearest_tri(point).map(|tri| tri.component)
    }

    pub(crate) fn support_distance(&self, point: P3) -> Option<f32> {
        self.tris
            .iter()
            .map(|tri| {
                let closest = closest_point_on_triangle(point, tri.vertices);
                squared_distance(point, closest).sqrt()
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn nearest_tri(&self, point: P3) -> Option<NavTri> {
        self.tris
            .iter()
            .min_by(|a, b| {
                point_score(a.center, point)
                    .partial_cmp(&point_score(b.center, point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    pub(crate) fn ring_score(&self, pose: Pose, target: P3) -> Option<f32> {
        let main_component = self.main_component?;
        let pose_tri = self.nearest_tri(pose.pos)?;
        let target_tri = self.nearest_tri(target)?;
        if pose_tri.component != main_component || target_tri.component != main_component {
            return None;
        }

        let pose_angle = ring_angle(self.main_center, pose.pos);
        let target_angle = ring_angle(self.main_center, target);
        let orientation = ring_orientation(self.main_center, pose);
        let forward_delta = if orientation >= 0.0 {
            positive_angle_delta(target_angle - pose_angle)
        } else {
            positive_angle_delta(pose_angle - target_angle)
        };

        let distance = point_score(target, pose.pos);
        Some(forward_delta * 90.0 + distance * 0.15)
    }
}

/// Up-component of the triangle normal, corrected for winding flips caused by
/// mirroring transforms: a negative-determinant world matrix reverses the
/// authored winding, so the raw cross-product normal must be negated to
/// recover the authored facing.
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

fn largest_component(tris: &[NavTri]) -> Option<usize> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for tri in tris {
        *counts.entry(tri.component).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(component, _)| component)
}

fn component_center(tris: &[NavTri], component: Option<usize>) -> P3 {
    let Some(component) = component else {
        return [0.0, 0.0, 0.0];
    };
    let mut sum = [0.0, 0.0, 0.0];
    let mut count = 0.0;
    for tri in tris.iter().filter(|tri| tri.component == component) {
        sum[0] += tri.center[0];
        sum[1] += tri.center[1];
        sum[2] += tri.center[2];
        count += 1.0;
    }
    if count == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [sum[0] / count, sum[1] / count, sum[2] / count]
    }
}

fn ring_angle(center: P3, point: P3) -> f32 {
    (point[1] - center[1]).atan2(point[0] - center[0])
}

fn ring_orientation(center: P3, pose: Pose) -> f32 {
    let radial = [pose.pos[0] - center[0], pose.pos[1] - center[1]];
    let tangent_ccw = [-radial[1], radial[0]];
    let (nav_heading, _) = navigation_heading(pose);
    let heading = [nav_heading.cos(), nav_heading.sin()];
    tangent_ccw[0] * heading[0] + tangent_ccw[1] * heading[1]
}

fn positive_angle_delta(angle: f32) -> f32 {
    let mut angle = normalize_angle(angle);
    if angle < 0.0 {
        angle += 2.0 * PI;
    }
    angle
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct VertexKey(i32, i32, i32);

impl VertexKey {
    fn from(p: P3) -> Self {
        Self(
            (p[0] * 10.0).round() as i32,
            (p[1] * 10.0).round() as i32,
            (p[2] * 10.0).round() as i32,
        )
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    fn mirror_x() -> [f32; 16] {
        let mut m = IDENTITY;
        m[0] = -1.0;
        m
    }

    #[test]
    fn oriented_normal_z_corrects_mirrored_winding() {
        // CCW-up triangle in the XY plane.
        let local: [P3; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(oriented_normal_z(&IDENTITY, local) > 0.99);

        let mirror = mirror_x();
        let mirrored = [
            xform(&mirror, local[0]),
            xform(&mirror, local[1]),
            xform(&mirror, local[2]),
        ];
        // The raw cross-product normal flips under the mirror...
        assert!(tri_normal(mirrored)[2] < -0.99);
        // ...but the determinant-corrected value recovers the authored facing.
        assert!(oriented_normal_z(&mirror, mirrored) > 0.99);
    }
}
