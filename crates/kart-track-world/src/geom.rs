//! Pure f32 geometry core: triangle queries, sphere/OBB sweeps, rotating
//! OBB CCD, and quaternion helpers. No IO, no platform dependencies.

pub type P3 = [f32; 3];

pub fn point_distance_2d(a: P3, b: P3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

pub fn squared_distance(a: P3, b: P3) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

pub fn closest_point_on_triangle(p: P3, tri: [P3; 3]) -> P3 {
    let [a, b, c] = tri;
    let sub = |x: P3, y: P3| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let add_scaled = |x: P3, v: P3, scale: f32| {
        [
            x[0] + v[0] * scale,
            x[1] + v[1] * scale,
            x[2] + v[2] * scale,
        ]
    };
    let dot = |x: P3, y: P3| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add_scaled(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add_scaled(a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add_scaled(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add_scaled(add_scaled(a, ab, v), ac, w)
}

/// Two-sided Moller-Trumbore intersection over the closed segment [start,end].
pub fn segment_triangle_intersection(start: P3, end: P3, tri: [P3; 3]) -> Option<(f32, P3)> {
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: P3, b: P3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let direction = sub(end, start);
    let edge1 = sub(tri[1], tri[0]);
    let edge2 = sub(tri[2], tri[0]);
    let pvec = cross(direction, edge2);
    let determinant = dot(edge1, pvec);
    if determinant.abs() <= 1.0e-7 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let tvec = sub(start, tri[0]);
    let u = dot(tvec, pvec) * inverse;
    if !(-1.0e-6..=1.0 + 1.0e-6).contains(&u) {
        return None;
    }
    let qvec = cross(tvec, edge1);
    let v = dot(direction, qvec) * inverse;
    if v < -1.0e-6 || u + v > 1.0 + 1.0e-6 {
        return None;
    }
    let fraction = dot(edge2, qvec) * inverse;
    if !(-1.0e-6..=1.0 + 1.0e-6).contains(&fraction) {
        return None;
    }
    let fraction = fraction.clamp(0.0, 1.0);
    Some((
        fraction,
        [
            start[0] + direction[0] * fraction,
            start[1] + direction[1] * fraction,
            start[2] + direction[2] * fraction,
        ],
    ))
}

#[derive(Clone, Copy, Debug)]
pub struct SphereTriangleHit {
    pub fraction: f32,
    pub center: P3,
    pub point: P3,
    pub normal: P3,
}

pub fn swept_sphere_triangle_intersection(
    start: P3,
    end: P3,
    radius: f32,
    tri: [P3; 3],
) -> Option<SphereTriangleHit> {
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: P3, b: P3| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let scale = |a: P3, s: f32| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let normalize = |a: P3| {
        let length = dot(a, a).sqrt();
        if length > 1.0e-7 {
            scale(a, 1.0 / length)
        } else {
            [0.0, 0.0, 0.0]
        }
    };
    let direction = sub(end, start);
    let center_at = |t: f32| add(start, scale(direction, t));
    let mut best: Option<SphereTriangleHit> = None;
    let mut consider = |fraction: f32, point: P3, fallback_normal: P3| {
        if !(-1.0e-6..=1.0 + 1.0e-6).contains(&fraction) {
            return;
        }
        let fraction = fraction.clamp(0.0, 1.0);
        if best.as_ref().is_some_and(|hit| fraction >= hit.fraction) {
            return;
        }
        let center = center_at(fraction);
        let delta = sub(center, point);
        let normal = if dot(delta, delta) > 1.0e-12 {
            normalize(delta)
        } else {
            fallback_normal
        };
        best = Some(SphereTriangleHit {
            fraction,
            center,
            point,
            normal,
        });
    };

    let initial_point = closest_point_on_triangle(start, tri);
    if squared_distance(start, initial_point) <= radius * radius + 1.0e-6 {
        consider(0.0, initial_point, tri_normal(tri));
        return best;
    }

    let plane_normal = tri_normal(tri);
    if dot(plane_normal, plane_normal) > 0.5 {
        let distance0 = dot(sub(start, tri[0]), plane_normal);
        let distance_delta = dot(direction, plane_normal);
        if distance_delta.abs() > 1.0e-7 {
            for sign in [-1.0f32, 1.0] {
                let signed_radius = sign * radius;
                let t = (signed_radius - distance0) / distance_delta;
                if (-1.0e-6..=1.0 + 1.0e-6).contains(&t) {
                    let center = center_at(t.clamp(0.0, 1.0));
                    let point = sub(center, scale(plane_normal, signed_radius));
                    if squared_distance(point, closest_point_on_triangle(point, tri)) <= 1.0e-6 {
                        consider(t, point, scale(plane_normal, sign));
                    }
                }
            }
        }
    }

    let roots = |a: f32, b: f32, c: f32| -> Vec<f32> {
        if a.abs() <= 1.0e-10 {
            return if b.abs() > 1.0e-10 {
                vec![-c / b]
            } else {
                Vec::new()
            };
        }
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            Vec::new()
        } else {
            let root = discriminant.max(0.0).sqrt();
            vec![(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        }
    };
    for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
        let edge = sub(b, a);
        let edge_length_sq = dot(edge, edge);
        if edge_length_sq <= 1.0e-10 {
            continue;
        }
        let offset = sub(start, a);
        let offset_perp = sub(offset, scale(edge, dot(offset, edge) / edge_length_sq));
        let direction_perp = sub(
            direction,
            scale(edge, dot(direction, edge) / edge_length_sq),
        );
        for t in roots(
            dot(direction_perp, direction_perp),
            2.0 * dot(offset_perp, direction_perp),
            dot(offset_perp, offset_perp) - radius * radius,
        ) {
            if !(-1.0e-6..=1.0 + 1.0e-6).contains(&t) {
                continue;
            }
            let center = center_at(t.clamp(0.0, 1.0));
            let edge_fraction = dot(sub(center, a), edge) / edge_length_sq;
            if (-1.0e-6..=1.0 + 1.0e-6).contains(&edge_fraction) {
                consider(
                    t,
                    add(a, scale(edge, edge_fraction.clamp(0.0, 1.0))),
                    plane_normal,
                );
            }
        }
    }
    for vertex in tri {
        let offset = sub(start, vertex);
        for t in roots(
            dot(direction, direction),
            2.0 * dot(offset, direction),
            dot(offset, offset) - radius * radius,
        ) {
            consider(t, vertex, plane_normal);
        }
    }
    best
}

#[derive(Clone, Copy, Debug)]
pub struct ObbTriangleHit {
    pub fraction: f32,
    pub center: P3,
    pub support_point: P3,
    /// SAT/CCD separating axis used to locate the contact.
    pub normal: P3,
    /// Winding-preserving authored face normal published by CollisionFace.
    pub face_normal: P3,
}

pub fn validate_obb(axes: [P3; 3], half_extents: P3) -> Result<(), String> {
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if half_extents
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("OBB half extents must be finite and non-negative".to_string());
    }
    for axis in axes {
        if (dot(axis, axis) - 1.0).abs() > 1.0e-3 {
            return Err("OBB axes must be unit length".to_string());
        }
    }
    if dot(axes[0], axes[1]).abs() > 1.0e-3
        || dot(axes[0], axes[2]).abs() > 1.0e-3
        || dot(axes[1], axes[2]).abs() > 1.0e-3
    {
        return Err("OBB axes must be orthogonal".to_string());
    }
    Ok(())
}

/// Exact CCD for a fixed-orientation OBB translated linearly over one step.
pub fn swept_obb_triangle_intersection(
    start: P3,
    end: P3,
    axes: [P3; 3],
    half_extents: P3,
    tri: [P3; 3],
) -> Option<ObbTriangleHit> {
    if validate_obb(axes, half_extents).is_err() {
        return None;
    }
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: P3, b: P3| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let scale = |a: P3, s: f32| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: P3, b: P3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let direction = sub(end, start);
    // KartRider's A5C010 first subtracts the OBB origin and transforms the
    // triangle into kart-local coordinates, then A888A0 tests it against an
    // origin-centred AABB. Preserve that float32 order for instantaneous
    // overlap; world-space projections around authored coordinates (~10^3)
    // lose several 1e-5 units through cancellation and fire one epoch early.
    if dot(direction, direction) == 0.0 {
        if !runtime_local_triangle_aabb_overlap(start, axes, half_extents, tri) {
            return None;
        }
        // A5C010 keeps the authored triangle winding for the contact normal.
        // The SAT minimum-translation axis has an arbitrary sign selected
        // from the box/triangle centres; that loses the sign on adjacent
        // wall triangles and can turn an approaching contact into a
        // separating one.  SAT remains the overlap predicate, while the
        // response normal is the winding-preserving face normal observed at
        // the runtime contact sink.
        let face = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
        let face_length_sq = dot(face, face);
        if face_length_sq <= 1.0e-12 {
            return None;
        }
        let normal = scale(face, 1.0 / face_length_sq.sqrt());
        let support_coefficient = |axis: P3, extent: f32| {
            if dot(normal, axis) >= 0.0 {
                -extent
            } else {
                extent
            }
        };
        let support_point = add(
            start,
            add(
                add(
                    scale(axes[0], support_coefficient(axes[0], half_extents[0])),
                    scale(axes[1], support_coefficient(axes[1], half_extents[1])),
                ),
                scale(axes[2], support_coefficient(axes[2], half_extents[2])),
            ),
        );
        return Some(ObbTriangleHit {
            fraction: 0.0,
            center: start,
            support_point,
            normal,
            face_normal: normal,
        });
    }
    let edges = [
        sub(tri[1], tri[0]),
        sub(tri[2], tri[1]),
        sub(tri[0], tri[2]),
    ];
    let mut candidates = vec![axes[0], axes[1], axes[2], cross(edges[0], edges[1])];
    for box_axis in axes {
        for edge in edges {
            candidates.push(cross(box_axis, edge));
        }
    }
    let mut enter = 0.0f32;
    let mut exit = 1.0f32;
    let mut enter_normal = [0.0, 0.0, 0.0];
    let mut minimum_initial_overlap = f32::INFINITY;
    let mut initial_normal = [0.0, 0.0, 0.0];
    for raw_axis in candidates {
        let length_sq = dot(raw_axis, raw_axis);
        if length_sq <= 1.0e-12 {
            continue;
        }
        let axis = scale(raw_axis, 1.0 / length_sq.sqrt());
        let center_projection = dot(start, axis);
        let velocity_projection = dot(direction, axis);
        let radius = half_extents[0] * dot(axes[0], axis).abs()
            + half_extents[1] * dot(axes[1], axis).abs()
            + half_extents[2] * dot(axes[2], axis).abs();
        let tri_projection = [dot(tri[0], axis), dot(tri[1], axis), dot(tri[2], axis)];
        let tri_min = tri_projection.into_iter().fold(f32::INFINITY, f32::min);
        let tri_max = tri_projection.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let overlap =
            (center_projection + radius).min(tri_max) - (center_projection - radius).max(tri_min);
        if overlap >= 0.0 && overlap < minimum_initial_overlap {
            minimum_initial_overlap = overlap;
            let tri_center = (tri_min + tri_max) * 0.5;
            initial_normal = scale(
                axis,
                if center_projection >= tri_center {
                    1.0
                } else {
                    -1.0
                },
            );
        }
        if velocity_projection.abs() <= 1.0e-9 {
            if center_projection + radius < tri_min || center_projection - radius > tri_max {
                return None;
            }
            continue;
        }
        let t0 = (tri_min - (center_projection + radius)) / velocity_projection;
        let t1 = (tri_max - (center_projection - radius)) / velocity_projection;
        let axis_enter = t0.min(t1);
        let axis_exit = t0.max(t1);
        if axis_enter > enter {
            enter = axis_enter;
            enter_normal = scale(axis, if velocity_projection > 0.0 { -1.0 } else { 1.0 });
        }
        exit = exit.min(axis_exit);
        if enter > exit {
            return None;
        }
    }
    if exit < 0.0 || enter > 1.0 {
        return None;
    }
    let fraction = enter.clamp(0.0, 1.0);
    let normal = if fraction == 0.0 {
        initial_normal
    } else {
        enter_normal
    };
    let center = add(start, scale(direction, fraction));
    let support_coefficient = |axis: P3, extent: f32| {
        if dot(normal, axis) >= 0.0 {
            -extent
        } else {
            extent
        }
    };
    let support_point = add(
        center,
        add(
            add(
                scale(axes[0], support_coefficient(axes[0], half_extents[0])),
                scale(axes[1], support_coefficient(axes[1], half_extents[1])),
            ),
            scale(axes[2], support_coefficient(axes[2], half_extents[2])),
        ),
    );
    Some(ObbTriangleHit {
        fraction,
        center,
        support_point,
        normal,
        face_normal: tri_normal(tri),
    })
}

#[derive(Clone, Copy)]
pub struct ObbSatState {
    pub separation: f32,
    pub normal: P3,
}

pub fn runtime_local_triangle_aabb_overlap(
    center: P3,
    axes: [P3; 3],
    half_extents: P3,
    tri: [P3; 3],
) -> bool {
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: P3, b: P3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let vertices = tri.map(|point| {
        let relative = sub(point, center);
        [
            dot(relative, axes[0]),
            dot(relative, axes[1]),
            dot(relative, axes[2]),
        ]
    });
    let edges = [
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[1]),
        sub(vertices[0], vertices[2]),
    ];
    for edge in edges {
        // A888A0 keeps these axes unnormalised. This comparison order avoids
        // the sqrt/divide rounding that moved authored wall contacts by an
        // entire 2 ms epoch in the former world-space SAT.
        let cross_axes = [
            [0.0, -edge[2], edge[1]],
            [edge[2], 0.0, -edge[0]],
            [-edge[1], edge[0], 0.0],
        ];
        for axis in cross_axes {
            let projections = [
                dot(vertices[0], axis),
                dot(vertices[1], axis),
                dot(vertices[2], axis),
            ];
            let min = projections.into_iter().fold(f32::INFINITY, f32::min);
            let max = projections.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let radius = half_extents[0] * axis[0].abs()
                + half_extents[1] * axis[1].abs()
                + half_extents[2] * axis[2].abs();
            if min > radius || -radius > max {
                return false;
            }
        }
    }
    for axis in 0..3 {
        let min = vertices
            .iter()
            .map(|v| v[axis])
            .fold(f32::INFINITY, f32::min);
        let max = vertices
            .iter()
            .map(|v| v[axis])
            .fold(f32::NEG_INFINITY, f32::max);
        if min > half_extents[axis] || -half_extents[axis] > max {
            return false;
        }
    }
    let normal = cross(edges[0], edges[1]);
    let plane = dot(normal, vertices[0]);
    let radius = half_extents[0] * normal[0].abs()
        + half_extents[1] * normal[1].abs()
        + half_extents[2] * normal[2].abs();
    plane.abs() <= radius
}

pub fn obb_triangle_sat_state(
    center: P3,
    axes: [P3; 3],
    half_extents: P3,
    tri: [P3; 3],
) -> ObbSatState {
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let scale = |a: P3, s: f32| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: P3, b: P3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let local_tri = tri.map(|point| {
        let relative = sub(point, center);
        [
            dot(relative, axes[0]),
            dot(relative, axes[1]),
            dot(relative, axes[2]),
        ]
    });
    let local_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let edges = [
        sub(local_tri[1], local_tri[0]),
        sub(local_tri[2], local_tri[1]),
        sub(local_tri[0], local_tri[2]),
    ];
    let mut candidates = vec![
        local_axes[0],
        local_axes[1],
        local_axes[2],
        cross(edges[0], edges[1]),
    ];
    for box_axis in local_axes {
        for edge in edges {
            candidates.push(cross(box_axis, edge));
        }
    }
    let mut maximum_gap = 0.0f32;
    let mut separating_normal = [0.0, 0.0, 0.0];
    let mut minimum_overlap = f32::INFINITY;
    let mut overlap_normal = [0.0, 0.0, 0.0];
    for raw_axis in candidates {
        let length_sq = dot(raw_axis, raw_axis);
        if length_sq <= 1.0e-12 {
            continue;
        }
        let axis = scale(raw_axis, 1.0 / length_sq.sqrt());
        let center_projection = 0.0;
        let radius = half_extents[0] * axis[0].abs()
            + half_extents[1] * axis[1].abs()
            + half_extents[2] * axis[2].abs();
        let projections = [
            dot(local_tri[0], axis),
            dot(local_tri[1], axis),
            dot(local_tri[2], axis),
        ];
        let tri_min = projections.into_iter().fold(f32::INFINITY, f32::min);
        let tri_max = projections.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let box_min = center_projection - radius;
        let box_max = center_projection + radius;
        let gap = (tri_min - box_max).max(box_min - tri_max);
        let tri_center = (tri_min + tri_max) * 0.5;
        let local_normal = scale(
            axis,
            if center_projection >= tri_center {
                1.0
            } else {
                -1.0
            },
        );
        let normal = [
            axes[0][0] * local_normal[0]
                + axes[1][0] * local_normal[1]
                + axes[2][0] * local_normal[2],
            axes[0][1] * local_normal[0]
                + axes[1][1] * local_normal[1]
                + axes[2][1] * local_normal[2],
            axes[0][2] * local_normal[0]
                + axes[1][2] * local_normal[1]
                + axes[2][2] * local_normal[2],
        ];
        if gap > maximum_gap {
            maximum_gap = gap;
            separating_normal = normal;
        }
        let overlap = box_max.min(tri_max) - box_min.max(tri_min);
        if overlap >= 0.0 && overlap < minimum_overlap {
            minimum_overlap = overlap;
            overlap_normal = normal;
        }
    }
    if maximum_gap > 0.0 {
        ObbSatState {
            separation: maximum_gap,
            normal: separating_normal,
        }
    } else {
        ObbSatState {
            separation: -minimum_overlap.max(0.0),
            normal: overlap_normal,
        }
    }
}

pub type Quat = [f32; 4];

pub fn normalized_matrix_axes(matrix: [f32; 16]) -> Result<[P3; 3], String> {
    let mut axes = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]],
    ];
    for axis in &mut axes {
        let length = axis.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !length.is_finite() || length <= 1.0e-9 {
            return Err("dynamic object world transform has a degenerate axis".to_string());
        }
        for value in axis {
            *value /= length;
        }
    }
    Ok(axes)
}

pub fn angular_velocity_between(q0: Quat, mut q1: Quat, dt: f32) -> P3 {
    if q0.iter().zip(q1).map(|(a, b)| a * b).sum::<f32>() < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
    }
    let conjugate_q0 = [q0[0], -q0[1], -q0[2], -q0[3]];
    let [aw, ax, ay, az] = q1;
    let [bw, bx, by, bz] = conjugate_q0;
    let mut delta = [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ];
    if delta[0] < 0.0 {
        delta = [-delta[0], -delta[1], -delta[2], -delta[3]];
    }
    let half_angle = delta[0].clamp(-1.0, 1.0).acos();
    let sine = half_angle.sin();
    if sine.abs() <= 1.0e-7 || dt <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let factor = 2.0 * half_angle / (sine * dt);
    [delta[1] * factor, delta[2] * factor, delta[3] * factor]
}

pub fn axes_to_quat(axes: [P3; 3]) -> Quat {
    let m00 = axes[0][0];
    let m01 = axes[1][0];
    let m02 = axes[2][0];
    let m10 = axes[0][1];
    let m11 = axes[1][1];
    let m12 = axes[2][1];
    let m20 = axes[0][2];
    let m21 = axes[1][2];
    let m22 = axes[2][2];
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    };
    let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

pub fn quat_slerp(mut a: Quat, mut b: Quat, t: f32) -> Quat {
    let mut cosine = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if cosine < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        cosine = -cosine;
    }
    if cosine > 0.9995 {
        let mut q = [0.0; 4];
        for i in 0..4 {
            q[i] = a[i] + (b[i] - a[i]) * t;
        }
        let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        for value in &mut q {
            *value /= norm;
        }
        return q;
    }
    cosine = cosine.clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let denominator = angle.sin();
    let wa = ((1.0 - t) * angle).sin() / denominator;
    let wb = (t * angle).sin() / denominator;
    for i in 0..4 {
        a[i] = a[i] * wa + b[i] * wb;
    }
    a
}

pub fn quat_to_axes(q: Quat) -> [P3; 3] {
    let [w, x, y, z] = q;
    let m00 = 1.0 - 2.0 * (y * y + z * z);
    let m01 = 2.0 * (x * y - z * w);
    let m02 = 2.0 * (x * z + y * w);
    let m10 = 2.0 * (x * y + z * w);
    let m11 = 1.0 - 2.0 * (x * x + z * z);
    let m12 = 2.0 * (y * z - x * w);
    let m20 = 2.0 * (x * z - y * w);
    let m21 = 2.0 * (y * z + x * w);
    let m22 = 1.0 - 2.0 * (x * x + y * y);
    [[m00, m10, m20], [m01, m11, m21], [m02, m12, m22]]
}

/// Conservative continuous collision detection for linearly translating and
/// shortest-arc rotating OBBs. The SAT separation is advanced by a bound on
/// point motion (linear travel + angular travel * box radius), so rotation-only
/// contacts cannot fall between temporal samples.
pub fn swept_rotating_obb_triangle_intersection(
    start: P3,
    end: P3,
    start_axes: [P3; 3],
    end_axes: [P3; 3],
    half_extents: P3,
    tri: [P3; 3],
) -> Option<ObbTriangleHit> {
    if validate_obb(start_axes, half_extents).is_err()
        || validate_obb(end_axes, half_extents).is_err()
    {
        return None;
    }
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: P3, b: P3| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let scale = |a: P3, s: f32| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let direction = sub(end, start);
    let q0 = axes_to_quat(start_axes);
    let mut q1 = axes_to_quat(end_axes);
    let mut qdot = q0.iter().zip(q1).map(|(a, b)| a * b).sum::<f32>();
    if qdot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        qdot = -qdot;
    }
    let angle = 2.0 * qdot.clamp(-1.0, 1.0).acos();
    if angle <= 1.0e-7 {
        return swept_obb_triangle_intersection(start, end, start_axes, half_extents, tri);
    }
    let linear_travel = dot(direction, direction).sqrt();
    let box_radius = half_extents.iter().map(|v| v * v).sum::<f32>().sqrt();
    let motion_bound = linear_travel + angle * box_radius;
    if motion_bound <= 1.0e-12 {
        return None;
    }
    let center_at = |t: f32| add(start, scale(direction, t));
    let axes_at = |t: f32| quat_to_axes(quat_slerp(q0, q1, t));
    const CONTACT_TOLERANCE: f32 = 1.0e-6;
    let mut t = 0.0f32;
    let mut lower = 0.0f32;
    for _ in 0..256 {
        let state = obb_triangle_sat_state(center_at(t), axes_at(t), half_extents, tri);
        if state.separation <= CONTACT_TOLERANCE {
            if t > 0.0 {
                let mut lo = lower;
                let mut hi = t;
                for _ in 0..28 {
                    let mid = (lo + hi) * 0.5;
                    if obb_triangle_sat_state(center_at(mid), axes_at(mid), half_extents, tri)
                        .separation
                        <= CONTACT_TOLERANCE
                    {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                t = hi;
            }
            let center = center_at(t);
            let axes = axes_at(t);
            let normal = obb_triangle_sat_state(center, axes, half_extents, tri).normal;
            let support_coefficient = |axis: P3, extent: f32| {
                if dot(normal, axis) >= 0.0 {
                    -extent
                } else {
                    extent
                }
            };
            let support_point = add(
                center,
                add(
                    add(
                        scale(axes[0], support_coefficient(axes[0], half_extents[0])),
                        scale(axes[1], support_coefficient(axes[1], half_extents[1])),
                    ),
                    scale(axes[2], support_coefficient(axes[2], half_extents[2])),
                ),
            );
            return Some(ObbTriangleHit {
                fraction: t,
                center,
                support_point,
                normal,
                face_normal: tri_normal(tri),
            });
        }
        lower = t;
        t += state.separation / motion_bound;
        if t > 1.0 + 1.0e-7 {
            return None;
        }
        t = t.min(1.0);
    }
    None
}

pub fn point_distance_weighted(a: P3, b: P3) -> f32 {
    let horizontal = point_distance_2d(a, b);
    horizontal + (a[2] - b[2]).abs() * 4.0
}

pub fn geom_triangles(g: &kart_fmt::kart_object::Geom) -> Vec<[u32; 3]> {
    if g.strip {
        (0..g.indices.len().saturating_sub(2))
            .filter_map(|k| {
                let (a, b, c) = (g.indices[k], g.indices[k + 1], g.indices[k + 2]);
                if a == b || b == c || a == c {
                    None
                } else if k % 2 == 0 {
                    Some([a, b, c])
                } else {
                    Some([b, a, c])
                }
            })
            .collect()
    } else {
        g.indices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }
}

pub fn xform(m: &[f32; 16], p: P3) -> P3 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

pub fn tri_center(tri: [P3; 3]) -> P3 {
    [
        (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
        (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
        (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
    ]
}

pub fn tri_normal(tri: [P3; 3]) -> P3 {
    let ux = tri[1][0] - tri[0][0];
    let uy = tri[1][1] - tri[0][1];
    let uz = tri[1][2] - tri[0][2];
    let vx = tri[2][0] - tri[0][0];
    let vy = tri[2][1] - tri[0][1];
    let vz = tri[2][2] - tri[0][2];
    let n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

pub fn shortest_directed_triangle_edge(tri: [P3; 3]) -> P3 {
    let edges = [
        [
            tri[1][0] - tri[0][0],
            tri[1][1] - tri[0][1],
            tri[1][2] - tri[0][2],
        ],
        [
            tri[2][0] - tri[0][0],
            tri[2][1] - tri[0][1],
            tri[2][2] - tri[0][2],
        ],
        [
            tri[2][0] - tri[1][0],
            tri[2][1] - tri[1][1],
            tri[2][2] - tri[1][2],
        ],
    ];
    let mut selected = edges[0];
    let mut selected_squared =
        selected[0] * selected[0] + selected[1] * selected[1] + selected[2] * selected[2];
    for edge in edges.into_iter().skip(1) {
        let squared = edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2];
        if squared < selected_squared {
            selected = edge;
            selected_squared = squared;
        }
    }
    if selected_squared <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        let length = selected_squared.sqrt();
        [
            selected[0] / length,
            selected[1] / length,
            selected[2] / length,
        ]
    }
}
