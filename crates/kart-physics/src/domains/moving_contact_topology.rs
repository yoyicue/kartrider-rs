//! Port of `scripts/evaluate_moving_contact_topology.py` — rigid
//! moving-reference-frame kinematics for dynamic contact ownership.
//!
//! Numeric discipline: the Python module is pure f64 (no `_f32` sites); every
//! dot product is a CPython `sum()` over three products, so all of them route
//! through [`dot3_sum`] / [`sum_f64`] (Neumaier, see crate::num). The only
//! mode-dependent numerics live inside the two
//! [`evaluate_orientation_integrator`] calls (`runtime_float32_exact`).
//! Component-count validation is type-enforced by the fixed-size arguments.

use crate::domains::orientation_integrator::{
    evaluate_orientation_integrator, OrientationIntegratorResult,
};
use crate::num::{dot3_sum, matrix_columns, sum_f64};
use crate::state::MovingContactAnchor;

/// `_local_to_world`: `sum(columns[col][axis] * vector[col] for col in
/// range(3))` per output axis (CPython `sum()` semantics).
fn local_to_world(matrix: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    let columns = matrix_columns(matrix);
    let mut out = [0.0; 3];
    for (axis, slot) in out.iter_mut().enumerate() {
        *slot = sum_f64(&[
            columns[0][axis] * vector[0],
            columns[1][axis] * vector[1],
            columns[2][axis] * vector[2],
        ]);
    }
    out
}

/// `_world_to_local`: per column `sum(column[axis] * vector[axis])`.
fn world_to_local(matrix: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    let columns = matrix_columns(matrix);
    [
        dot3_sum(columns[0], vector),
        dot3_sum(columns[1], vector),
        dot3_sum(columns[2], vector),
    ]
}

/// `_compose(left, right)`: column-major composition flattened row-major.
fn compose(left: [f64; 9], right: [f64; 9]) -> [f64; 9] {
    let right_columns = matrix_columns(right);
    let output_columns = [
        local_to_world(left, right_columns[0]),
        local_to_world(left, right_columns[1]),
        local_to_world(left, right_columns[2]),
    ];
    let mut out = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            out[row * 3 + column] = output_columns[column][row];
        }
    }
    out
}

/// `_relative_matrix(parent, child)`: child columns expressed in the parent
/// frame, flattened row-major.
fn relative_matrix(parent: [f64; 9], child: [f64; 9]) -> [f64; 9] {
    let child_columns = matrix_columns(child);
    let output_columns = [
        world_to_local(parent, child_columns[0]),
        world_to_local(parent, child_columns[1]),
        world_to_local(parent, child_columns[2]),
    ];
    let mut out = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            out[row * 3 + column] = output_columns[column][row];
        }
    }
    out
}

/// `_cross` (plain f64, fixed operand order).
fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

/// `_normalize`: `math.sqrt(sum(c * c))` (Neumaier square-sum) with the
/// exactly-zero guard.
fn normalize(value: [f64; 3]) -> Result<[f64; 3], String> {
    let magnitude = sum_f64(&[
        value[0] * value[0],
        value[1] * value[1],
        value[2] * value[2],
    ])
    .sqrt();
    if magnitude == 0.0 {
        return Err("contact normal must be nonzero".to_string());
    }
    Ok([
        value[0] / magnitude,
        value[1] / magnitude,
        value[2] / magnitude,
    ])
}

/// `_project_tangent`.
fn project_tangent(value: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    let normal_component = dot3_sum(value, normal);
    [
        value[0] - normal[0] * normal_component,
        value[1] - normal[1] * normal_component,
        value[2] - normal[2] * normal_component,
    ]
}

/// CPython `repr`/`str` of a finite-or-not f64, for error messages that
/// embed a float (Python f-string interpolation). Rust `{:?}` produces the
/// same shortest round-trip digits but formats exponents differently
/// (`1e-5` vs Python `1e-05`, `1e16` vs `1e+16`), so the notation is
/// re-selected with CPython's rules: fixed for decimal exponent in
/// [-4, 16), scientific with a signed two-digit exponent otherwise.
pub(crate) fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let rendered = format!("{value:e}");
    let (mantissa, exponent) = rendered
        .split_once('e')
        .expect("LowerExp always contains 'e'");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let negative = mantissa.starts_with('-');
    let sign = if negative { "-" } else { "" };
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let count = digits.len() as i32;
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point >= count {
            format!("{}{}.0", digits, "0".repeat((point - count) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// `acquire_moving_contact_frame(...)` result dict
/// (evaluate_moving_contact_topology.py:153-175).
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactFrame {
    pub object_id: String,
    pub local_position: [f64; 3],
    pub local_velocity: [f64; 3],
    pub local_contact_normal: [f64; 3],
    pub local_plane_offset: f64,
    pub discarded_normal_velocity: f64,
    pub relative_matrix: [f64; 9],
    pub local_angular_velocity: [f64; 3],
    pub object_point_velocity: [f64; 3],
    pub kinematic_contract: &'static str,
}

impl MovingContactFrame {
    /// The frame as the storable `moving_contact_anchor` latch (the
    /// producer's full key set, including the evidence fields).
    pub fn anchor(&self) -> MovingContactAnchor {
        MovingContactAnchor {
            object_id: self.object_id.clone(),
            local_position: self.local_position,
            local_velocity: self.local_velocity,
            relative_matrix: self.relative_matrix,
            local_angular_velocity: self.local_angular_velocity,
            local_contact_normal: self.local_contact_normal,
            local_plane_offset: self.local_plane_offset,
            discarded_normal_velocity: Some(self.discarded_normal_velocity),
            object_point_velocity: Some(self.object_point_velocity),
            kinematic_contract: Some(self.kinematic_contract.to_string()),
        }
    }
}

/// `acquire_moving_contact_frame(...)`: express the complete vehicle state
/// in a moving object's local frame.
#[allow(clippy::too_many_arguments)]
pub fn acquire_moving_contact_frame(
    vehicle_position: [f64; 3],
    vehicle_velocity: [f64; 3],
    vehicle_matrix: [f64; 9],
    vehicle_angular_velocity: [f64; 3],
    object_id: &str,
    object_position: [f64; 3],
    object_velocity: [f64; 3],
    object_matrix: [f64; 9],
    object_angular_velocity: [f64; 3],
    contact_normal_world: [f64; 3],
    separation_tolerance: f64,
) -> Result<MovingContactFrame, String> {
    if object_id.is_empty() {
        return Err("object_id is required".to_string());
    }
    let offset_world = [
        vehicle_position[0] - object_position[0],
        vehicle_position[1] - object_position[1],
        vehicle_position[2] - object_position[2],
    ];
    let rotational = cross(object_angular_velocity, offset_world);
    let point_velocity = [
        object_velocity[0] + rotational[0],
        object_velocity[1] + rotational[1],
        object_velocity[2] + rotational[2],
    ];
    let local_position = world_to_local(object_matrix, offset_world);
    let local_contact_normal =
        normalize(world_to_local(object_matrix, contact_normal_world))?;
    let unconstrained_local_velocity = world_to_local(
        object_matrix,
        [
            vehicle_velocity[0] - point_velocity[0],
            vehicle_velocity[1] - point_velocity[1],
            vehicle_velocity[2] - point_velocity[2],
        ],
    );
    let relative_normal_velocity = dot3_sum(
        unconstrained_local_velocity,
        local_contact_normal,
    );
    if relative_normal_velocity > separation_tolerance {
        return Err(format!(
            "cannot acquire separating moving contact (relative normal velocity {})",
            py_float_repr(relative_normal_velocity),
        ));
    }
    Ok(MovingContactFrame {
        object_id: object_id.to_string(),
        local_position,
        local_velocity: project_tangent(
            unconstrained_local_velocity,
            local_contact_normal,
        ),
        local_contact_normal,
        local_plane_offset: dot3_sum(local_position, local_contact_normal),
        discarded_normal_velocity: relative_normal_velocity,
        relative_matrix: relative_matrix(object_matrix, vehicle_matrix),
        local_angular_velocity: world_to_local(
            object_matrix,
            [
                vehicle_angular_velocity[0] - object_angular_velocity[0],
                vehicle_angular_velocity[1] - object_angular_velocity[1],
                vehicle_angular_velocity[2] - object_angular_velocity[2],
            ],
        ),
        object_point_velocity: point_velocity,
        kinematic_contract:
            "v_vehicle = v_object + omega_object x r + R_object * v_relative_local",
    })
}

/// `evaluate_moving_contact_selector(...)` result. Key presence follows the
/// Python branch: the early identity/geometry returns carry only `sustain`,
/// `reason` and `relative_normal_velocity`; the manifold-mismatch return adds
/// `normal_alignment`, `normal_alignment_tolerance` and `authority`; the full
/// evaluation adds every field.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactSelectorResult {
    pub sustain: bool,
    pub reason: String,
    pub relative_normal_velocity: Option<f64>,
    pub separation_tolerance: Option<f64>,
    pub normal_alignment: Option<f64>,
    pub normal_alignment_tolerance: Option<f64>,
    pub object_point_velocity: Option<[f64; 3]>,
    pub authority: Option<&'static str>,
}

/// `evaluate_moving_contact_selector(...)`: select sustain/release from
/// current geometry and relative normal speed.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_moving_contact_selector(
    anchor: &MovingContactAnchor,
    vehicle_position: [f64; 3],
    vehicle_velocity: [f64; 3],
    object_id: &str,
    object_position: [f64; 3],
    object_velocity: [f64; 3],
    object_matrix: [f64; 9],
    object_angular_velocity: [f64; 3],
    contact_normal_world: [f64; 3],
    contact_active: bool,
    separation_tolerance: f64,
    normal_alignment_tolerance: f64,
) -> Result<MovingContactSelectorResult, String> {
    if anchor.object_id != object_id {
        return Ok(MovingContactSelectorResult {
            sustain: false,
            reason: "object_identity_changed".to_string(),
            relative_normal_velocity: None,
            separation_tolerance: None,
            normal_alignment: None,
            normal_alignment_tolerance: None,
            object_point_velocity: None,
            authority: None,
        });
    }
    if !contact_active {
        return Ok(MovingContactSelectorResult {
            sustain: false,
            reason: "contact_geometry_released".to_string(),
            relative_normal_velocity: None,
            separation_tolerance: None,
            normal_alignment: None,
            normal_alignment_tolerance: None,
            object_point_velocity: None,
            authority: None,
        });
    }
    if separation_tolerance < 0.0 {
        return Err("separation_tolerance must be nonnegative".to_string());
    }
    if !(0.0..1.0).contains(&normal_alignment_tolerance) {
        return Err("normal_alignment_tolerance must be in [0, 1)".to_string());
    }
    let stored_normal = normalize(anchor.local_contact_normal)?;
    let current_normal =
        normalize(world_to_local(object_matrix, contact_normal_world))?;
    let normal_alignment = dot3_sum(stored_normal, current_normal);
    if normal_alignment < 1.0 - normal_alignment_tolerance {
        return Ok(MovingContactSelectorResult {
            sustain: false,
            reason: "contact_manifold_changed".to_string(),
            relative_normal_velocity: None,
            separation_tolerance: None,
            normal_alignment: Some(normal_alignment),
            normal_alignment_tolerance: Some(normal_alignment_tolerance),
            object_point_velocity: None,
            authority: Some("current_contact_geometry_and_relative_rigid_body_state"),
        });
    }
    let offset_world = [
        vehicle_position[0] - object_position[0],
        vehicle_position[1] - object_position[1],
        vehicle_position[2] - object_position[2],
    ];
    let rotational = cross(object_angular_velocity, offset_world);
    let point_velocity = [
        object_velocity[0] + rotational[0],
        object_velocity[1] + rotational[1],
        object_velocity[2] + rotational[2],
    ];
    let relative_velocity_local = world_to_local(
        object_matrix,
        [
            vehicle_velocity[0] - point_velocity[0],
            vehicle_velocity[1] - point_velocity[1],
            vehicle_velocity[2] - point_velocity[2],
        ],
    );
    let relative_normal_velocity = dot3_sum(relative_velocity_local, stored_normal);
    let sustain = relative_normal_velocity <= separation_tolerance;
    Ok(MovingContactSelectorResult {
        sustain,
        reason: if sustain {
            "contact_active_nonseparating".to_string()
        } else {
            "relative_normal_separation".to_string()
        },
        relative_normal_velocity: Some(relative_normal_velocity),
        separation_tolerance: Some(separation_tolerance),
        normal_alignment: Some(normal_alignment),
        normal_alignment_tolerance: Some(normal_alignment_tolerance),
        object_point_velocity: Some(point_velocity),
        authority: Some("current_contact_geometry_and_relative_rigid_body_state"),
    })
}

/// `evaluate_moving_contact_step(...)` result dict
/// (evaluate_moving_contact_topology.py:389-411).
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactStep {
    pub position_after: [f64; 3],
    pub velocity_after: [f64; 3],
    pub matrix_after: [f64; 9],
    pub angular_velocity_after: [f64; 3],
    pub anchor_after: MovingContactAnchor,
    pub relative_orientation_integrator: OrientationIntegratorResult,
    /// `None` on the `"end"` phase (Python `object_orientation = None`).
    pub object_orientation_integrator: Option<OrientationIntegratorResult>,
    pub object_position_after: [f64; 3],
    pub object_matrix_after: [f64; 9],
    pub object_state_phase: String,
    pub object_point_velocity: [f64; 3],
    pub reference_frame_owner: String,
    pub contact_plane_error_before_projection: f64,
    pub closure_status: &'static str,
}

/// `evaluate_moving_contact_step(...)`: advance relative motion and
/// reconstruct the world rigid-body state. `"start"` advances the supplied
/// dynamic-body pose through `dt` first; `"end"` accepts an already advanced
/// authoritative pose.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_moving_contact_step(
    anchor: &MovingContactAnchor,
    object_id: &str,
    object_position: [f64; 3],
    object_velocity: [f64; 3],
    object_matrix: [f64; 9],
    object_angular_velocity: [f64; 3],
    dt: f64,
    numeric_mode: &str,
    object_state_phase: &str,
) -> Result<MovingContactStep, String> {
    if anchor.object_id != object_id {
        return Err("moving-contact object identity changed".to_string());
    }
    if dt <= 0.0 {
        return Err("dt must be positive".to_string());
    }
    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    if object_state_phase != "start" && object_state_phase != "end" {
        return Err("object_state_phase must be start or end".to_string());
    }
    let (object_position_after, object_matrix_after, object_orientation) =
        if object_state_phase == "start" {
            let position_after = [
                object_position[0] + object_velocity[0] * dt,
                object_position[1] + object_velocity[1] * dt,
                object_position[2] + object_velocity[2] * dt,
            ];
            let object_angular_body =
                world_to_local(object_matrix, object_angular_velocity);
            let orientation = evaluate_orientation_integrator(
                object_matrix,
                object_angular_body,
                dt,
                false,
                true,
                0,
                numeric_mode == "float32",
            )?;
            (position_after, orientation.matrix_9c_after, Some(orientation))
        } else {
            (object_position, object_matrix, None)
        };

    let normal = normalize(anchor.local_contact_normal)?;
    let tangent_velocity = project_tangent(anchor.local_velocity, normal);
    let unconstrained_position = [
        anchor.local_position[0] + tangent_velocity[0] * dt,
        anchor.local_position[1] + tangent_velocity[1] * dt,
        anchor.local_position[2] + tangent_velocity[2] * dt,
    ];
    let plane_error =
        anchor.local_plane_offset - dot3_sum(unconstrained_position, normal);
    let local_position = [
        unconstrained_position[0] + normal[0] * plane_error,
        unconstrained_position[1] + normal[1] * plane_error,
        unconstrained_position[2] + normal[2] * plane_error,
    ];
    let offset_world = local_to_world(object_matrix_after, local_position);
    let relative_velocity_world =
        local_to_world(object_matrix_after, tangent_velocity);
    let point_velocity = cross(object_angular_velocity, offset_world);
    let velocity = [
        object_velocity[0] + point_velocity[0] + relative_velocity_world[0],
        object_velocity[1] + point_velocity[1] + relative_velocity_world[1],
        object_velocity[2] + point_velocity[2] + relative_velocity_world[2],
    ];
    let relative_angular_world =
        local_to_world(object_matrix_after, anchor.local_angular_velocity);
    // local_angular_velocity is expressed in the platform frame (a spatial
    // angular vector). B2FD80 consumes body-local angular velocity, so
    // convert through the current relative orientation before integrating
    // R_relative.
    let relative_angular_body =
        world_to_local(anchor.relative_matrix, anchor.local_angular_velocity);
    let relative_orientation = evaluate_orientation_integrator(
        anchor.relative_matrix,
        relative_angular_body,
        dt,
        false,
        true,
        0,
        numeric_mode == "float32",
    )?;
    let relative_matrix_after = relative_orientation.matrix_9c_after;
    let mut anchor_after = anchor.clone();
    anchor_after.local_position = local_position;
    anchor_after.local_velocity = tangent_velocity;
    anchor_after.relative_matrix = relative_matrix_after;
    Ok(MovingContactStep {
        position_after: [
            object_position_after[0] + offset_world[0],
            object_position_after[1] + offset_world[1],
            object_position_after[2] + offset_world[2],
        ],
        velocity_after: velocity,
        matrix_after: compose(object_matrix_after, relative_matrix_after),
        angular_velocity_after: [
            object_angular_velocity[0] + relative_angular_world[0],
            object_angular_velocity[1] + relative_angular_world[1],
            object_angular_velocity[2] + relative_angular_world[2],
        ],
        anchor_after,
        relative_orientation_integrator: relative_orientation,
        object_orientation_integrator: object_orientation,
        object_position_after,
        object_matrix_after,
        object_state_phase: object_state_phase.to_string(),
        object_point_velocity: [
            object_velocity[0] + point_velocity[0],
            object_velocity[1] + point_velocity[1],
            object_velocity[2] + point_velocity[2],
        ],
        reference_frame_owner: object_id.to_string(),
        contact_plane_error_before_projection: plane_error,
        closure_status: "generic_rigid_moving_frame_contact_manifold_executable",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn still_frame() -> MovingContactFrame {
        // Vehicle and object share the identity orientation; the vehicle
        // rests on the object at +0.5 z with the world-up contact normal.
        acquire_moving_contact_frame(
            [0.25, -0.5, 0.5],
            [1.0, 0.0, 0.0],
            IDENTITY,
            [0.0, 0.0, 0.0],
            "roadobj:7:node3",
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            IDENTITY,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1e-6,
        )
        .unwrap()
    }

    #[test]
    fn py_float_repr_matches_cpython() {
        // Constants are CPython 3.14.6 repr()/str() outputs.
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(-0.0), "-0.0");
        assert_eq!(py_float_repr(0.0), "0.0");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1234567890123456.0), "1234567890123456.0");
        assert_eq!(py_float_repr(-2.5e-7), "-2.5e-07");
        assert_eq!(py_float_repr(100.0), "100.0");
        assert_eq!(py_float_repr(f64::NAN), "nan");
        assert_eq!(py_float_repr(f64::INFINITY), "inf");
        assert_eq!(py_float_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(py_float_repr(1.5e16), "1.5e+16");
    }

    #[test]
    fn acquire_requires_object_id() {
        assert_eq!(
            acquire_moving_contact_frame(
                [0.0; 3], [0.0; 3], IDENTITY, [0.0; 3], "", [0.0; 3], [0.0; 3],
                IDENTITY, [0.0; 3], [0.0, 0.0, 1.0], 1e-6,
            )
            .unwrap_err(),
            "object_id is required"
        );
    }

    #[test]
    fn acquire_rejects_separating_contact() {
        let error = acquire_moving_contact_frame(
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.25],
            IDENTITY,
            [0.0; 3],
            "roadobj:7:node3",
            [0.0; 3],
            [0.0; 3],
            IDENTITY,
            [0.0; 3],
            [0.0, 0.0, 1.0],
            1e-6,
        )
        .unwrap_err();
        // Python: f"cannot acquire separating moving contact (relative
        // normal velocity {relative_normal_velocity})".
        assert_eq!(
            error,
            "cannot acquire separating moving contact (relative normal velocity 0.25)"
        );
    }

    #[test]
    fn acquire_frame_translation_only() {
        // Constants computed with the Python authority
        // (evaluate_moving_contact_topology.acquire_moving_contact_frame).
        let frame = still_frame();
        assert_eq!(frame.local_position, [0.25, -0.5, 0.5]);
        assert_eq!(frame.local_velocity, [0.0, 0.0, 0.0]);
        assert_eq!(frame.local_contact_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.local_plane_offset, 0.5);
        assert_eq!(frame.discarded_normal_velocity, 0.0);
        assert_eq!(frame.relative_matrix, IDENTITY);
        assert_eq!(frame.object_point_velocity, [1.0, 0.0, 0.0]);
        assert_eq!(
            frame.kinematic_contract,
            "v_vehicle = v_object + omega_object x r + R_object * v_relative_local"
        );
    }

    #[test]
    fn selector_sustains_co_moving_contact() {
        let anchor = still_frame().anchor();
        let result = evaluate_moving_contact_selector(
            &anchor,
            [0.25, -0.5, 0.5],
            [1.0, 0.0, 0.0],
            "roadobj:7:node3",
            [0.0; 3],
            [1.0, 0.0, 0.0],
            IDENTITY,
            [0.0; 3],
            [0.0, 0.0, 1.0],
            true,
            1e-6,
            1e-5,
        )
        .unwrap();
        assert!(result.sustain);
        assert_eq!(result.reason, "contact_active_nonseparating");
        assert_eq!(result.relative_normal_velocity, Some(0.0));
        assert_eq!(result.object_point_velocity, Some([1.0, 0.0, 0.0]));
        assert_eq!(
            result.authority,
            Some("current_contact_geometry_and_relative_rigid_body_state")
        );
    }

    #[test]
    fn selector_early_returns_skip_validation() {
        let anchor = still_frame().anchor();
        // Identity change wins over every later check (even invalid
        // tolerances).
        let result = evaluate_moving_contact_selector(
            &anchor,
            [0.0; 3],
            [0.0; 3],
            "roadobj:other",
            [0.0; 3],
            [0.0; 3],
            IDENTITY,
            [0.0; 3],
            [0.0; 3],
            false,
            -1.0,
            2.0,
        )
        .unwrap();
        assert!(!result.sustain);
        assert_eq!(result.reason, "object_identity_changed");
        assert_eq!(result.relative_normal_velocity, None);
        assert_eq!(result.authority, None);
        // Inactive geometry wins over the tolerance validation.
        let result = evaluate_moving_contact_selector(
            &anchor,
            [0.0; 3],
            [0.0; 3],
            "roadobj:7:node3",
            [0.0; 3],
            [0.0; 3],
            IDENTITY,
            [0.0; 3],
            [0.0; 3],
            false,
            -1.0,
            2.0,
        )
        .unwrap();
        assert_eq!(result.reason, "contact_geometry_released");
    }

    #[test]
    fn selector_validates_tolerances_in_order() {
        let anchor = still_frame().anchor();
        let error = evaluate_moving_contact_selector(
            &anchor,
            [0.0; 3],
            [0.0; 3],
            "roadobj:7:node3",
            [0.0; 3],
            [0.0; 3],
            IDENTITY,
            [0.0; 3],
            [0.0, 0.0, 1.0],
            true,
            -1.0,
            0.5,
        )
        .unwrap_err();
        assert_eq!(error, "separation_tolerance must be nonnegative");
        let error = evaluate_moving_contact_selector(
            &anchor,
            [0.0; 3],
            [0.0; 3],
            "roadobj:7:node3",
            [0.0; 3],
            [0.0; 3],
            IDENTITY,
            [0.0; 3],
            [0.0, 0.0, 1.0],
            true,
            0.0,
            1.0,
        )
        .unwrap_err();
        assert_eq!(error, "normal_alignment_tolerance must be in [0, 1)");
    }

    #[test]
    fn step_end_phase_translation() {
        // Object moving at +1 x for 2 ms; the vehicle rides along. Constants
        // computed with the Python authority (evaluate_moving_contact_step,
        // object_state_phase="end", numeric_mode="float64").
        let anchor = still_frame().anchor();
        let result = evaluate_moving_contact_step(
            &anchor,
            "roadobj:7:node3",
            [0.002, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            IDENTITY,
            [0.0; 3],
            0.002,
            "float64",
            "end",
        )
        .unwrap();
        assert_eq!(result.position_after, [0.252, -0.5, 0.5]);
        assert_eq!(result.velocity_after, [1.0, 0.0, 0.0]);
        assert_eq!(result.matrix_after, IDENTITY);
        assert_eq!(result.angular_velocity_after, [0.0; 3]);
        assert_eq!(result.anchor_after.local_position, [0.25, -0.5, 0.5]);
        assert_eq!(result.object_position_after, [0.002, 0.0, 0.0]);
        assert!(result.object_orientation_integrator.is_none());
        assert_eq!(result.contact_plane_error_before_projection, 0.0);
        assert_eq!(result.reference_frame_owner, "roadobj:7:node3");
        assert_eq!(
            result.closure_status,
            "generic_rigid_moving_frame_contact_manifold_executable"
        );
    }

    #[test]
    fn step_guards_match_python_order() {
        let anchor = still_frame().anchor();
        let error = evaluate_moving_contact_step(
            &anchor, "roadobj:other", [0.0; 3], [0.0; 3], IDENTITY, [0.0; 3],
            0.002, "float64", "end",
        )
        .unwrap_err();
        assert_eq!(error, "moving-contact object identity changed");
        let error = evaluate_moving_contact_step(
            &anchor, "roadobj:7:node3", [0.0; 3], [0.0; 3], IDENTITY, [0.0; 3],
            0.0, "float64", "end",
        )
        .unwrap_err();
        assert_eq!(error, "dt must be positive");
        let error = evaluate_moving_contact_step(
            &anchor, "roadobj:7:node3", [0.0; 3], [0.0; 3], IDENTITY, [0.0; 3],
            0.002, "float16", "end",
        )
        .unwrap_err();
        assert_eq!(error, "numeric_mode must be float32 or float64");
        let error = evaluate_moving_contact_step(
            &anchor, "roadobj:7:node3", [0.0; 3], [0.0; 3], IDENTITY, [0.0; 3],
            0.002, "float64", "middle",
        )
        .unwrap_err();
        assert_eq!(error, "object_state_phase must be start or end");
    }
}
