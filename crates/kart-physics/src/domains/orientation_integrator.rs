//! Port of `scripts/evaluate_orientation_integrator.py` — exact quaternion
//! orientation update/stabilization half of sub_B2FD80.
//!
//! Dual numeric path:
//! - `runtime_float32_exact=true` uses the `_runtime_*` f32-quantized helpers
//!   (crate::num f* chain; normalize has no zero guard, matching Python where
//!   a zero length would raise ZeroDivisionError — unreachable in practice).
//! - otherwise the plain f64 path (`_quat_mul`/`_normalize`/... helpers).
//!
//! Stabilization retry loop (`scale_xy_0.1` x3 then `clear_xy`) is disabled
//! when `stabilization_disabled_b70` or `rail_state_1cc in (4, 6)`.

use crate::num::{f32q, fadd, fdiv, fmul, fsub};

/// One entry of the Python `stabilization_retries` list.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizationRetry {
    pub attempt: i64,
    pub axis2_z_before: f64,
    pub operation: &'static str,
    pub angular_velocity_after_operation: [f64; 3],
    pub axis2_z_after: f64,
}

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct OrientationIntegratorResult {
    pub matrix_9c_before: [f64; 9],
    pub quaternion_backup: [f64; 4],
    pub quaternion_after: [f64; 4],
    pub matrix_9c_after: [f64; 9],
    pub angular_velocity_cc_before: [f64; 3],
    pub angular_velocity_cc_after: [f64; 3],
    pub pre_clear_xy: bool,
    pub stabilization_enabled: bool,
    pub stabilization_retries: Vec<StabilizationRetry>,
    pub axis2_z_after: f64,
}

// --- runtime float32-exact helpers (`_runtime_*`) ---

pub(crate) fn runtime_normalize(q: [f64; 4]) -> [f64; 4] {
    let length2 = fadd(
        fadd(fadd(fmul(q[0], q[0]), fmul(q[1], q[1])), fmul(q[2], q[2])),
        fmul(q[3], q[3]),
    );
    let length = f32q(length2.sqrt());
    [
        fdiv(q[0], length),
        fdiv(q[1], length),
        fdiv(q[2], length),
        fdiv(q[3], length),
    ]
}

pub(crate) fn runtime_matrix_to_quat_upright(m: [f64; 9]) -> Result<[f64; 4], String> {
    let trace = fadd(fadd(m[0], m[4]), m[8]);
    if trace > 0.0 {
        let root = f32q(fadd(trace, 1.0).sqrt());
        let scale = fdiv(0.5, root);
        return Ok([
            fmul(root, 0.5),
            fmul(fsub(m[7], m[5]), scale),
            fmul(fsub(m[2], m[6]), scale),
            fmul(fsub(m[3], m[1]), scale),
        ]);
    }
    let diagonal = [m[0], m[4], m[8]];
    // Python max(range(3), key=diagonal.__getitem__): first maximum wins ties.
    let mut i = 0usize;
    for candidate in 1..3 {
        if diagonal[candidate] > diagonal[i] {
            i = candidate;
        }
    }
    let j = (i + 1) % 3;
    let k = (i + 2) % 3;
    let root_arg = fadd(fsub(fsub(diagonal[i], diagonal[j]), diagonal[k]), 1.0);
    // Python max(0.0, root_arg): NaN root_arg also selects 0.0.
    let root = f32q(if root_arg > 0.0 { root_arg } else { 0.0 }.sqrt());
    if root == 0.0 {
        return Err("degenerate runtime rotation matrix".to_string());
    }
    let scale = fdiv(0.5, root);
    let mut q = [0.0, 0.0, 0.0, 0.0];
    q[i + 1] = fmul(root, 0.5);
    q[0] = fmul(fsub(m[3 * k + j], m[3 * j + k]), scale);
    q[j + 1] = fmul(fadd(m[3 * i + j], m[3 * j + i]), scale);
    q[k + 1] = fmul(fadd(m[3 * i + k], m[3 * k + i]), scale);
    Ok(q)
}

pub(crate) fn runtime_quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        fsub(
            fsub(fsub(fmul(aw, bw), fmul(ax, bx)), fmul(ay, by)),
            fmul(az, bz),
        ),
        fsub(
            fadd(fadd(fmul(aw, bx), fmul(ax, bw)), fmul(ay, bz)),
            fmul(az, by),
        ),
        fsub(
            fadd(fadd(fmul(aw, by), fmul(ay, bw)), fmul(az, bx)),
            fmul(ax, bz),
        ),
        fsub(
            fadd(fadd(fmul(aw, bz), fmul(az, bw)), fmul(ax, by)),
            fmul(ay, bx),
        ),
    ]
}

pub(crate) fn runtime_quat_to_matrix(q: [f64; 4]) -> [f64; 9] {
    let [w, x, y, z] = q;
    let (tx, ty, tz) = (fmul(2.0, x), fmul(2.0, y), fmul(2.0, z));
    let (xw, yw, zw) = (fmul(tx, w), fmul(ty, w), fmul(tz, w));
    let (xx, yx, zx) = (fmul(tx, x), fmul(ty, x), fmul(tz, x));
    let (yy, zy, zz) = (fmul(ty, y), fmul(tz, y), fmul(tz, z));
    [
        fsub(1.0, fadd(yy, zz)),
        fsub(yx, zw),
        fadd(zx, yw),
        fadd(yx, zw),
        fsub(1.0, fadd(xx, zz)),
        fsub(zy, xw),
        fsub(zx, yw),
        fadd(zy, xw),
        fsub(1.0, fadd(xx, yy)),
    ]
}

/// `(backup_quat, integrated_quat, matrix)` triple returned by
/// `_runtime_integrate_upright`.
pub(crate) type BackupQuatMatrix = ([f64; 4], [f64; 4], [f64; 9]);

/// `_runtime_integrate_upright`: no pre-integration normalization of the
/// matrix-derived quaternion (see Python comment on B2FEC0).
pub(crate) fn runtime_integrate_upright(
    matrix: [f64; 9],
    omega: [f64; 3],
    dt: f64,
) -> Result<BackupQuatMatrix, String> {
    let backup = runtime_matrix_to_quat_upright(matrix)?;
    let derivative = runtime_quat_mul(backup, [0.0, f32q(omega[0]), f32q(omega[1]), f32q(omega[2])]);
    let half_dt = fmul(dt, 0.5);
    let mut q = [0.0, 0.0, 0.0, 0.0];
    for i in 0..4 {
        q[i] = fadd(backup[i], fmul(derivative[i], half_dt));
    }
    let q = runtime_normalize(q);
    Ok((backup, q, runtime_quat_to_matrix(q)))
}

// --- plain f64 helpers ---

pub(crate) fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
    ]
}

pub(crate) fn normalize(q: [f64; 4]) -> Result<[f64; 4], String> {
    // Python: math.sqrt(sum(value * value for value in q)); CPython 3.12+
    // sum() is Neumaier-compensated (see crate::num::sum_f64).
    let length = crate::num::sum_f64(&[q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3]]).sqrt();
    if length == 0.0 {
        return Err("cannot normalize zero quaternion".to_string());
    }
    Ok([
        q[0] / length,
        q[1] / length,
        q[2] / length,
        q[3] / length,
    ])
}

pub(crate) fn quat_to_matrix(q: [f64; 4]) -> [f64; 9] {
    let [w, x, y, z] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

pub(crate) fn matrix_to_quat(m: [f64; 9]) -> Result<[f64; 4], String> {
    let trace = m[0] + m[4] + m[8];
    if trace > 0.0 {
        let root = (trace + 1.0).sqrt();
        let scale = 0.5 / root;
        return Ok([
            0.5 * root,
            (m[7] - m[5]) * scale,
            (m[2] - m[6]) * scale,
            (m[3] - m[1]) * scale,
        ]);
    }
    let diagonal = [m[0], m[4], m[8]];
    // First maximum wins ties (see runtime variant).
    let mut i = 0usize;
    for candidate in 1..3 {
        if diagonal[candidate] > diagonal[i] {
            i = candidate;
        }
    }
    let j = (i + 1) % 3;
    let k = (j + 1) % 3;
    let root_arg = diagonal[i] - diagonal[j] - diagonal[k] + 1.0;
    let root = (if root_arg > 0.0 { root_arg } else { 0.0 }).sqrt();
    if root == 0.0 {
        return Err("degenerate rotation matrix".to_string());
    }
    let scale = 0.5 / root;
    let mut q = [0.0, 0.0, 0.0, 0.0];
    q[i + 1] = 0.5 * root;
    q[0] = (m[3 * k + j] - m[3 * j + k]) * scale;
    q[j + 1] = (m[3 * i + j] + m[3 * j + i]) * scale;
    q[k + 1] = (m[3 * i + k] + m[3 * k + i]) * scale;
    Ok(q)
}

fn integrate_from_backup(
    backup_q: [f64; 4],
    omega: [f64; 3],
    dt: f64,
) -> Result<([f64; 4], [f64; 9]), String> {
    let derivative = quat_mul(backup_q, [0.0, omega[0], omega[1], omega[2]]);
    let mut q = [0.0, 0.0, 0.0, 0.0];
    for i in 0..4 {
        q[i] = backup_q[i] + derivative[i] * (dt * 0.5);
    }
    let q = normalize(q)?;
    Ok((q, quat_to_matrix(q)))
}

/// `evaluate_orientation_integrator(...)`.
pub fn evaluate_orientation_integrator(
    matrix_9c_before: [f64; 9],
    angular_velocity_cc_before: [f64; 3],
    dt: f64,
    pre_clear_xy: bool,
    stabilization_disabled_b70: bool,
    rail_state_1cc: i64,
    runtime_float32_exact: bool,
) -> Result<OrientationIntegratorResult, String> {
    let mut omega = angular_velocity_cc_before;
    if pre_clear_xy {
        omega[0] = 0.0;
        omega[1] = 0.0;
    }
    let (backup_q, mut q, mut matrix) = if runtime_float32_exact {
        runtime_integrate_upright(matrix_9c_before, omega, dt)?
    } else {
        let backup_q = normalize(matrix_to_quat(matrix_9c_before)?)?;
        let (q, matrix) = integrate_from_backup(backup_q, omega, dt)?;
        (backup_q, q, matrix)
    };
    let mut retries = Vec::new();
    let stabilization_enabled =
        !stabilization_disabled_b70 && rail_state_1cc != 4 && rail_state_1cc != 6;
    if stabilization_enabled {
        for attempt in 0..4 {
            let axis2_z_before = matrix[8];
            if axis2_z_before >= 0.5 {
                break;
            }
            let operation;
            if attempt == 3 {
                omega[0] = 0.0;
                omega[1] = 0.0;
                operation = "clear_xy";
            } else {
                omega[0] *= 0.1;
                omega[1] *= 0.1;
                operation = "scale_xy_0.1";
            }
            if runtime_float32_exact {
                let derivative = runtime_quat_mul(
                    backup_q,
                    [0.0, f32q(omega[0]), f32q(omega[1]), f32q(omega[2])],
                );
                let half_dt = fmul(dt, 0.5);
                let mut qn = [0.0, 0.0, 0.0, 0.0];
                for i in 0..4 {
                    qn[i] = fadd(backup_q[i], fmul(derivative[i], half_dt));
                }
                q = runtime_normalize(qn);
                matrix = runtime_quat_to_matrix(q);
            } else {
                let (qn, m) = integrate_from_backup(backup_q, omega, dt)?;
                q = qn;
                matrix = m;
            }
            retries.push(StabilizationRetry {
                attempt,
                axis2_z_before,
                operation,
                angular_velocity_after_operation: omega,
                axis2_z_after: matrix[8],
            });
        }
    }
    Ok(OrientationIntegratorResult {
        matrix_9c_before,
        quaternion_backup: backup_q,
        quaternion_after: q,
        matrix_9c_after: matrix,
        angular_velocity_cc_before,
        angular_velocity_cc_after: omega,
        pre_clear_xy,
        stabilization_enabled,
        stabilization_retries: retries,
        axis2_z_after: matrix[8],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rotation of +90 deg about X: axis2_z == 0, forces all 3 retries.
    const TILTED: [f64; 9] = [
        1.0, 0.0, 0.0, //
        0.0, 0.0, -1.0, //
        0.0, 1.0, 0.0,
    ];

    #[test]
    fn identity_float64_no_retries() {
        // Constants computed with the Python authority:
        //   evaluate_orientation_integrator(matrix_9c_before=identity,
        //     angular_velocity_cc_before=[0.1, -0.2, 0.3], dt=0.002)
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = evaluate_orientation_integrator(
            identity,
            [0.1, -0.2, 0.3],
            0.002,
            false,
            false,
            0,
            false,
        )
        .unwrap();
        assert!(r.stabilization_enabled);
        assert!(r.stabilization_retries.is_empty());
        assert_eq!(
            r.quaternion_after,
            [
                0.9999999300000073,
                9.999999300000074e-5,
                -0.00019999998600000147,
                0.00029999997900000217
            ]
        );
        assert_eq!(r.axis2_z_after, 0.999999900000014);
    }

    #[test]
    fn identity_runtime_float32_no_retries() {
        // Constants computed with the Python authority (same inputs,
        // runtime_float32_exact=True).
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = evaluate_orientation_integrator(
            identity,
            [0.1, -0.2, 0.3],
            0.002,
            false,
            false,
            0,
            true,
        )
        .unwrap();
        assert_eq!(
            r.quaternion_after,
            [
                1.0,
                0.00010000000474974513,
                -0.00020000000949949026,
                0.0003000000142492354
            ]
        );
        assert_eq!(r.axis2_z_after, 0.9999998807907104);
    }

    #[test]
    fn tilted_matrix_runs_all_retries_float64() {
        // Python authority: TILTED + omega [0,0,0] yields 4 retries
        // (3x scale_xy_0.1 then clear_xy); axis2_z stays < 0.5 throughout.
        let r = evaluate_orientation_integrator(
            TILTED,
            [0.0, 0.0, 0.0],
            0.002,
            false,
            false,
            0,
            false,
        )
        .unwrap();
        assert_eq!(r.stabilization_retries.len(), 4);
        assert_eq!(r.stabilization_retries[0].operation, "scale_xy_0.1");
        assert_eq!(r.stabilization_retries[1].operation, "scale_xy_0.1");
        assert_eq!(r.stabilization_retries[2].operation, "scale_xy_0.1");
        assert_eq!(r.stabilization_retries[3].operation, "clear_xy");
        assert_eq!(r.axis2_z_after, 2.220446049250313e-16);
    }

    #[test]
    fn tilted_matrix_runs_all_retries_runtime_float32() {
        // Python authority: same as above with runtime_float32_exact=True;
        // quantized axis2_z stays < 0.5 through all retries.
        let r = evaluate_orientation_integrator(
            TILTED,
            [0.0, 0.0, 0.0],
            0.002,
            false,
            false,
            0,
            true,
        )
        .unwrap();
        assert_eq!(r.stabilization_retries.len(), 4);
        assert_eq!(r.axis2_z_after, -1.1920928955078125e-7);
        assert_eq!(r.quaternion_backup, [0.7071067690849304, 0.7071067690849304, 0.0, 0.0]);
    }

    #[test]
    fn rail_state_4_and_6_disable_stabilization() {
        for rail in [4, 6] {
            let r = evaluate_orientation_integrator(
                TILTED,
                [0.5, 0.5, 0.0],
                0.002,
                false,
                false,
                rail,
                false,
            )
            .unwrap();
            assert!(!r.stabilization_enabled);
            assert!(r.stabilization_retries.is_empty());
        }
    }

    #[test]
    fn pre_clear_xy_zeroes_omega_xy() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = evaluate_orientation_integrator(
            identity,
            [1.0, 2.0, 3.0],
            0.002,
            true,
            false,
            0,
            false,
        )
        .unwrap();
        assert_eq!(r.angular_velocity_cc_after, [0.0, 0.0, 3.0]);
        // Python authority: quaternion_after with omega z-only.
        assert_eq!(
            r.quaternion_after,
            [0.9999955000303749, 0.0, 0.0, 0.002999986500091125]
        );
    }
}
