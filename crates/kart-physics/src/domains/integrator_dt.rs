//! Port of `scripts/evaluate_integrator_dt.py` — authoritative integrator dt
//! selection, with an audit-only legacy recovery path.

pub const KNOWN_SUBSTEPS: [f64; 2] = [0.001, 0.002];

/// Legacy row fields consumed by `select_integrator_dt` when no captured dt
/// is present (`pre_x/y/z`, `post_x/y/z`, `sub_660` in the Python row dict).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegacyDtRow {
    pub pre_xyz: [f64; 3],
    pub post_xyz: [f64; 3],
    pub sub_660: f64,
}

/// `recover_integrator_dt(delta_velocity, total_force, mass)`.
///
/// Least-squares dt recovery; intended only for legacy captured output
/// audits. `KNOWN_SUBSTEPS` selection replicates Python
/// `min(KNOWN_SUBSTEPS, key=abs(value - raw))` tie-breaking (first minimal
/// element wins, so an exact midpoint selects 0.001).
pub fn recover_integrator_dt(
    delta_velocity: [f64; 3],
    total_force: [f64; 3],
    mass: f64,
) -> Result<f64, String> {
    if !mass.is_finite() || mass <= 0.0 {
        return Err("mass must be positive and finite".to_string());
    }
    let acceleration = [
        total_force[0] / mass,
        total_force[1] / mass,
        total_force[2] / mass,
    ];
    // Python sum(...) over three floats: Neumaier-compensated in CPython
    // 3.12+ (see crate::num::sum_f64), replicated bit-exactly here.
    let denominator = crate::num::sum_f64(&[
        acceleration[0] * acceleration[0],
        acceleration[1] * acceleration[1],
        acceleration[2] * acceleration[2],
    ]);
    if denominator <= 1e-20 {
        return Err("cannot recover dt from zero acceleration".to_string());
    }
    let numerator = crate::num::sum_f64(&[
        delta_velocity[0] * acceleration[0],
        delta_velocity[1] * acceleration[1],
        delta_velocity[2] * acceleration[2],
    ]);
    let raw = numerator / denominator;
    // Python min(key=...): iterates in order, replaces only on strictly
    // smaller key, so ties keep the earlier (smaller) substep.
    let mut selected = KNOWN_SUBSTEPS[0];
    for &value in &KNOWN_SUBSTEPS[1..] {
        if (value - raw).abs() < (selected - raw).abs() {
            selected = value;
        }
    }
    if (raw - selected).abs() > 5e-5 {
        return Err(format!("recovered dt {raw:?} is not a known substep"));
    }
    Ok(selected)
}

/// `select_integrator_dt(row, total_force=...)`.
///
/// `captured_integrator_dt` models `row.get("integrator_dt")`: `None`
/// covers both Python `None` and `""`. When recovery is needed, Python would
/// raise `KeyError` on missing row fields; Rust requires `legacy_row` and
/// reports a plain `Err` instead (error text is not part of the parity
/// contract).
pub fn select_integrator_dt(
    captured_integrator_dt: Option<f64>,
    total_force: Option<[f64; 3]>,
    legacy_row: Option<LegacyDtRow>,
) -> Result<(f64, &'static str), String> {
    if let Some(captured) = captured_integrator_dt {
        if !captured.is_finite() || captured <= 0.0 {
            return Err(format!("invalid captured integrator_dt {captured:?}"));
        }
        return Ok((captured, "captured_caller_arg"));
    }
    let Some(total_force) = total_force else {
        return Err("legacy dt recovery requires total_force".to_string());
    };
    let Some(legacy_row) = legacy_row else {
        return Err("legacy dt recovery requires pre/post/sub_660 row fields".to_string());
    };
    let mut delta = [0.0, 0.0, 0.0];
    for i in 0..3 {
        delta[i] = legacy_row.post_xyz[i] - legacy_row.pre_xyz[i];
    }
    let dt = recover_integrator_dt(delta, total_force, legacy_row.sub_660)?;
    Ok((dt, "legacy_output_recovered_audit_only"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_dt_wins() {
        let (dt, source) = select_integrator_dt(Some(0.002), None, None).unwrap();
        assert_eq!(dt, 0.002);
        assert_eq!(source, "captured_caller_arg");
    }

    #[test]
    fn legacy_recovery_roundtrip() {
        // force/mass gives acc = [2.0, -1.0, 0.5]; delta = acc * 0.001.
        let legacy = LegacyDtRow {
            pre_xyz: [1.0, 2.0, 3.0],
            post_xyz: [1.002, 1.999, 3.0005],
            sub_660: 4.0,
        };
        let (dt, source) =
            select_integrator_dt(None, Some([8.0, -4.0, 2.0]), Some(legacy)).unwrap();
        assert_eq!(dt, 0.001);
        assert_eq!(source, "legacy_output_recovered_audit_only");
    }

    #[test]
    fn midpoint_and_nearby_raw_are_outside_tolerance() {
        // raw = 0.0015 is the exact KNOWN_SUBSTEPS midpoint: |raw - selected|
        // = 5e-4, far beyond the strict 5e-5 tolerance, so recovery errors
        // instead of exercising min() tie-breaking. The tie path is therefore
        // unobservable through the public function; it is ported faithfully
        // regardless. Error text matches Python repr for these values.
        assert_eq!(
            recover_integrator_dt([0.0015, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0).unwrap_err(),
            "recovered dt 0.0015 is not a known substep"
        );
        assert_eq!(
            recover_integrator_dt([0.0015000001, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0).unwrap_err(),
            "recovered dt 0.0015000001 is not a known substep"
        );
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            recover_integrator_dt([0.0; 3], [1.0; 3], 0.0).unwrap_err(),
            "mass must be positive and finite"
        );
        assert_eq!(
            recover_integrator_dt([0.0; 3], [0.0; 3], 1.0).unwrap_err(),
            "cannot recover dt from zero acceleration"
        );
        assert_eq!(
            select_integrator_dt(None, None, None).unwrap_err(),
            "legacy dt recovery requires total_force"
        );
        assert!(recover_integrator_dt([0.01, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0).is_err());
    }
}
