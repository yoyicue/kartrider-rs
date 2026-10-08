//! Port of the pure decoded-defaults helpers from
//! `scripts/surface_tune_defaults.py`. Not an `evaluate_*` module itself;
//! ported here because `evaluate_ground_drive_force_body` and
//! `evaluate_drift_recovery_timer` import it (tiny pure helper, kept as its
//! own module so both consumers share one transliteration).

/// `RECORD_CTOR_DEFAULT` — ctor/reset initialization of one 0x2C record.
pub const RECORD_CTOR_DEFAULT: [f64; 11] = [120.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// `build_surface_records()`: six records, then the sub_B279D0 overrides.
pub fn build_surface_records() -> [[f64; 11]; 6] {
    let mut records = [RECORD_CTOR_DEFAULT; 6];
    records[2][0] = 340.0;
    records[2][1] = -1.0;
    records[2][2] = -1.0;
    records[2][7] = 0.6;
    records[3][7] = 0.85;
    records[4][0] = 180.0;
    records[4][3] = 0.5;
    records[4][7] = 0.6;
    records[5][1] = -2.0;
    records[5][2] = -2.0;
    records
}

/// Python list indexing: negative indices wrap around (`records[-1]` is the
/// last entry). Only reachable for negative `surface_mode`; a double
/// out-of-range would be a Python `IndexError`, which no caller exercises.
fn record_at(records: &[[f64; 11]; 6], index: i64) -> [f64; 11] {
    let wrapped = if index < 0 { index + 6 } else { index };
    records[wrapped as usize]
}

/// `ground_surface_profile(surface_mode)` return dict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundSurfaceProfile {
    pub speed_denominator: f64,
    pub speed_quadratic_gain: f64,
    pub speed_bias: f64,
    pub force_scale: f64,
}

/// `ground_surface_profile(surface_mode)`: `None` for mode 0, `ValueError`
/// for an uninitialized mode.
pub fn ground_surface_profile(surface_mode: i64) -> Result<Option<GroundSurfaceProfile>, String> {
    if surface_mode == 0 {
        return Ok(None);
    }
    let records = build_surface_records();
    let index = surface_mode * 2;
    if index >= records.len() as i64 {
        return Err(format!("unsupported initialized surface_mode {surface_mode}"));
    }
    let record = record_at(&records, index);
    Ok(Some(GroundSurfaceProfile {
        speed_denominator: record[0],
        speed_quadratic_gain: 2.0, // GoPlayKart+0x650
        speed_bias: 0.5,           // GoPlayKart+0x654
        force_scale: record[3],
    }))
}

/// One half of the `surface1_recovery_profiles()` dict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecoverySurfaceProfile {
    pub longitudinal_gain_offset: f64,
    pub lateral_gain_offset: f64,
    pub force_scale: f64,
    pub forward_target_gain: f64,
}

/// `surface1_recovery_profiles()` return dict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface1RecoveryProfiles {
    pub active_surface: RecoverySurfaceProfile,
    pub timer_surface: RecoverySurfaceProfile,
}

/// `surface1_recovery_profiles()`.
pub fn surface1_recovery_profiles() -> Surface1RecoveryProfiles {
    let record = build_surface_records()[3];
    Surface1RecoveryProfiles {
        active_surface: RecoverySurfaceProfile {
            longitudinal_gain_offset: record[1],
            lateral_gain_offset: record[2],
            force_scale: record[3],
            forward_target_gain: record[9],
        },
        timer_surface: RecoverySurfaceProfile {
            longitudinal_gain_offset: record[4],
            lateral_gain_offset: record[5],
            force_scale: record[6],
            forward_target_gain: record[10],
        },
    }
}

/// `active_non_surface1_profile(surface_mode)` return dict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonSurfaceProfile {
    pub longitudinal_gain_offset: f64,
    pub lateral_gain_offset: f64,
    pub force_scale: f64,
}

/// `active_non_surface1_profile(surface_mode)`.
pub fn active_non_surface1_profile(surface_mode: i64) -> Result<NonSurfaceProfile, String> {
    let records = build_surface_records();
    let index = surface_mode * 2 + 1;
    if index >= records.len() as i64 {
        return Err(format!("unsupported initialized surface_mode {surface_mode}"));
    }
    let record = record_at(&records, index);
    Ok(NonSurfaceProfile {
        longitudinal_gain_offset: record[1],
        lateral_gain_offset: record[2],
        force_scale: record[3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_match_python_defaults() {
        // Constants computed with the Python authority (surface_tune_defaults).
        let mode1 = ground_surface_profile(1).unwrap().unwrap();
        assert_eq!(
            mode1,
            GroundSurfaceProfile {
                speed_denominator: 340.0,
                speed_quadratic_gain: 2.0,
                speed_bias: 0.5,
                force_scale: 1.0,
            }
        );
        let mode2 = ground_surface_profile(2).unwrap().unwrap();
        assert_eq!(mode2.speed_denominator, 180.0);
        assert_eq!(mode2.force_scale, 0.5);
        assert!(ground_surface_profile(0).unwrap().is_none());
        assert_eq!(
            ground_surface_profile(3).unwrap_err(),
            "unsupported initialized surface_mode 3"
        );
        let recovery = surface1_recovery_profiles();
        assert_eq!(recovery.active_surface.force_scale, 1.0);
        assert_eq!(recovery.timer_surface.force_scale, 1.0);
        let non_surface = active_non_surface1_profile(2).unwrap();
        assert_eq!(
            non_surface,
            NonSurfaceProfile {
                longitudinal_gain_offset: -2.0,
                lateral_gain_offset: -2.0,
                force_scale: 1.0,
            }
        );
    }
}
