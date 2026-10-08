//! KAP path topology helpers shared by the query CLI and kart-autoprobe's
//! drive-side gap-transfer controller.

use crate::geom::{point_distance_2d, P3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KapGapWindow {
    pub approach_start_s: f32,
    pub entry_s: f32,
    pub exit_s: f32,
    pub required_speed: f32,
}

/// Finds unsupported KAP path spans with supported takeoff/landing points.
///
/// The arithmetic and iteration order are the historical kart-autoprobe
/// implementation: the TSV baseline for `kap-topology` depends on these
/// exact binary32 operations and six-decimal formatting.
pub fn kap_gap_windows(
    points: &[P3],
    mut has_static_support: impl FnMut(P3) -> bool,
) -> Vec<KapGapWindow> {
    if points.len() < 2 {
        return Vec::new();
    }

    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0);
    let mut length = 0.0;
    for pair in points.windows(2) {
        length += point_distance_2d(pair[0], pair[1]);
        cumulative.push(length);
    }

    let unsupported = points
        .iter()
        .map(|point| !has_static_support(*point))
        .collect::<Vec<_>>();
    let mut gaps = Vec::new();
    let mut index = 0usize;
    while index < unsupported.len() {
        if !unsupported[index] {
            index += 1;
            continue;
        }
        let first = index;
        while index + 1 < unsupported.len() && unsupported[index + 1] {
            index += 1;
        }
        let last = index;
        // A topology transfer needs a supported takeoff and landing.
        if first > 0 && last + 1 < points.len() {
            let entry_index = first - 1;
            let exit_index = last + 1;
            let entry_s = cumulative[entry_index];
            let exit_s = cumulative[exit_index];
            let horizontal_gap = point_distance_2d(points[entry_index], points[exit_index]);
            if horizontal_gap >= 6.0 {
                let takeoff_z = points[entry_index][2];
                let landing_z = points[exit_index][2];
                let apex_z = points[entry_index..=exit_index]
                    .iter()
                    .map(|point| point[2])
                    .fold(takeoff_z.max(landing_z), f32::max);
                let gravity = 9.80665f32;
                let rise = (apex_z - takeoff_z).max(0.25);
                let fall = (apex_z - landing_z).max(0.25);
                let flight_time = (2.0 * rise / gravity).sqrt() + (2.0 * fall / gravity).sqrt();
                let required_speed = (horizontal_gap / flight_time.max(0.25))
                    .mul_add(1.15, 0.0)
                    .clamp(12.0, 25.0);
                let approach_m = (required_speed * 2.5).clamp(35.0, 60.0);
                gaps.push(KapGapWindow {
                    approach_start_s: (entry_s - approach_m).max(0.0),
                    entry_s,
                    exit_s,
                    required_speed,
                });
            }
        }
        index += 1;
    }
    gaps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_jump_arc() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ];
        let gaps = kap_gap_windows(&points, |point| point[0] < 15.0 || point[0] > 35.0);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].entry_s, 10.0);
        assert_eq!(gaps[0].exit_s, 40.0);
        assert!(gaps[0].required_speed >= 12.0);
        assert!(gaps[0].approach_start_s <= gaps[0].entry_s);
    }
}
