//! Port of `scripts/evaluate_drive_basis_input.py` — exact non-conflicting
//! arrow-key snapshot for the GoPlayKart+0xA80 drive basis.
//!
//! All outputs are exact small integers as f64 (0.0/±1.0) or byte-valued
//! ints; the Python module performs no float32 quantization, so this port
//! uses none either.

/// Python result key `selection_authority`.
pub const SELECTION_AUTHORITY: &str = "static_dispatch_plus_clean_runtime_bridge";
/// Python result key `scope`.
pub const SCOPE: &str =
    "held-state snapshot after sub_BD1120 dispatch; conflicting opposite-key ordering is rejected";

/// Python `evaluate_drive_basis_input(...)` result dict minus the static
/// `selection_authority`/`scope` metadata keys. The `dispatch_cases` mapping
/// is flattened into the three `dispatch_*` fields.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveBasisInput {
    pub drive_raw_aa0: f64,
    pub drive_sign_aac: i64,
    pub drive_signed_a80: f64,
    pub drive_slot0_a84: f64,
    pub drive_slot1_a90: f64,
    pub drive_slot_selector_a9c: i64,
    pub drive_selected_slot: f64,
    pub drive_opposite_slot: f64,
    /// `dispatch_cases.left_right_signed_axis`.
    pub dispatch_left_right_signed_axis: Option<i64>,
    /// `dispatch_cases.up_slot0`.
    pub dispatch_up_slot0: Option<i64>,
    /// `dispatch_cases.down_slot1`.
    pub dispatch_down_slot1: Option<i64>,
}

/// `evaluate_drive_basis_input(keys=..., selector_a9c=0, sign_aac=0,
/// sign_abc=0)`. Conflicting opposite keys are rejected before the byte-range
/// guard, exactly like the Python module.
pub fn evaluate_drive_basis_input(
    keys: &[String],
    selector_a9c: i64,
    sign_aac: i64,
    sign_abc: i64,
) -> Result<DriveBasisInput, String> {
    let active: std::collections::BTreeSet<String> =
        keys.iter().map(|key| key.to_uppercase()).collect();
    if active.contains("LEFT") && active.contains("RIGHT") {
        return Err("simultaneous LEFT+RIGHT depends on dispatcher event order".to_string());
    }
    if active.contains("UP") && active.contains("DOWN") {
        return Err("simultaneous UP+DOWN depends on dispatcher event order".to_string());
    }
    for value in [selector_a9c, sign_aac, sign_abc] {
        if !(0..=255).contains(&value) {
            return Err("drive-basis selector/sign fields must be byte values".to_string());
        }
    }
    let raw_aa0 = if active.contains("LEFT") {
        1.0
    } else if active.contains("RIGHT") {
        -1.0
    } else {
        0.0
    };
    let slot0_a84 = if active.contains("UP") { 1.0 } else { 0.0 };
    let slot1_a90 = if active.contains("DOWN") { 1.0 } else { 0.0 };
    let (selected, opposite) = if selector_a9c != 0 {
        (slot1_a90, slot0_a84)
    } else {
        (slot0_a84, slot1_a90)
    };
    Ok(DriveBasisInput {
        drive_raw_aa0: raw_aa0,
        drive_sign_aac: sign_aac,
        drive_signed_a80: raw_aa0 * if sign_aac != 0 { -1.0 } else { 1.0 },
        drive_slot0_a84: slot0_a84,
        drive_slot1_a90: slot1_a90,
        drive_slot_selector_a9c: selector_a9c,
        drive_selected_slot: selected,
        drive_opposite_slot: opposite * if sign_abc != 0 { -1.0 } else { 1.0 },
        dispatch_left_right_signed_axis: if active.contains("LEFT") {
            Some(0)
        } else if active.contains("RIGHT") {
            Some(1)
        } else {
            None
        },
        dispatch_up_slot0: if active.contains("UP") { Some(2) } else { None },
        dispatch_down_slot1: if active.contains("DOWN") { Some(3) } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn up_left_snapshot_matches_python() {
        // Computed with the Python authority:
        //   evaluate_drive_basis_input(keys=["UP", "LEFT"])
        let result = evaluate_drive_basis_input(&keys(&["UP", "LEFT"]), 0, 0, 0).unwrap();
        assert_eq!(result.drive_raw_aa0, 1.0);
        assert_eq!(result.drive_signed_a80, 1.0);
        assert_eq!(result.drive_slot0_a84, 1.0);
        assert_eq!(result.drive_slot1_a90, 0.0);
        assert_eq!(result.drive_selected_slot, 1.0);
        assert_eq!(result.drive_opposite_slot, 0.0);
        assert_eq!(result.dispatch_left_right_signed_axis, Some(0));
        assert_eq!(result.dispatch_up_slot0, Some(2));
        assert_eq!(result.dispatch_down_slot1, None);
    }

    #[test]
    fn selector_swaps_selected_and_opposite_slots() {
        // evaluate_drive_basis_input(keys=["DOWN"], selector_a9c=1, sign_abc=1)
        let result = evaluate_drive_basis_input(&keys(&["DOWN"]), 1, 0, 1).unwrap();
        assert_eq!(result.drive_selected_slot, 1.0);
        assert_eq!(result.drive_opposite_slot, -0.0);
        assert_eq!(result.drive_slot_selector_a9c, 1);
    }

    #[test]
    fn sign_aac_flips_signed_axis() {
        // evaluate_drive_basis_input(keys=["RIGHT"], sign_aac=1)
        let result = evaluate_drive_basis_input(&keys(&["RIGHT"]), 0, 1, 0).unwrap();
        assert_eq!(result.drive_raw_aa0, -1.0);
        assert_eq!(result.drive_signed_a80, 1.0);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            evaluate_drive_basis_input(&keys(&["LEFT", "RIGHT"]), 0, 0, 0).unwrap_err(),
            "simultaneous LEFT+RIGHT depends on dispatcher event order"
        );
        assert_eq!(
            evaluate_drive_basis_input(&keys(&["UP", "DOWN"]), 0, 0, 0).unwrap_err(),
            "simultaneous UP+DOWN depends on dispatcher event order"
        );
        assert_eq!(
            evaluate_drive_basis_input(&keys(&[]), 256, 0, 0).unwrap_err(),
            "drive-basis selector/sign fields must be byte values"
        );
        assert_eq!(
            evaluate_drive_basis_input(&keys(&[]), 0, -1, 0).unwrap_err(),
            "drive-basis selector/sign fields must be byte values"
        );
    }
}
