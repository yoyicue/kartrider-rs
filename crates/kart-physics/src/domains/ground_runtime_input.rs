//! Port of `scripts/evaluate_ground_runtime_input.py` — map normal-ground
//! runtime locals into the executable force-body contract. Consumed by
//! `step_control` in `exact_drive_basis` input mode when the control frame
//! carries `ground_runtime`.
//!
//! All arithmetic is plain f64 in Python's evaluation order;
//! `math.hypot(a, b)` is CPython's double-double `vector_norm`
//! ([`cpython_hypot2`]), not the platform libm hypot.

use crate::num::cpython_hypot2;

/// Python `select_normal_ground_tune_gain` result key `source`.
pub const TUNE_SELECTOR_SOURCE: &str = "AF4==0 && AF8<=0 && ObfByte(AFC)==0 && field_65c<=0";
/// Python low-projection payload key `runtime_adapter`.
pub const LOW_PROJECTION_RUNTIME_ADAPTER: &str = "low_projection_raw_object_snapshot_v1";

/// Python `select_normal_ground_tune_gain(...)` result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalGroundTuneGain {
    pub enabled: bool,
    /// `longitudinal_gain_scale`: `Some(1.0)` when enabled, else `None`.
    pub longitudinal_gain_scale: Option<f64>,
}

/// `select_normal_ground_tune_gain(selector_af4=..., field_af8=...,
/// selector_afc_obf=..., field_65c=...)`: the var_11 gate selected before
/// the normal-ground body.
pub fn select_normal_ground_tune_gain(
    selector_af4: bool,
    field_af8: f64,
    selector_afc_obf: bool,
    field_65c: f64,
) -> Result<NormalGroundTuneGain, String> {
    if !field_af8.is_finite() || !field_65c.is_finite() {
        return Err("normal-ground tune selector inputs must be finite".to_string());
    }
    let enabled = !selector_af4 && field_af8 <= 0.0 && !selector_afc_obf && field_65c <= 0.0;
    Ok(NormalGroundTuneGain {
        enabled,
        longitudinal_gain_scale: if enabled { Some(1.0) } else { None },
    })
}

/// Python `evaluate_low_projection_runtime_input(...)` payload minus the
/// constant `numeric_mode`/`runtime_adapter` metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowProjectionRuntimeInput {
    pub projection_len: f64,
    pub ground_forward_term: f64,
    pub drive_direction: f64,
    pub normalized_velocity_raw: f64,
    pub correction_raw: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub base_lateral_gain: f64,
    pub axis_long: f64,
    pub axis_lat: f64,
}

/// `evaluate_low_projection_runtime_input(...)`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_low_projection_runtime_input(
    field_adc: f64,
    field_ae0: f64,
    field_d4: f64,
    field_9f0: f64,
    tune_778: f64,
    tune_77c: f64,
    axis_long: f64,
    axis_lat: f64,
    ground_forward_term: f64,
) -> Result<LowProjectionRuntimeInput, String> {
    let values = [
        field_adc,
        field_ae0,
        field_d4,
        field_9f0,
        tune_778,
        tune_77c,
        axis_long,
        axis_lat,
        ground_forward_term,
    ];
    if !values.iter().all(|value| value.is_finite()) {
        return Err("low-projection runtime inputs must be finite".to_string());
    }
    Ok(LowProjectionRuntimeInput {
        projection_len: cpython_hypot2(field_adc, field_ae0),
        ground_forward_term,
        drive_direction: if field_adc > 0.0 { 1.0 } else { -1.0 },
        normalized_velocity_raw: field_ae0,
        correction_raw: field_d4,
        mass_scale: field_9f0,
        base_longitudinal_gain: tune_778,
        base_lateral_gain: tune_77c,
        axis_long,
        axis_lat,
    })
}

/// Python `evaluate_normal_ground_runtime_input(...)` payload minus the
/// constant `numeric_mode` metadata. `runtime_adapter` is retained because
/// `step_control` surfaces it in the control prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalGroundRuntimeInput {
    pub surface_mode: i64,
    pub speed_magnitude: f64,
    pub ground_forward_term: f64,
    pub contact_basis_product: f64,
    pub drive_direction: f64,
    pub normalized_velocity: f64,
    pub longitudinal_correction: f64,
    pub lateral_correction: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub base_lateral_gain: f64,
    pub ground_force_multiplier: f64,
    pub settlement_multiplier: f64,
    pub tune_75c: f64,
    pub torque_y_gain: f64,
    pub axis_long: f64,
    pub axis_lat: f64,
    pub runtime_adapter: &'static str,
}

/// `evaluate_normal_ground_runtime_input(...)`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_normal_ground_runtime_input(
    surface_mode: i64,
    field_adc: f64,
    normalized_velocity: f64,
    longitudinal_correction: f64,
    lateral_correction: f64,
    field_9f0: f64,
    tune_778: f64,
    tune_77c: f64,
    tune_75c: f64,
    tune_758: f64,
    axis_long: f64,
    axis_lat: f64,
    settlement_y: f64,
    ground_forward_term: f64,
    selector_af4: bool,
    field_af8: f64,
    selector_afc_obf: bool,
    field_65c: f64,
) -> Result<NormalGroundRuntimeInput, String> {
    let values = [
        field_adc,
        normalized_velocity,
        longitudinal_correction,
        lateral_correction,
        field_9f0,
        tune_778,
        tune_77c,
        tune_75c,
        tune_758,
        axis_long,
        axis_lat,
        settlement_y,
        ground_forward_term,
        field_af8,
        field_65c,
    ];
    if !values.iter().all(|value| value.is_finite()) {
        return Err("normal-ground runtime inputs must be finite".to_string());
    }
    if surface_mode != 0 {
        return Err(
            "normal-ground runtime adapter currently requires surface_mode=0".to_string(),
        );
    }
    let tune_selector =
        select_normal_ground_tune_gain(selector_af4, field_af8, selector_afc_obf, field_65c)?;
    let longitudinal_gain = if tune_selector.enabled { tune_778 } else { 1.0 };
    let lateral_gain = if tune_selector.enabled { tune_77c } else { 1.0 };
    // Settlement is captured as the already selected stack Y term. The body
    // contract represents it as a multiplier of -abs(force_x), whose force_x
    // can be reconstructed from the same runtime operands.
    let longitudinal = field_9f0
        * longitudinal_gain
        * (ground_forward_term * if field_adc > 0.0 { 1.0 } else { -1.0 }
            - normalized_velocity
            - longitudinal_correction);
    let lateral = field_9f0 * lateral_gain * (-normalized_velocity + lateral_correction);
    let force_x = longitudinal + lateral;
    let settlement_multiplier = if force_x != 0.0 {
        settlement_y.abs() / force_x.abs()
    } else {
        0.0
    };
    Ok(NormalGroundRuntimeInput {
        surface_mode: 0,
        speed_magnitude: 0.0,
        ground_forward_term,
        contact_basis_product: 0.0,
        drive_direction: if field_adc > 0.0 { 1.0 } else { -1.0 },
        normalized_velocity,
        longitudinal_correction,
        lateral_correction,
        mass_scale: 1.0,
        base_longitudinal_gain: longitudinal_gain,
        base_lateral_gain: lateral_gain,
        ground_force_multiplier: field_9f0,
        settlement_multiplier,
        tune_75c,
        torque_y_gain: if tune_selector.enabled { tune_75c } else { tune_758 },
        axis_long,
        axis_lat,
        runtime_adapter: if tune_selector.enabled {
            "normal_ground_static_tune_selector_v2_tuned"
        } else {
            "normal_ground_static_tune_selector_v2_fallback"
        },
    })
}

/// Keyword arguments of `select_ground_runtime_input` beyond the four
/// positional runtime locals (Python `**fields`). The high-projection branch
/// reads every field; the low branch reads only the shared ones. Fields the
/// low branch does not read are `Option` and a missing key on the high
/// branch raises Python's `KeyError` message.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct GroundRuntimeFields {
    pub field_9f0: f64,
    pub tune_778: f64,
    pub tune_77c: f64,
    pub axis_long: f64,
    pub axis_lat: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub surface_mode: Option<i64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub normalized_velocity: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub longitudinal_correction: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub lateral_correction: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub tune_75c: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub settlement_y: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub tune_758: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub selector_af4: Option<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub field_af8: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub selector_afc_obf: Option<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub field_65c: Option<f64>,
}

/// Python `select_ground_runtime_input(...)` result: `{"branch": ...,
/// "payload": ...}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroundRuntimeSelection {
    /// `{"branch": "low_projection", "payload": ...}`.
    LowProjection(LowProjectionRuntimeInput),
    /// `{"branch": "ground_force", "payload": ...}`.
    GroundForce(NormalGroundRuntimeInput),
}

impl GroundRuntimeSelection {
    /// Python `ground_selection["branch"]`.
    pub fn branch(&self) -> &'static str {
        match self {
            GroundRuntimeSelection::LowProjection(_) => "low_projection",
            GroundRuntimeSelection::GroundForce(_) => "ground_force",
        }
    }

    /// Python `payload["runtime_adapter"]` (popped into the control
    /// prediction by `step_control`).
    pub fn runtime_adapter(&self) -> &'static str {
        match self {
            GroundRuntimeSelection::LowProjection(_) => LOW_PROJECTION_RUNTIME_ADAPTER,
            GroundRuntimeSelection::GroundForce(payload) => payload.runtime_adapter,
        }
    }
}

/// `select_ground_runtime_input(ground_forward_term=..., field_adc=...,
/// field_ae0=..., field_d4=..., **fields)`: select the low/high projection
/// body from the runtime projection itself.
pub fn select_ground_runtime_input(
    ground_forward_term: f64,
    field_adc: f64,
    field_ae0: f64,
    field_d4: f64,
    fields: &GroundRuntimeFields,
) -> Result<GroundRuntimeSelection, String> {
    let projection_len = cpython_hypot2(field_adc, field_ae0);
    if projection_len <= 5.0 {
        return Ok(GroundRuntimeSelection::LowProjection(
            evaluate_low_projection_runtime_input(
                field_adc,
                field_ae0,
                field_d4,
                fields.field_9f0,
                fields.tune_778,
                fields.tune_77c,
                fields.axis_long,
                fields.axis_lat,
                ground_forward_term,
            )?,
        ));
    }
    // Python reads these via `fields[...]`; a missing key is a KeyError.
    fn required(value: Option<f64>, name: &str) -> Result<f64, String> {
        value.ok_or_else(|| format!("KeyError: '{name}'"))
    }
    let payload = evaluate_normal_ground_runtime_input(
        fields.surface_mode.unwrap_or(0),
        field_adc,
        required(fields.normalized_velocity, "normalized_velocity")?,
        required(fields.longitudinal_correction, "longitudinal_correction")?,
        required(fields.lateral_correction, "lateral_correction")?,
        fields.field_9f0,
        fields.tune_778,
        fields.tune_77c,
        required(fields.tune_75c, "tune_75c")?,
        required(fields.tune_758, "tune_758")?,
        fields.axis_long,
        fields.axis_lat,
        required(fields.settlement_y, "settlement_y")?,
        ground_forward_term,
        fields
            .selector_af4
            .ok_or_else(|| "KeyError: 'selector_af4'".to_string())?,
        required(fields.field_af8, "field_af8")?,
        fields
            .selector_afc_obf
            .ok_or_else(|| "KeyError: 'selector_afc_obf'".to_string())?,
        required(fields.field_65c, "field_65c")?,
    )?;
    Ok(GroundRuntimeSelection::GroundForce(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields() -> GroundRuntimeFields {
        GroundRuntimeFields {
            field_9f0: 2.0,
            tune_778: 0.8,
            tune_77c: 0.6,
            axis_long: 1.0,
            axis_lat: 1.0,
            surface_mode: None,
            normalized_velocity: Some(0.1),
            longitudinal_correction: Some(0.05),
            lateral_correction: Some(0.02),
            tune_75c: Some(0.9),
            settlement_y: Some(-0.5),
            tune_758: Some(0.7),
            selector_af4: Some(false),
            field_af8: Some(0.0),
            selector_afc_obf: Some(false),
            field_65c: Some(0.0),
        }
    }

    #[test]
    fn low_projection_branch_below_five() {
        // Computed with the Python authority:
        //   select_ground_runtime_input(ground_forward_term=0.25, field_adc=3.0,
        //     field_ae0=4.0, field_d4=0.5, **fields())  # projection_len == 5.0
        let selection = select_ground_runtime_input(0.25, 3.0, 4.0, 0.5, &fields()).unwrap();
        let GroundRuntimeSelection::LowProjection(payload) = selection else {
            panic!("projection_len 5.0 selects the low branch")
        };
        assert_eq!(selection.branch(), "low_projection");
        assert_eq!(payload.projection_len, 5.0);
        assert_eq!(payload.drive_direction, 1.0);
        assert_eq!(payload.mass_scale, 2.0);
        assert_eq!(
            selection.runtime_adapter(),
            "low_projection_raw_object_snapshot_v1"
        );
    }

    #[test]
    fn ground_force_branch_above_five_tuned() {
        // select_ground_runtime_input(ground_forward_term=0.25, field_adc=3.6,
        //   field_ae0=4.8, field_d4=0.5, **fields())  # projection_len == 6.0
        let selection = select_ground_runtime_input(0.25, 3.6, 4.8, 0.5, &fields()).unwrap();
        let GroundRuntimeSelection::GroundForce(payload) = selection else {
            panic!("projection_len 6.0 selects the ground_force branch")
        };
        assert_eq!(selection.branch(), "ground_force");
        assert_eq!(payload.surface_mode, 0);
        assert_eq!(payload.base_longitudinal_gain, 0.8);
        assert_eq!(payload.base_lateral_gain, 0.6);
        assert_eq!(payload.torque_y_gain, 0.9);
        assert_eq!(
            selection.runtime_adapter(),
            "normal_ground_static_tune_selector_v2_tuned"
        );
        // longitudinal = 2.0 * 0.8 * (0.25 * 1.0 - 0.1 - 0.05)
        // lateral = 2.0 * 0.6 * (-0.1 + 0.02)
        let longitudinal: f64 = 2.0 * 0.8 * (0.25 * 1.0 - 0.1 - 0.05);
        let lateral: f64 = 2.0 * 0.6 * (-0.1 + 0.02);
        assert_eq!(
            payload.settlement_multiplier,
            (-0.5_f64).abs() / (longitudinal + lateral).abs()
        );
    }

    #[test]
    fn fallback_adapter_when_tune_gate_disabled() {
        let mut disabled = fields();
        disabled.selector_af4 = Some(true);
        let selection = select_ground_runtime_input(0.25, 3.6, 4.8, 0.5, &disabled).unwrap();
        let GroundRuntimeSelection::GroundForce(payload) = selection else {
            panic!("projection_len 6.0 selects the ground_force branch")
        };
        assert_eq!(payload.base_longitudinal_gain, 1.0);
        assert_eq!(payload.torque_y_gain, 0.7);
        assert_eq!(
            selection.runtime_adapter(),
            "normal_ground_static_tune_selector_v2_fallback"
        );
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            select_normal_ground_tune_gain(false, f64::NAN, false, 0.0).unwrap_err(),
            "normal-ground tune selector inputs must be finite"
        );
        let mut bad_mode = fields();
        bad_mode.surface_mode = Some(2);
        assert_eq!(
            select_ground_runtime_input(0.25, 3.6, 4.8, 0.5, &bad_mode).unwrap_err(),
            "normal-ground runtime adapter currently requires surface_mode=0"
        );
        let mut missing = fields();
        missing.normalized_velocity = None;
        assert_eq!(
            select_ground_runtime_input(0.25, 3.6, 4.8, 0.5, &missing).unwrap_err(),
            "KeyError: 'normalized_velocity'"
        );
    }
}
