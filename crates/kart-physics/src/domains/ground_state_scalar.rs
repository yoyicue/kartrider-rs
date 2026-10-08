//! Port of `scripts/evaluate_ground_state_scalar.py` — reference evaluator
//! for the ordered sub_B2A100 state scalar pipeline, including
//! `evaluate_ground_correction_pipeline` and its per-action evaluators.

use crate::num::{fadd, fmul, sum_f64};

/// `evaluate_primary_force_delta` scope string.
pub const PRIMARY_SCOPE: &str =
    "closed primary normal/HW correction term; excludes optional add-ons and later producers";

/// Python `evaluate_primary_force_delta(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryForceDelta {
    pub force_delta: [f64; 3],
    pub scalar: f64,
    pub scope: &'static str,
}

/// `evaluate_primary_force_delta(...)`.
pub fn evaluate_primary_force_delta(
    correction_dir: [f64; 3],
    direction_gain: f64,
    selected_slot: f64,
    state_mult: f64,
) -> PrimaryForceDelta {
    let scalar = direction_gain * selected_slot * state_mult;
    PrimaryForceDelta {
        force_delta: [
            correction_dir[0] * scalar,
            correction_dir[1] * scalar,
            correction_dir[2] * scalar,
        ],
        scalar,
        scope: PRIMARY_SCOPE,
    }
}

/// Python `evaluate_negative_adc_addon(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeAdcAddon {
    pub branch: &'static str,
    pub force_delta: [f64; 3],
    pub scalar: f64,
    pub evidence: &'static str,
}

/// `evaluate_negative_adc_addon(...)`.
pub fn evaluate_negative_adc_addon(
    field_adc: f64,
    flag_af4: bool,
    flag_afc: bool,
    sub_ad0: [f64; 3],
    sub_af0: f64,
    field_9f0: f64,
    cap: f64,
) -> NegativeAdcAddon {
    if field_adc >= 0.0 {
        return NegativeAdcAddon {
            branch: "none",
            force_delta: [0.0, 0.0, 0.0],
            scalar: 0.0,
            evidence: "exact gate; no negative-ADC add-on",
        };
    }
    let active = flag_af4 || flag_afc;
    let branch = if active {
        "kind31_uncapped"
    } else {
        "kind32_capped"
    };
    let scalar = (if active {
        sub_af0
    } else {
        crate::num::min2_f64(sub_af0, cap)
    }) * field_9f0;
    NegativeAdcAddon {
        branch,
        force_delta: [sub_ad0[0] * scalar, sub_ad0[1] * scalar, sub_ad0[2] * scalar],
        scalar,
        evidence: "static exact/runtime formula observed",
    }
}

/// Python `evaluate_fallback_opposite_add(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackOppositeAdd {
    pub branch: &'static str,
    pub force_delta: [f64; 3],
    pub scalar: f64,
    pub evidence: &'static str,
}

/// `evaluate_fallback_opposite_add(...)`.
pub fn evaluate_fallback_opposite_add(
    sub_ad0: [f64; 3],
    sub_6f4: f64,
    sub_bc4: f64,
    drive_slot1_a90: f64,
) -> FallbackOppositeAdd {
    let scalar = -sub_6f4 * sub_bc4 * drive_slot1_a90;
    FallbackOppositeAdd {
        branch: "kind33_fallback_opposite_add",
        force_delta: [sub_ad0[0] * scalar, sub_ad0[1] * scalar, sub_ad0[2] * scalar],
        scalar,
        evidence: "runtime formula observed (2928 rows)",
    }
}

/// Python `evaluate_lateral_subtract(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct LateralSubtractResult {
    pub branch: &'static str,
    pub source: [f64; 3],
    pub accumulator_delta: [f64; 3],
    pub lateral_unit_residual: [f64; 3],
    pub evidence: &'static str,
}

/// `evaluate_lateral_subtract(...)`.
pub fn evaluate_lateral_subtract(
    kind: i64,
    velocity_c0: [f64; 3],
    axis_ac4: [f64; 3],
    sub_720: f64,
    sub_724: f64,
) -> Result<LateralSubtractResult, String> {
    if kind != 34 && kind != 35 {
        return Err("kind must be 34 or 35".to_string());
    }
    // Python: speed = math.sqrt(sum(c * c for c in velocity_c0)); sum() is
    // Neumaier-compensated (crate::num::sum_f64).
    let speed = sum_f64(&[
        velocity_c0[0] * velocity_c0[0],
        velocity_c0[1] * velocity_c0[1],
        velocity_c0[2] * velocity_c0[2],
    ])
    .sqrt();
    let unit = if speed > 0.0 {
        [
            velocity_c0[0] / speed,
            velocity_c0[1] / speed,
            velocity_c0[2] / speed,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    let projection = sum_f64(&[
        unit[0] * axis_ac4[0],
        unit[1] * axis_ac4[1],
        unit[2] * axis_ac4[2],
    ]);
    let lateral = [
        unit[0] - projection * axis_ac4[0],
        unit[1] - projection * axis_ac4[1],
        unit[2] - projection * axis_ac4[2],
    ];
    let scalar = if kind == 34 { sub_720 } else { sub_724 };
    let source = [lateral[0] * scalar, lateral[1] * scalar, lateral[2] * scalar];
    Ok(LateralSubtractResult {
        branch: if kind == 34 {
            "kind34_lateral_high_dot_subtract"
        } else {
            "kind35_lateral_low_dot_subtract"
        },
        source,
        accumulator_delta: [-source[0], -source[1], -source[2]],
        lateral_unit_residual: lateral,
        evidence: if kind == 34 {
            "runtime formula observed (501 rows)"
        } else {
            "runtime formula observed (12 rows)"
        },
    })
}

/// Python `evaluate_velocity_zero_action(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocityZeroAction {
    pub branch: &'static str,
    pub pre_velocity: [f64; 3],
    pub post_velocity: [f64; 3],
    pub terminal_for_current_correction_path: bool,
    pub evidence: &'static str,
}

/// `evaluate_velocity_zero_action(...)`.
pub fn evaluate_velocity_zero_action(
    kind: i64,
    velocity_c0: [f64; 3],
) -> Result<VelocityZeroAction, String> {
    if kind != 36 && kind != 37 {
        return Err("kind must be 36 or 37".to_string());
    }
    Ok(VelocityZeroAction {
        branch: if kind == 36 {
            "kind36_early_fallback_zero"
        } else {
            "kind37_late_low_lateral_zero"
        },
        pre_velocity: velocity_c0,
        post_velocity: [0.0, 0.0, 0.0],
        terminal_for_current_correction_path: true,
        evidence: if kind == 36 {
            "static write and trigger exact; runtime row not yet observed"
        } else {
            "runtime semantic observed (104 rows, post speed exactly zero)"
        },
    })
}

/// Python `select_fallback_lateral_actions(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackLateralActions {
    pub actions_in_program_order: Vec<i64>,
    pub timer_a80_after: f64,
    pub primary_path_selected: bool,
    pub fallback_path_selected: bool,
    pub fallback_initial: bool,
    pub fallback_before_lateral: bool,
    pub skip_lateral: bool,
    pub evidence: &'static str,
}

/// `select_fallback_lateral_actions(...)`. `velocity_length_sq` and `b8c`
/// are accepted for signature parity with Python but unused there too.
pub fn select_fallback_lateral_actions(
    selected_slot: i64,
    opposite_slot: i64,
    gate_1c0: bool,
    gate_6f3b00: bool,
    gate_6f3ae0: bool,
    field_adc: f64,
    timer_a80_before: f64,
    dt: f64,
    field_ae0: f64,
    b8c: f64,
    velocity_length_sq: f64,
    lateral_dot: f64,
) -> FallbackLateralActions {
    let _ = (b8c, velocity_length_sq);
    let blocking = gate_1c0 || gate_6f3b00 || gate_6f3ae0;
    let fallback_path = selected_slot <= 0 || blocking;
    if !fallback_path {
        return FallbackLateralActions {
            actions_in_program_order: Vec::new(),
            timer_a80_after: 0.0,
            primary_path_selected: true,
            fallback_path_selected: false,
            fallback_initial: false,
            fallback_before_lateral: false,
            skip_lateral: true,
            evidence: "exact static outer gate: selected slot > 0 with no blocking gate stays on primary path",
        };
    }
    let fallback_active = opposite_slot > 0 || blocking;
    let mut timer_after = timer_a80_before;
    let mut actions: Vec<i64> = Vec::new();
    let mut fallback_still_active = fallback_active;

    if !fallback_active {
        if (-0.5..=0.5).contains(&field_adc) {
            timer_after += dt;
        }
        return FallbackLateralActions {
            actions_in_program_order: actions,
            timer_a80_after: timer_after,
            primary_path_selected: false,
            fallback_path_selected: true,
            fallback_initial: false,
            fallback_before_lateral: false,
            skip_lateral: true,
            evidence: "exact static no-op fallback tail; bounded ADC only advances raw +0xA80 timer",
        };
    }

    if fallback_active && field_adc < 0.5 {
        timer_after += dt;
        if field_adc < -0.5 {
            timer_after = 1.0;
        }
        if timer_after > 0.2 {
            if blocking {
                if field_adc >= -0.5 {
                    actions.push(36);
                    fallback_still_active = false;
                }
            } else {
                actions.push(33);
                fallback_still_active = false;
            }
        } else if field_ae0.abs() < 0.2 {
            actions.push(37);
            fallback_still_active = false;
        }
    }

    // Kind33 and both direct-zero branches leave the current correction path;
    // the lateral 34/35 split is reachable only while fallback remains live.
    let skip_lateral = !fallback_still_active;
    if !skip_lateral {
        actions.push(if lateral_dot > 0.8 { 34 } else { 35 });
    }
    FallbackLateralActions {
        actions_in_program_order: actions,
        timer_a80_after: timer_after,
        primary_path_selected: false,
        fallback_path_selected: true,
        fallback_initial: fallback_active,
        fallback_before_lateral: fallback_still_active,
        skip_lateral,
        evidence: "exact static control flow; formulas/runtime coverage are tracked per action",
    }
}

/// One entry of the Python `trace` list in the gain/multiplier pipelines.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceStep {
    pub step: &'static str,
    pub operation: &'static str,
    pub operand: f64,
    pub result: f64,
}

/// Python `evaluate_direction_gain(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionGainResult {
    pub direction_gain: f64,
    pub trace: Vec<TraceStep>,
}

/// `evaluate_direction_gain(...)`.
pub fn evaluate_direction_gain(
    base_gain: f64,
    decoded_state: i64,
    byte_62c: bool,
    byte_9b8: bool,
    flag_afc: bool,
    obf_838: f64,
    obf_844: f64,
    obf_74c: f64,
) -> DirectionGainResult {
    let mut value = base_gain;
    let mut trace = vec![TraceStep {
        step: "initial",
        operation: "assign",
        operand: base_gain,
        result: value,
    }];
    if decoded_state == 1 && !byte_62c {
        value = if byte_9b8 { obf_844 } else { obf_838 };
        trace.push(TraceStep {
            step: "state1",
            operation: "assign",
            operand: value,
            result: value,
        });
    }
    if flag_afc {
        value = obf_74c;
        trace.push(TraceStep {
            step: "afc_override",
            operation: "assign",
            operand: obf_74c,
            result: value,
        });
    }
    DirectionGainResult {
        direction_gain: value,
        trace,
    }
}

/// Python `evaluate_state_mult(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct StateMultResult {
    pub state_mult: f64,
    pub trace: Vec<TraceStep>,
}

/// `evaluate_state_mult(...)`.
pub fn evaluate_state_mult(
    decoded_state: i64,
    boost_family: bool,
    attached_state: i64,
    e8c: bool,
    efc: bool,
    flag_5bc: bool,
    raw_144: f64,
    obf_808: f64,
    obf_814: f64,
    obf_69c: f64,
    obf_684: f64,
    obf_8ac: f64,
    obf_8e8: f64,
    obf_94c: f64,
) -> StateMultResult {
    let mut value = 1.0;
    let mut trace = vec![TraceStep {
        step: "initial",
        operation: "assign",
        operand: 1.0,
        result: value,
    }];

    if boost_family {
        if matches!(attached_state, 2 | 4 | 5 | 6) && decoded_state != 2 {
            value = obf_808;
            trace.push(TraceStep {
                step: "boost_special_attached",
                operation: "assign",
                operand: obf_808,
                result: value,
            });
        } else {
            value = obf_814;
            trace.push(TraceStep {
                step: "boost_ordinary",
                operation: "assign",
                operand: obf_814,
                result: value,
            });
        }
    }

    if decoded_state == 2 && !e8c {
        value *= obf_69c;
        trace.push(TraceStep {
            step: "state2",
            operation: "multiply",
            operand: obf_69c,
            result: value,
        });
    }

    if efc {
        value *= obf_684;
        trace.push(TraceStep {
            step: "efc",
            operation: "multiply",
            operand: obf_684,
            result: value,
        });
    }

    if decoded_state == 7 {
        value = 1.0;
        trace.push(TraceStep {
            step: "state7_barrier",
            operation: "assign",
            operand: 1.0,
            result: value,
        });
    }

    if decoded_state == 10 {
        value *= obf_8ac;
        trace.push(TraceStep {
            step: "state10",
            operation: "multiply",
            operand: obf_8ac,
            result: value,
        });
    }

    if flag_5bc {
        let operand = obf_8e8 * (if obf_94c > raw_144 { 2.0 } else { 1.0 });
        value *= operand;
        trace.push(TraceStep {
            step: "flag_5bc",
            operation: "multiply",
            operand,
            result: value,
        });
    }

    StateMultResult {
        state_mult: value,
        trace,
    }
}

/// One merged `{"kind": ..., **action}` event dict from the Python pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    /// `{"kind": "primary", **evaluate_primary_force_delta(...)}`
    Primary {
        force_delta: [f64; 3],
        scalar: f64,
        scope: &'static str,
    },
    /// `{"kind": 31|32, **evaluate_negative_adc_addon(...)}`
    NegativeAdc {
        kind: i64,
        branch: &'static str,
        force_delta: [f64; 3],
        scalar: f64,
        evidence: &'static str,
    },
    /// `{"kind": 33, **evaluate_fallback_opposite_add(...)}`
    FallbackOpposite {
        branch: &'static str,
        force_delta: [f64; 3],
        scalar: f64,
        evidence: &'static str,
    },
    /// `{"kind": 34|35, **evaluate_lateral_subtract(...)}`
    LateralSubtract {
        kind: i64,
        branch: &'static str,
        source: [f64; 3],
        accumulator_delta: [f64; 3],
        lateral_unit_residual: [f64; 3],
        evidence: &'static str,
    },
    /// `{"kind": 36|37, **evaluate_velocity_zero_action(...)}`
    VelocityZero {
        kind: i64,
        branch: &'static str,
        pre_velocity: [f64; 3],
        post_velocity: [f64; 3],
        terminal_for_current_correction_path: bool,
        evidence: &'static str,
    },
}

/// Python `evaluate_ground_correction_pipeline(...)` result dict minus
/// `static_source`. `selector` is absent in the Python dict when disabled.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundCorrectionPipelineResult {
    pub disabled: bool,
    pub selector: Option<FallbackLateralActions>,
    pub events: Vec<PipelineEvent>,
    pub timer_a80_after: f64,
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub velocity_c0_before: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub velocity_direct_write: bool,
}

/// `evaluate_ground_correction_pipeline(...)`.
///
/// Note: like the Python authority, an unknown `numeric_mode` silently takes
/// the float64 path (no mode validation in this module).
pub fn evaluate_ground_correction_pipeline(
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    velocity_c0: [f64; 3],
    disabled_b9c: bool,
    selected_slot: i64,
    opposite_slot: i64,
    gate_1c0: bool,
    gate_6f3b00: bool,
    gate_6f3ae0: bool,
    field_adc: f64,
    timer_a80_before: f64,
    dt: f64,
    field_ae0: f64,
    b8c: f64,
    lateral_dot: f64,
    correction_dir: [f64; 3],
    direction_gain: f64,
    state_mult: f64,
    flag_af4: bool,
    flag_afc: bool,
    sub_ad0: [f64; 3],
    sub_af0: f64,
    field_9f0: f64,
    sub_6f4: f64,
    sub_bc4: f64,
    drive_slot1_a90: f64,
    axis_ac4: [f64; 3],
    sub_720: f64,
    sub_724: f64,
    numeric_mode: &str,
) -> GroundCorrectionPipelineResult {
    let mut force = accumulator_614_before;
    let mut velocity = velocity_c0;
    let mut events: Vec<PipelineEvent> = Vec::new();
    if disabled_b9c {
        return GroundCorrectionPipelineResult {
            disabled: true,
            selector: None,
            events,
            timer_a80_after: timer_a80_before,
            accumulator_614_before,
            accumulator_614_after: force,
            accumulator_620_before,
            accumulator_620_after: accumulator_620_before,
            velocity_c0_before: velocity_c0,
            velocity_c0_after: velocity,
            velocity_direct_write: false,
        };
    }

    let selector = select_fallback_lateral_actions(
        selected_slot,
        opposite_slot,
        gate_1c0,
        gate_6f3b00,
        gate_6f3ae0,
        field_adc,
        timer_a80_before,
        dt,
        field_ae0,
        b8c,
        sum_f64(&[
            velocity[0] * velocity[0],
            velocity[1] * velocity[1],
            velocity[2] * velocity[2],
        ]),
        lateral_dot,
    );
    if selector.primary_path_selected {
        let primary = evaluate_primary_force_delta(
            correction_dir,
            direction_gain,
            selected_slot as f64,
            state_mult,
        );
        let mut primary_scalar = primary.scalar;
        let mut primary_force_delta = primary.force_delta;
        if numeric_mode == "float32" {
            primary_scalar = fmul(fmul(direction_gain, selected_slot as f64), state_mult);
            primary_force_delta = [
                fmul(correction_dir[0], primary_scalar),
                fmul(correction_dir[1], primary_scalar),
                fmul(correction_dir[2], primary_scalar),
            ];
            force = [
                fadd(force[0], primary_force_delta[0]),
                fadd(force[1], primary_force_delta[1]),
                fadd(force[2], primary_force_delta[2]),
            ];
        } else {
            force = [
                force[0] + primary_force_delta[0],
                force[1] + primary_force_delta[1],
                force[2] + primary_force_delta[2],
            ];
        }
        events.push(PipelineEvent::Primary {
            force_delta: primary_force_delta,
            scalar: primary_scalar,
            scope: primary.scope,
        });
        let addon = evaluate_negative_adc_addon(
            field_adc, flag_af4, flag_afc, sub_ad0, sub_af0, field_9f0, 5.0,
        );
        if addon.branch != "none" {
            force = [
                force[0] + addon.force_delta[0],
                force[1] + addon.force_delta[1],
                force[2] + addon.force_delta[2],
            ];
            events.push(PipelineEvent::NegativeAdc {
                kind: if addon.branch.starts_with("kind31") {
                    31
                } else {
                    32
                },
                branch: addon.branch,
                force_delta: addon.force_delta,
                scalar: addon.scalar,
                evidence: addon.evidence,
            });
        }
    } else {
        for &kind in &selector.actions_in_program_order {
            if kind == 33 {
                let action = evaluate_fallback_opposite_add(sub_ad0, sub_6f4, sub_bc4, drive_slot1_a90);
                force = [
                    force[0] + action.force_delta[0],
                    force[1] + action.force_delta[1],
                    force[2] + action.force_delta[2],
                ];
                events.push(PipelineEvent::FallbackOpposite {
                    branch: action.branch,
                    force_delta: action.force_delta,
                    scalar: action.scalar,
                    evidence: action.evidence,
                });
            } else if kind == 36 || kind == 37 {
                let action = evaluate_velocity_zero_action(kind, velocity)
                    .expect("kind 36/37 is pre-validated");
                velocity = action.post_velocity;
                events.push(PipelineEvent::VelocityZero {
                    kind,
                    branch: action.branch,
                    pre_velocity: action.pre_velocity,
                    post_velocity: action.post_velocity,
                    terminal_for_current_correction_path: action
                        .terminal_for_current_correction_path,
                    evidence: action.evidence,
                });
            } else {
                let action = evaluate_lateral_subtract(kind, velocity, axis_ac4, sub_720, sub_724)
                    .expect("kind 34/35 is pre-validated");
                force = [
                    force[0] + action.accumulator_delta[0],
                    force[1] + action.accumulator_delta[1],
                    force[2] + action.accumulator_delta[2],
                ];
                events.push(PipelineEvent::LateralSubtract {
                    kind,
                    branch: action.branch,
                    source: action.source,
                    accumulator_delta: action.accumulator_delta,
                    lateral_unit_residual: action.lateral_unit_residual,
                    evidence: action.evidence,
                });
            }
        }
    }
    let timer_a80_after = selector.timer_a80_after;
    GroundCorrectionPipelineResult {
        disabled: false,
        selector: Some(selector),
        events,
        timer_a80_after,
        accumulator_614_before,
        accumulator_614_after: force,
        accumulator_620_before,
        accumulator_620_after: accumulator_620_before,
        velocity_c0_before: velocity_c0,
        velocity_c0_after: velocity,
        velocity_direct_write: velocity != velocity_c0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline_args(
        selected_slot: i64,
        opposite_slot: i64,
        field_adc: f64,
        timer_a80_before: f64,
        numeric_mode: &str,
    ) -> GroundCorrectionPipelineResult {
        evaluate_ground_correction_pipeline(
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 0.25],
            [3.0, 0.0, 4.0],
            false,
            selected_slot,
            opposite_slot,
            false,
            false,
            false,
            field_adc,
            timer_a80_before,
            0.002,
            0.0,
            0.0,
            0.0,
            [1.0, -0.5, 0.25],
            1.5,
            1.25,
            false,
            false,
            [0.5, 1.0, -0.5],
            3.0,
            0.5,
            1.5,
            0.75,
            2.0,
            [0.0, 1.0, 0.0],
            1.25,
            0.75,
            numeric_mode,
        )
    }

    #[test]
    fn pipeline_primary_with_kind32_addon_float64() {
        // Constants computed with the Python authority:
        //   evaluate_ground_correction_pipeline(selected_slot=1, field_adc=-1.0,
        //     numeric_mode="float64")
        let r = pipeline_args(1, 0, -1.0, 0.0, "float64");
        assert!(!r.disabled);
        assert_eq!(r.events.len(), 2);
        assert_eq!(r.accumulator_614_after, [3.625, 2.5625, 2.71875]);
        assert!(!r.velocity_direct_write);
    }

    #[test]
    fn pipeline_primary_with_kind32_addon_float32() {
        // Constants computed with the Python authority (same inputs, float32).
        let r = pipeline_args(1, 0, -1.0, 0.0, "float32");
        assert_eq!(r.accumulator_614_after, [3.625, 2.5625, 2.71875]);
    }

    #[test]
    fn pipeline_kind33_fallback() {
        // Constants computed with the Python authority (selected_slot=0,
        // opposite_slot=2, timer_a80_before=0.25).
        let r = pipeline_args(0, 2, 0.0, 0.25, "float64");
        let selector = r.selector.as_ref().unwrap();
        assert_eq!(selector.actions_in_program_order, vec![33]);
        assert_eq!(r.timer_a80_after, 0.252);
        assert_eq!(r.accumulator_614_after, [-0.125, -0.25, 4.125]);
    }

    #[test]
    fn lateral_subtract_kind34() {
        // Constants computed with the Python authority:
        //   evaluate_lateral_subtract(kind=34, velocity_c0=[3,0,4],
        //     axis_ac4=[0,1,0], sub_720=1.25, sub_724=0.75)
        let r = evaluate_lateral_subtract(34, [3.0, 0.0, 4.0], [0.0, 1.0, 0.0], 1.25, 0.75)
            .unwrap();
        assert_eq!(r.source, [0.75, 0.0, 1.0]);
        assert_eq!(r.lateral_unit_residual, [0.6, 0.0, 0.8]);
        // Negating +0.0 keeps the sign bit (Python -0.0).
        assert!(r.accumulator_delta[1].is_sign_negative());
        assert_eq!(r.accumulator_delta, [-0.75, -0.0, -1.0]);
        assert_eq!(
            evaluate_lateral_subtract(33, [3.0, 0.0, 4.0], [0.0, 1.0, 0.0], 1.25, 0.75)
                .unwrap_err(),
            "kind must be 34 or 35"
        );
    }

    #[test]
    fn direction_gain_afc_override() {
        // Constants computed with the Python authority (decoded_state=1,
        // byte_9b8=True, flag_afc=True).
        let r = evaluate_direction_gain(1.0, 1, false, true, true, 2.0, 3.0, 4.0);
        assert_eq!(r.direction_gain, 4.0);
        assert_eq!(r.trace.len(), 3);
        assert_eq!(r.trace[1].step, "state1");
        assert_eq!(r.trace[1].result, 3.0);
        assert_eq!(r.trace[2].step, "afc_override");
    }

    #[test]
    fn state_mult_full_ladder() {
        // Constants computed with the Python authority (decoded_state=10,
        // boost_family=True, attached_state=4, efc=True, flag_5bc=True).
        let r = evaluate_state_mult(
            10, true, 4, false, true, true, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0,
        );
        assert_eq!(r.state_mult, 7.392);
        let steps: Vec<&str> = r.trace.iter().map(|step| step.step).collect();
        assert_eq!(
            steps,
            ["initial", "boost_special_attached", "efc", "state10", "flag_5bc"]
        );
        assert_eq!(r.trace[4].operand, 3.2);
    }
}
