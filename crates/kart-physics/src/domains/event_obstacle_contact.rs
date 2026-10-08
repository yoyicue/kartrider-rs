//! Bit-exact port of `scripts/evaluate_event_obstacle_contact.py`.
//!
//! This is the P1-closed factory_I02 event-1743 `GoItemObstacle` CONTACT
//! writer.  It is neither the static wall response nor MOVING-CONTACT
//! support ownership.

use crate::num::{
    dot3_f32, f32q, fadd, fdiv, fmul, length3_f32, min2_f64, sub3_f32,
};

const OBSTACLE_MOTION_PHI: f64 = 1.618_000_030_517_578_1;
const OBSTACLE_TANGENT_SCALE: f64 = 0.600_000_023_841_857_9;
const OBSTACLE_NORMAL_SCALE: f64 = 1.5;
const OBSTACLE_NORMAL_RESPONSE: f64 = -1.5;
const OBSTACLE_SPECIAL_VELOCITY_DIVISOR: f64 = 2.0;
const OBSTACLE_SPECIAL_DOT_THRESHOLD: f64 = -0.7;
const OBSTACLE_SIDE_NORMAL_Z_MAX: f64 = 0.65;

fn f32_vec3(value: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    let result = [f32q(value[0]), f32q(value[1]), f32q(value[2])];
    if result.iter().all(|component| component.is_finite()) {
        Ok(result)
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn mul3_f32(value: [f64; 3], scalar: f64) -> [f64; 3] {
    [
        fmul(value[0], scalar),
        fmul(value[1], scalar),
        fmul(value[2], scalar),
    ]
}

fn normalize3_f32(value: [f64; 3]) -> [f64; 3] {
    let length = length3_f32(value);
    if length == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [
            fdiv(value[0], length),
            fdiv(value[1], length),
            fdiv(value[2], length),
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventObstacleContactSource {
    pub branch: &'static str,
    pub motion_normal_dot_f32: f64,
    pub approaching_dot_f32: f64,
    pub approaching_gate: bool,
    pub side_normal_z_gate: bool,
    pub normal_component: [f64; 3],
    pub tangent: [f64; 3],
    pub tangent_length_f32: f64,
    pub normal_component_length_f32: f64,
    pub response_scale_f32: f64,
    pub velocity_source: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventObstacleContactResponse {
    pub source: EventObstacleContactSource,
    pub selected: bool,
    pub velocity_before: [f64; 3],
    pub velocity_after: [f64; 3],
}

pub fn evaluate_event_obstacle_contact_source(
    velocity: [f64; 3],
    normal: [f64; 3],
    obstacle_motion: [f64; 3],
) -> Result<EventObstacleContactSource, String> {
    let velocity = f32_vec3(velocity, "velocity")?;
    let normal = f32_vec3(normal, "normal")?;
    let obstacle_motion = f32_vec3(obstacle_motion, "obstacle_motion")?;
    let projected_motion = [obstacle_motion[0], obstacle_motion[1], f32q(0.0)];
    let projected_normal = [normal[0], normal[1], f32q(0.0)];
    let motion_normal_dot = dot3_f32(projected_motion, projected_normal);
    let special = motion_normal_dot < f32q(OBSTACLE_SPECIAL_DOT_THRESHOLD);
    let (projection_velocity, tangent_velocity) = if special {
        let divided = [
            fdiv(velocity[0], OBSTACLE_SPECIAL_VELOCITY_DIVISOR),
            fdiv(velocity[1], OBSTACLE_SPECIAL_VELOCITY_DIVISOR),
            fdiv(velocity[2], OBSTACLE_SPECIAL_VELOCITY_DIVISOR),
        ];
        (divided, divided)
    } else {
        (
            sub3_f32(
                velocity,
                mul3_f32(obstacle_motion, OBSTACLE_MOTION_PHI),
            ),
            velocity,
        )
    };
    let normal_dot = dot3_f32(projection_velocity, normal);
    let normal_component = mul3_f32(normal, normal_dot);
    let tangent = sub3_f32(tangent_velocity, normal_component);
    let tangent_length = length3_f32(tangent);
    let normal_length = length3_f32(normal_component);
    let tangent_direction = normalize3_f32(tangent);
    let response_scale = min2_f64(
        fmul(tangent_length, OBSTACLE_TANGENT_SCALE),
        fmul(normal_length, OBSTACLE_NORMAL_SCALE),
    );
    let mut source = sub3_f32(
        mul3_f32(normal_component, OBSTACLE_NORMAL_RESPONSE),
        mul3_f32(tangent_direction, response_scale),
    );
    source = sub3_f32(
        source,
        mul3_f32([0.0, 0.0, 1.0], source[2]),
    );
    Ok(EventObstacleContactSource {
        branch: if special {
            "moving_special"
        } else {
            "ordinary_relative"
        },
        motion_normal_dot_f32: motion_normal_dot,
        approaching_dot_f32: normal_dot,
        approaching_gate: normal_dot < 0.0,
        side_normal_z_gate: normal[2] <= f32q(OBSTACLE_SIDE_NORMAL_Z_MAX),
        normal_component,
        tangent,
        tangent_length_f32: tangent_length,
        normal_component_length_f32: normal_length,
        response_scale_f32: response_scale,
        velocity_source: source,
    })
}

pub fn apply_event_obstacle_contact_source(
    velocity: [f64; 3],
    normal: [f64; 3],
    obstacle_motion: [f64; 3],
) -> Result<EventObstacleContactResponse, String> {
    let velocity_before = f32_vec3(velocity, "velocity")?;
    let source =
        evaluate_event_obstacle_contact_source(velocity, normal, obstacle_motion)?;
    let selected = source.approaching_gate && source.side_normal_z_gate;
    let velocity_after = if selected {
        [
            fadd(velocity_before[0], source.velocity_source[0]),
            fadd(velocity_before[1], source.velocity_source[1]),
            fadd(velocity_before[2], source.velocity_source[2]),
        ]
    } else {
        velocity_before
    };
    Ok(EventObstacleContactResponse {
        source,
        selected,
        velocity_before,
        velocity_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run16_primary_n2o_impact_is_bit_exact() {
        let response = apply_event_obstacle_contact_source(
            [34.823_501_6, -22.845_993, 0.015_695_085_7],
            [-0.904_148_161, 0.427_219_033, 0.000_005_808_697_1],
            [-12.069_702_1, 4.006_778_72, 0.0],
        )
        .unwrap();
        assert!(response.selected);
        assert_eq!(
            response.source.velocity_source,
            [
                f32q(-71.078_994_8),
                f32q(37.420_455_9),
                f32q(0.0)
            ]
        );
        assert_eq!(
            response.velocity_after,
            [
                f32q(-36.255_493_2),
                f32q(14.574_462_9),
                f32q(0.015_695_085_7)
            ]
        );
    }
}
