//! One module per Python `scripts/evaluate_*` domain module (same base names).
//! Faithful transliteration over idiom; see crate::num for the numeric rules.
//! (`surface_tune_defaults` is the pure decoded-defaults helper module two of
//! the evaluate_* modules import.)

pub mod airborne_topology;
pub mod angular_velocity_integrator;
pub mod booster_resource;
pub mod booster_vehicle_step;
pub mod branch_aware_vehicle_step;
pub mod bs_zone_override;
pub mod checkpoint_respawn_state_replacement;
pub mod checkpoint_target_selector;
pub mod collision_response_selector;
pub mod common_damping;
pub mod contact_accumulator_producer;
pub mod control_response_input;
pub mod drift_recovery_timer;
pub mod drift_window_branch;
pub mod drive_basis_input;
pub mod drive_target_state;
pub mod event_obstacle_contact;
pub mod fall_clamp_state_replacement;
pub mod ground_accumulator_sink;
pub mod ground_drive_force_body;
pub mod ground_runtime_input;
pub mod ground_state_scalar;
pub mod hard_contact_angular_response;
pub mod hard_contact_velocity_response;
pub mod integrator_dt;
pub mod jm_dj_trigger;
pub mod linear_velocity_integrator;
pub mod low_projection_ground_branch;
pub mod moving_contact_topology;
pub mod mz_magnet_trigger;
pub mod network_correction_topology;
pub mod orientation_integrator;
pub mod pit_resource_step;
pub mod position_integrator;
pub mod post_bs_reset_constraint;
pub mod rail_accumulator_producers;
pub mod rail_base_orientation;
pub mod rail_capture_selector;
pub mod rail_constraint_orientation;
pub mod rail_path_interpolation;
pub mod rail_query_context_selector;
pub mod rail_query_scalar;
pub mod rail_state1_query;
pub mod rail_topology;
pub mod soft_contact_angular_response;
pub mod soft_contact_source;
pub mod soft_contact_velocity_response;
pub mod speed_threshold_latch;
pub mod state18_feedback;
pub mod substep_accumulator_boundary;
pub mod surface_tune_defaults;
pub mod tune_boost_gauge;
