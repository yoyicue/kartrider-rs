//! Deterministic Rust port of the KartRider vehicle physics engine.
//!
//! Authority: `scripts/kartrider_vehicle_engine.py` (+ `evaluate_*` modules).
//! Porting contract: docs/kart-physics-rust-handoff-2026-07-28.md.
//! - Production path is f32-only and must be bit-exact with the Python engine
//!   in `numeric_mode="float32"`.
//! - All cross-substep state lives in `state::VehicleState`; the engine struct
//!   holds only state + params + prediction history.
//! - The clock is integer milliseconds (`u32` ticks / `u64` accumulators).

pub mod domains;
pub mod engine;
pub mod ffi;
pub mod num;
pub mod params;
pub mod rail;
pub mod scheduler;
pub mod state;
pub mod topology;
pub mod world;
