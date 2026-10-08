//! Clean-room parsers for KartRider's `.1s` asset formats (raw models, KartObject
//! track/scene graphs, skeletal + rigid animation, DDS textures, vehicle params).
//!
//! # Malformed-input contract
//!
//! These parsers are faithful, byte-for-byte transliterations of the validated
//! C# reference. Like that reference, they assume the input is a well-formed
//! asset of the format their entry point names: the byte readers index directly
//! and **panic on a truncated or desynced stream** rather than threading a
//! `Result` through every field read (which would obscure the 1:1 correspondence
//! the project validates against). A panic therefore means "these bytes are not
//! the format I was asked to parse," not a bug to handle in-band.
//!
//! The trust boundary is the caller. When feeding bytes that may not be valid
//! (e.g. a whole asset tree of mixed/unknown classes), wrap the public `parse`
//! call in [`std::panic::catch_unwind`] — this is exactly what `kart-pipeline`
//! does per file (and it installs a quiet panic hook once at startup so recovered
//! probes stay silent). The one place a parser deliberately probes-and-recovers
//! internally is the optional trailing-MorphTontroller read, which mirrors the
//! C# `try { ... } catch { rewind }`; it uses `catch_unwind` locally and does
//! **not** touch the process-global panic hook.

pub mod anim_clip;
pub mod bml;
pub mod challenge;
pub mod dds;
pub mod kap;
pub mod kart_anim;
pub mod kart_object;
pub mod keyframe;
pub mod loader;
pub mod param;
pub mod rail;
pub mod random_track;
pub mod raw_model;
pub mod road_corridor;
pub mod track_course;
pub mod track_list;
pub mod track_nav;
