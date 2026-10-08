//! kart-rho — clean-room reader for KartRider `.rho`/`.rho5` archives.
//!
//! Milestone roadmap (see ../../README.md):
//!   M1a  crypto primitives (this) — key derivation, XOR + header ciphers
//!   M1b  .rho unpack — header, block-info table, folder/file tree, zlib blocks
//!   M1c  .rho5 unpack
//!
//! The crypto layer is verified against KartLibrary's formulas via unit tests.

pub mod archive;
pub mod crypto;
pub mod encoded_block;
pub mod kr_data;
pub mod pack;
pub mod pin;
pub mod rho5_archive;
pub mod rho5_key;
pub mod rho5_keyprovider;
mod rho5_tables;
mod vectors;
