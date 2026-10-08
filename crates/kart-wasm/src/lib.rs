//! `kart-wasm`: the WASM (wasm32-unknown-unknown) boundary for the
//! kart-physics engine plus the kart-track-world `TrackWorld` — W3 of
//! `docs/kart-physics-wasm-demo-handoff-2026-07-29.md`.
//!
//! Red lines honored here (handoff §3):
//!
//! - **No wasm-bindgen.** Plain `extern "C"` exports over linear memory.
//! - **JS never computes a physical number.** JS boundary shapes are exactly
//!   the two handoff forms — `step(keys_bitmask, elapsed_ms)` and
//!   `snapshot() -> pointer/length POD view` — plus asset/spec feed-in and
//!   result/error read-out. All f64s cross as raw little-endian bytes inside
//!   [`KartWasmSnapshot`]; JS copies them out with `DataView.getFloat64(...,
//!   true)` and never writes back.
//! - **All numerics stay in wasm.** The W3 parity gate runs the W2 spec JSON
//!   end-to-end inside the .wasm via [`kart_wasm_run_replay`], which calls
//!   the SAME `kart_track_world::replay::run_live_world_replay`
//!   implementation as the native `live_world_trace` example; the emitted
//!   row bytes are identical in shape, so the existing comparison/budget
//!   logic applies unchanged.
//!
//! # Session model
//!
//! One global session (wasm is single-threaded): assets first
//! ([`kart_wasm_load_world`], optional [`kart_wasm_load_kap`] BEFORE the
//! world), then [`kart_wasm_create_engine`], optional
//! [`kart_wasm_set_parameters`], then the frame loop
//! [`kart_wasm_step`] / [`kart_wasm_snapshot_ptr`]. The W2-style parity
//! replay ([`kart_wasm_run_replay`]) is independent of the session engine.
//!
//! # Memory protocol
//!
//! - Host→wasm buffers: the host calls [`kart_wasm_alloc`], copies its bytes
//!   into the returned pointer, passes pointer+length to an export, and
//!   releases the buffer with [`kart_wasm_free`] after the call returns
//!   (every export copies what it needs before returning).
//! - wasm→host buffers: result/error/snapshot bytes live in session-owned
//!   memory. `kart_wasm_result_ptr/len`, `kart_wasm_last_error_ptr/len`, and
//!   `kart_wasm_snapshot_ptr/len` return stable pointers valid until the
//!   next session-mutating call; the host copies, never frees.
//!
//! # Snapshot POD layout
//!
//! [`KartWasmSnapshot`] is `#[repr(C)]`; its size is reported by
//! [`kart_wasm_snapshot_len`] and the field offsets are pinned by the
//! `layout` unit test (W4 consumes them verbatim). Domain/state label
//! encoding: `domain`, `previous_domain`, `return_domain` carry the
//! `KART_WASM_DOMAIN_*` ids, which match the kart-physics FFI `KP_DOMAIN_*`
//! ids (`MotionDomain` declaration order): 0=GND, 1=GND-SLIP, 2=CONTACT,
//! 3=AIR-ENTRY, 4=AIR, 5=LAND, 6=RAIL, 7=MOVING-CONTACT, 8=RESET, 9=WATER,
//! 10=NET-CORRECTION. `rail_state` is the engine's i64 RAIL substate;
//! `frame_index` increments once per successful [`kart_wasm_step`].
//!
//! # Error model
//!
//! Exports return `KART_WASM_OK` (0) or a `KART_WASM_ERR_*` code; the
//! human-readable message is available via `kart_wasm_last_error_ptr/len`.
//! Rust panics are NOT caught (wasm traps); the crate is written to return
//! errors instead of panicking on all host-driven inputs.

use std::cell::RefCell;
use std::path::PathBuf;

use kart_physics::domains::tune_boost_gauge::current_family_gauge_snapshot;
use kart_physics::engine::{KartRiderVehicleEngine, NumericMode};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use kart_physics::topology::MotionDomain;
use kart_physics::world::World;
use kart_track_world::world::TrackWorld;

// ---------------------------------------------------------------------------
// ABI constants
// ---------------------------------------------------------------------------

/// kart-wasm ABI version; bump on any layout or signature change. v4
/// tail-appends GoItemBanana B9C state. v3 tail-appends ordinary-N2O state and adds
/// `KART_WASM_KEY_ITEM_USE`.
pub const KART_WASM_ABI_VERSION: u32 = 4;
/// Deterministic snapshot field bytes. ABI v4 has no implicit padding: the
/// explicit reserved alignment byte is initialized to zero and is hashed.
pub const KART_WASM_SNAPSHOT_FIELD_LEN: usize = 312;

/// Status: success.
pub const KART_WASM_OK: i32 = 0;
/// Status: a required pointer argument was NULL (with a nonzero length).
pub const KART_WASM_ERR_NULL: i32 = 1;
/// Status: initial state failed JSON decode or `VehicleState.validate`.
pub const KART_WASM_ERR_INVALID_STATE: i32 = 2;
/// Status: parameters failed JSON decode or `VehicleParameters.validate`.
pub const KART_WASM_ERR_INVALID_PARAMS: i32 = 3;
/// Status: inputs invalid (unknown key bits, unknown numeric mode, non-UTF-8
/// path/spec, or a call made before its prerequisite).
pub const KART_WASM_ERR_INVALID_INPUT: i32 = 4;
/// Status: the track world failed to load (bad bytes, unparseable track).
pub const KART_WASM_ERR_WORLD: i32 = 5;
/// Status: the engine step itself returned an error (message has details).
pub const KART_WASM_ERR_STEP: i32 = 7;
/// Status: no engine exists yet (`kart_wasm_create_engine` required first).
pub const KART_WASM_ERR_NO_ENGINE: i32 = 9;

/// Key bit: throttle.
pub const KART_WASM_KEY_UP: u32 = 1 << 0;
/// Key bit: brake/reverse.
pub const KART_WASM_KEY_DOWN: u32 = 1 << 1;
/// Key bit: steer left.
pub const KART_WASM_KEY_LEFT: u32 = 1 << 2;
/// Key bit: steer right.
pub const KART_WASM_KEY_RIGHT: u32 = 1 << 3;
/// Key bit: manual restart (R).
pub const KART_WASM_KEY_R: u32 = 1 << 4;
/// Key bit: drift selector (either Control key).
pub const KART_WASM_KEY_CTRL: u32 = 1 << 5;
/// Key bit: consume one current-family ordinary-N2O inventory item.
pub const KART_WASM_KEY_ITEM_USE: u32 = 1 << 6;
/// All recognized key bits; anything outside is `KART_WASM_ERR_INVALID_INPUT`.
pub const KART_WASM_KEY_MASK_ALL: u32 = KART_WASM_KEY_UP
    | KART_WASM_KEY_DOWN
    | KART_WASM_KEY_LEFT
    | KART_WASM_KEY_RIGHT
    | KART_WASM_KEY_R
    | KART_WASM_KEY_CTRL
    | KART_WASM_KEY_ITEM_USE;

/// Numeric mode: production float32 (bit-exact with the Python engine).
pub const KART_WASM_NUMERIC_MODE_FLOAT32: u32 = 0;
/// Numeric mode: research float64.
pub const KART_WASM_NUMERIC_MODE_FLOAT64: u32 = 1;

/// Motion domain ids (match the kart-physics FFI `KP_DOMAIN_*` ids, i.e.
/// `MotionDomain` declaration order).
pub const KART_WASM_DOMAIN_GND: u8 = 0;
pub const KART_WASM_DOMAIN_GND_SLIP: u8 = 1;
pub const KART_WASM_DOMAIN_CONTACT: u8 = 2;
pub const KART_WASM_DOMAIN_AIR_ENTRY: u8 = 3;
pub const KART_WASM_DOMAIN_AIR: u8 = 4;
pub const KART_WASM_DOMAIN_LAND: u8 = 5;
pub const KART_WASM_DOMAIN_RAIL: u8 = 6;
pub const KART_WASM_DOMAIN_MOVING_CONTACT: u8 = 7;
pub const KART_WASM_DOMAIN_RESET: u8 = 8;
pub const KART_WASM_DOMAIN_WATER: u8 = 9;
pub const KART_WASM_DOMAIN_NET_CORRECTION: u8 = 10;

fn domain_to_u8(domain: MotionDomain) -> u8 {
    match domain {
        MotionDomain::Ground => KART_WASM_DOMAIN_GND,
        MotionDomain::GroundSlip => KART_WASM_DOMAIN_GND_SLIP,
        MotionDomain::Contact => KART_WASM_DOMAIN_CONTACT,
        MotionDomain::AirEntry => KART_WASM_DOMAIN_AIR_ENTRY,
        MotionDomain::Air => KART_WASM_DOMAIN_AIR,
        MotionDomain::Land => KART_WASM_DOMAIN_LAND,
        MotionDomain::Rail => KART_WASM_DOMAIN_RAIL,
        MotionDomain::MovingContact => KART_WASM_DOMAIN_MOVING_CONTACT,
        MotionDomain::Reset => KART_WASM_DOMAIN_RESET,
        MotionDomain::Water => KART_WASM_DOMAIN_WATER,
        MotionDomain::NetworkCorrection => KART_WASM_DOMAIN_NET_CORRECTION,
    }
}

fn keys_from_mask(mask: u32) -> Result<Vec<String>, String> {
    if mask & !KART_WASM_KEY_MASK_ALL != 0 {
        return Err(format!(
            "unknown key bits 0x{:x} (recognized mask 0x{KART_WASM_KEY_MASK_ALL:x})",
            mask & !KART_WASM_KEY_MASK_ALL
        ));
    }
    let mut keys = Vec::new();
    for (bit, name) in [
        (KART_WASM_KEY_UP, "UP"),
        (KART_WASM_KEY_DOWN, "DOWN"),
        (KART_WASM_KEY_LEFT, "LEFT"),
        (KART_WASM_KEY_RIGHT, "RIGHT"),
        (KART_WASM_KEY_R, "R"),
        (KART_WASM_KEY_CTRL, "CTRL"),
        (KART_WASM_KEY_ITEM_USE, "ITEM_USE"),
    ] {
        if mask & bit != 0 {
            keys.push(name.to_string());
        }
    }
    Ok(keys)
}

// ---------------------------------------------------------------------------
// Snapshot POD
// ---------------------------------------------------------------------------

/// The per-frame POD state view exported to JS (see module docs). All
/// multi-byte fields are native little-endian (wasm32 is LE); JS reads them
/// with `DataView.getFloat64/getBigInt64(offset, true)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KartWasmSnapshot {
    /// Number of successful `kart_wasm_step` calls since engine creation.
    pub frame_index: u64,
    /// Kart body position (world units), `state.position`.
    pub position: [f64; 3],
    /// `state.velocity`.
    pub velocity: [f64; 3],
    /// `state.angular_velocity`.
    pub angular_velocity: [f64; 3],
    /// `state.matrix` — 3x3 body orientation, engine row order.
    pub matrix: [f64; 9],
    /// Engine clock, `state.time_ms` (integer milliseconds, exact).
    pub time_ms: u64,
    /// `state.topology.rail_state_elapsed_ms`.
    pub rail_state_elapsed_ms: f64,
    /// `state.topology.transition_index`.
    pub transition_index: i64,
    /// `state.topology.domain_step_index`.
    pub domain_step_index: i64,
    /// `state.topology.rail_state` (RAIL substate; meaningful when
    /// `domain == KART_WASM_DOMAIN_RAIL`).
    pub rail_state: i64,
    /// `state.boost_state`.
    pub boost_state: i64,
    /// `state.boost_remaining_ms`.
    pub boost_remaining_ms: i64,
    /// `KART_WASM_DOMAIN_*` id of `state.topology.domain`.
    pub domain: u8,
    /// Whether `previous_domain` is valid.
    pub has_previous_domain: u8,
    /// `KART_WASM_DOMAIN_*` id (valid when `has_previous_domain != 0`).
    pub previous_domain: u8,
    /// Whether `return_domain` is valid.
    pub has_return_domain: u8,
    /// `KART_WASM_DOMAIN_*` id (valid when `has_return_domain != 0`).
    pub return_domain: u8,
    /// Drift-window selector (`VehicleState.selector_b08`).
    pub selector_b08: u8,
    /// Current-family ordinary-N2O item-use rising-edge latch.
    pub item_use_key_latched: u8,
    /// Explicit deterministic alignment byte. Must remain zero.
    pub reserved_v3_alignment: u8,
    /// Fixed current-manager slot item ids (index0, index1).
    pub booster_inventory_item_ids: [i64; 2],
    /// Fixed current-manager slot quantities (index0, index1).
    pub booster_inventory_quantities: [i64; 2],
    /// Raw cross-substep gauge fields.
    pub boost_gauge_committed_b10: f64,
    pub boost_gauge_progress_b20: f64,
    pub boost_gauge_pending_b2c: f64,
    pub boost_gauge_history_b38: f64,
    /// Rust-authority display amount: f32-clamped B10+B2C.
    pub boost_gauge_display: f64,
    /// Rust-authority f32 cap derived from the active parameters' raw760.
    pub boost_gauge_cap: f64,
    /// GoItemBanana +0xB9C remaining duration in integer milliseconds.
    pub banana_effect_remaining_ms: i64,
    /// Scoped authored event owner (1762/1818), or -1 while inactive.
    pub banana_effect_event_index: i64,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Session {
    world: Option<TrackWorld>,
    engine: Option<KartRiderVehicleEngine>,
    parameters: Option<VehicleParameters>,
    /// KAP payloads registered before the world load (`kart_wasm_load_kap`).
    pending_kaps: Vec<(PathBuf, Vec<u8>)>,
    /// Last replay/result document bytes (session-owned; see module docs).
    result: Vec<u8>,
    /// Last error message bytes (session-owned).
    last_error: Vec<u8>,
    snapshot: KartWasmSnapshot,
}

thread_local! {
    static SESSION: RefCell<Session> = RefCell::new(Session::default());
}

fn set_error(session: &mut Session, message: String) {
    session.last_error = message.into_bytes();
}

fn fail(session: &mut Session, code: i32, message: impl Into<String>) -> i32 {
    set_error(session, message.into());
    code
}

/// Copies the host bytes into a Rust value; every export funnels through
/// this so no session state ever aliases host memory.
unsafe fn host_slice<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], i32> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(KART_WASM_ERR_NULL);
    }
    // SAFETY: the host contract (module docs) guarantees `ptr..ptr+len` is a
    // valid readable allocation for the duration of the call.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn refresh_snapshot(session: &mut Session) -> Result<(), String> {
    let Some(engine) = &session.engine else {
        return Ok(());
    };
    let state = &engine.state;
    let topology = &state.topology;
    let raw_760 = session.parameters.as_ref().map_or_else(
        || VehicleParameters::default().boost_gauge_raw_760,
        |parameters| parameters.boost_gauge_raw_760,
    );
    let gauge = current_family_gauge_snapshot(
        state.boost_gauge_committed_b10,
        state.boost_gauge_pending_b2c,
        raw_760,
    )?;
    session.snapshot.position = state.position;
    session.snapshot.velocity = state.velocity;
    session.snapshot.angular_velocity = state.angular_velocity;
    session.snapshot.matrix = state.matrix;
    session.snapshot.time_ms = state.time_ms;
    session.snapshot.rail_state_elapsed_ms = topology.rail_state_elapsed_ms;
    session.snapshot.transition_index = topology.transition_index;
    session.snapshot.domain_step_index = topology.domain_step_index;
    session.snapshot.rail_state = topology.rail_state;
    session.snapshot.boost_state = state.boost_state;
    session.snapshot.boost_remaining_ms = state.boost_remaining_ms;
    session.snapshot.domain = domain_to_u8(topology.domain);
    session.snapshot.has_previous_domain = u8::from(topology.previous_domain.is_some());
    session.snapshot.previous_domain = topology.previous_domain.map_or(0, domain_to_u8);
    session.snapshot.has_return_domain = u8::from(topology.return_domain.is_some());
    session.snapshot.return_domain = topology.return_domain.map_or(0, domain_to_u8);
    session.snapshot.selector_b08 = u8::from(state.selector_b08);
    session.snapshot.item_use_key_latched = u8::from(state.item_use_key_latched);
    session.snapshot.reserved_v3_alignment = 0;
    let [slot_0, slot_1] = state.booster_inventory_slots.as_slice() else {
        return Err(format!(
            "booster_inventory_slots must contain exactly 2 slots, got {}",
            state.booster_inventory_slots.len()
        ));
    };
    session.snapshot.booster_inventory_item_ids = [slot_0.item_id, slot_1.item_id];
    session.snapshot.booster_inventory_quantities = [slot_0.quantity, slot_1.quantity];
    session.snapshot.boost_gauge_committed_b10 = state.boost_gauge_committed_b10;
    session.snapshot.boost_gauge_progress_b20 = state.boost_gauge_progress_b20;
    session.snapshot.boost_gauge_pending_b2c = state.boost_gauge_pending_b2c;
    session.snapshot.boost_gauge_history_b38 = state.boost_gauge_history_b38;
    session.snapshot.boost_gauge_display = gauge.accum_clamped;
    session.snapshot.boost_gauge_cap = gauge.cap;
    session.snapshot.banana_effect_remaining_ms = state.banana_effect_remaining_ms;
    session.snapshot.banana_effect_event_index = match state.banana_effect_owner_id.as_deref() {
        Some("banana:0xF0000039") => 1762,
        Some("banana:0xF000003B") => 1818,
        _ => -1,
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory protocol
// ---------------------------------------------------------------------------

/// Allocates `len` bytes of wasm linear memory for host→wasm transfer.
/// Free with [`kart_wasm_free`] (same `len`).
///
/// # Safety
/// The returned pointer must be freed exactly once with the same length.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let Ok(layout) = std::alloc::Layout::array::<u8>(len) else {
        return std::ptr::null_mut();
    };
    // SAFETY: `layout` has nonzero size (checked above).
    unsafe { std::alloc::alloc(layout) }
}

/// Releases a buffer from [`kart_wasm_alloc`].
///
/// # Safety
/// `ptr` must come from `kart_wasm_alloc(len)` with the same `len`.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let Ok(layout) = std::alloc::Layout::array::<u8>(len) else {
        return;
    };
    // SAFETY: per the host contract, `ptr`/`len` name a live allocation from
    // `kart_wasm_alloc`.
    unsafe { std::alloc::dealloc(ptr, layout) }
}

/// ABI version handshake (`KART_WASM_ABI_VERSION`).
#[no_mangle]
pub extern "C" fn kart_wasm_abi_version() -> u32 {
    KART_WASM_ABI_VERSION
}

// ---------------------------------------------------------------------------
// Asset loading (bytes in; the browser fetched them, there is no fs)
// ---------------------------------------------------------------------------

/// Loads the track world from `.track.1s` bytes. `path` is the identity
/// string the bytes are registered under — for parity with a native
/// `TrackWorld::from_path` run it must be the exact `scenario.track` string.
/// Applies any KAPs staged via [`kart_wasm_load_kap`]. Replaces any previous
/// world; the engine (if any) is left untouched and keeps its old world
/// reference, so recreate the engine after reloading the world.
///
/// # Safety
/// `track_ptr`/`track_len` and `path_ptr`/`path_len` must name valid
/// host-written buffers (see the module memory protocol); both are copied
/// before this call returns.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_load_world(
    track_ptr: *const u8,
    track_len: usize,
    path_ptr: *const u8,
    path_len: usize,
) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        let track = match unsafe { host_slice(track_ptr, track_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty track bytes"),
            Err(code) => return fail(&mut session, code, "null track bytes pointer"),
        };
        let path_bytes = match unsafe { host_slice(path_ptr, path_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty track path"),
            Err(code) => return fail(&mut session, code, "null track path pointer"),
        };
        let path = match std::str::from_utf8(path_bytes) {
            Ok(text) => PathBuf::from(text),
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_INPUT,
                    format!("track path is not UTF-8: {error}"),
                )
            }
        };
        let kaps = std::mem::take(&mut session.pending_kaps);
        let world = match TrackWorld::from_bytes(&path, track)
            .and_then(|world| world.with_kap_bytes(kaps))
        {
            Ok(world) => world,
            Err(message) => return fail(&mut session, KART_WASM_ERR_WORLD, message),
        };
        session.world = Some(world);
        set_error(&mut session, String::new());
        KART_WASM_OK
    })
}

/// Stages one KAP event-marker payload (checkpoint selectors) under its
/// identity path. MUST be called before [`kart_wasm_load_world`]; the
/// payloads are applied to the world when it loads.
///
/// # Safety
/// `kap_ptr`/`kap_len` and `path_ptr`/`path_len` must name valid
/// host-written buffers (see the module memory protocol); both are copied
/// before this call returns.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_load_kap(
    kap_ptr: *const u8,
    kap_len: usize,
    path_ptr: *const u8,
    path_len: usize,
) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        if session.world.is_some() {
            return fail(
                &mut session,
                KART_WASM_ERR_INVALID_INPUT,
                "kart_wasm_load_kap must be called before kart_wasm_load_world",
            );
        }
        let kap = match unsafe { host_slice(kap_ptr, kap_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes.to_vec(),
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty KAP bytes"),
            Err(code) => return fail(&mut session, code, "null KAP bytes pointer"),
        };
        let path_bytes = match unsafe { host_slice(path_ptr, path_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty KAP path"),
            Err(code) => return fail(&mut session, code, "null KAP path pointer"),
        };
        let path = match std::str::from_utf8(path_bytes) {
            Ok(text) => PathBuf::from(text),
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_INPUT,
                    format!("KAP path is not UTF-8: {error}"),
                )
            }
        };
        session.pending_kaps.push((path, kap));
        KART_WASM_OK
    })
}

// ---------------------------------------------------------------------------
// Engine lifecycle + the two handoff boundary shapes
// ---------------------------------------------------------------------------

/// Creates the engine from a `VehicleState` JSON document (the exact serde
/// shape used by the parity specs, e.g. a spec's `initial_state`) and a
/// `KART_WASM_NUMERIC_MODE_*` value. Requires a loaded world.
///
/// # Safety
/// `state_json_ptr`/`state_json_len` must name a valid host-written buffer
/// (see the module memory protocol); it is copied before this call returns.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_create_engine(
    state_json_ptr: *const u8,
    state_json_len: usize,
    numeric_mode: u32,
) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        if session.world.is_none() {
            return fail(
                &mut session,
                KART_WASM_ERR_INVALID_INPUT,
                "kart_wasm_load_world must succeed before kart_wasm_create_engine",
            );
        }
        let mode = match numeric_mode {
            KART_WASM_NUMERIC_MODE_FLOAT32 => NumericMode::Float32,
            KART_WASM_NUMERIC_MODE_FLOAT64 => NumericMode::Float64,
            other => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_INPUT,
                    format!("unknown numeric mode {other}"),
                )
            }
        };
        let bytes = match unsafe { host_slice(state_json_ptr, state_json_len) } {
            Ok(bytes) => bytes,
            Err(code) => return fail(&mut session, code, "null state JSON pointer"),
        };
        let state: VehicleState = match serde_json::from_slice(bytes) {
            Ok(state) => state,
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_STATE,
                    format!("state JSON decode: {error}"),
                )
            }
        };
        match KartRiderVehicleEngine::new(&state, mode) {
            Ok(engine) => {
                session.engine = Some(engine);
                session.snapshot = KartWasmSnapshot::default();
                if let Err(message) = refresh_snapshot(&mut session) {
                    session.engine = None;
                    return fail(&mut session, KART_WASM_ERR_INVALID_STATE, message);
                }
                set_error(&mut session, String::new());
                KART_WASM_OK
            }
            Err(message) => fail(&mut session, KART_WASM_ERR_INVALID_STATE, message),
        }
    })
}

/// Replaces the per-frame vehicle parameters with a `VehicleParameters` JSON
/// document (the exact serde shape carried per step in the parity specs —
/// pass the captured vehicle config here). Optional: before the first call
/// the engine steps with `VehicleParameters::default()` (the captured
/// reference configuration).
///
/// # Safety
/// `params_json_ptr`/`params_json_len` must name a valid host-written buffer
/// (see the module memory protocol); it is copied before this call returns.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_set_parameters(
    params_json_ptr: *const u8,
    params_json_len: usize,
) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        let bytes = match unsafe { host_slice(params_json_ptr, params_json_len) } {
            Ok(bytes) => bytes,
            Err(code) => return fail(&mut session, code, "null parameters JSON pointer"),
        };
        let parameters: VehicleParameters = match serde_json::from_slice(bytes) {
            Ok(parameters) => parameters,
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_PARAMS,
                    format!("parameters JSON decode: {error}"),
                )
            }
        };
        if let Err(message) = parameters.validate() {
            return fail(&mut session, KART_WASM_ERR_INVALID_PARAMS, message);
        }
        session.parameters = Some(parameters);
        if let Err(message) = refresh_snapshot(&mut session) {
            return fail(&mut session, KART_WASM_ERR_INVALID_STATE, message);
        }
        set_error(&mut session, String::new());
        KART_WASM_OK
    })
}

/// Handoff boundary shape #1: advance one game frame. `keys_bitmask` is
/// `KART_WASM_KEY_*`; `elapsed_ms` is the u32 game-clock tick — the
/// accumulated whole milliseconds since engine start (wrapping at 2^32 like
/// the engine scheduler; the fractional remainder stays host-side for the
/// next frame, exactly as handoff §2-W4 prescribes). The first call after
/// creation establishes the clock without advancing physics (the
/// `step_raw_frame` contract). On success the snapshot is refreshed and
/// `frame_index` increments.
#[no_mangle]
pub extern "C" fn kart_wasm_step(keys_bitmask: u32, elapsed_ms: u32) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        let keys = match keys_from_mask(keys_bitmask) {
            Ok(keys) => keys,
            Err(message) => return fail(&mut session, KART_WASM_ERR_INVALID_INPUT, message),
        };
        if session.engine.is_none() {
            return fail(
                &mut session,
                KART_WASM_ERR_NO_ENGINE,
                "kart_wasm_create_engine must succeed before kart_wasm_step",
            );
        }
        // Split the RefMut into disjoint field borrows: the engine steps
        // mutably while the world is shared immutably.
        let Session {
            engine,
            world,
            parameters,
            ..
        } = &mut *session;
        let parameters = parameters
            .clone()
            .unwrap_or_else(VehicleParameters::default);
        let world: Option<&dyn World> = world.as_ref().map(|w| w as &dyn World);
        let engine = engine.as_mut().expect("checked above");
        match engine.step_raw_frame(elapsed_ms, &keys, &parameters, world, None, None) {
            Ok(_outcome) => {
                if let Err(message) = refresh_snapshot(&mut session) {
                    return fail(&mut session, KART_WASM_ERR_INVALID_STATE, message);
                }
                session.snapshot.frame_index += 1;
                set_error(&mut session, String::new());
                KART_WASM_OK
            }
            Err(message) => fail(&mut session, KART_WASM_ERR_STEP, message),
        }
    })
}

/// Handoff boundary shape #2: pointer to the session-owned
/// [`KartWasmSnapshot`] POD view. Valid until the next session-mutating
/// call; JS copies, never writes back. Null when no engine exists.
#[no_mangle]
pub extern "C" fn kart_wasm_snapshot_ptr() -> *const u8 {
    SESSION.with(|cell| {
        let session = cell.borrow();
        if session.engine.is_none() {
            return std::ptr::null();
        }
        std::ptr::from_ref(&session.snapshot).cast::<u8>()
    })
}

/// Size of [`KartWasmSnapshot`] in bytes (the host should assert this equals
/// its expected layout size before reading fields).
#[no_mangle]
pub extern "C" fn kart_wasm_snapshot_len() -> usize {
    std::mem::size_of::<KartWasmSnapshot>()
}

/// Number of deterministic POD field bytes to hash/compare. ABI v4 defines
/// every byte, so this equals [`kart_wasm_snapshot_len`].
#[no_mangle]
pub extern "C" fn kart_wasm_snapshot_field_len() -> usize {
    KART_WASM_SNAPSHOT_FIELD_LEN
}

// ---------------------------------------------------------------------------
// W3 parity-gate replay (spec in, result rows out — all numerics in wasm)
// ---------------------------------------------------------------------------

/// Runs one W2-style live-world parity spec end-to-end INSIDE wasm: decodes
/// the spec JSON, registers the track bytes under `path` (which must equal
/// the spec's `scenario.track` string), and executes the shared
/// `kart_track_world::replay::run_live_world_replay` implementation — the
/// same function the native `live_world_trace` example runs. The emitted
/// `{"rows": [...], "world_response_diff": {...}}` JSON is stored as the
/// session result; read it with `kart_wasm_result_ptr/len`. Independent of
/// the session engine/world.
///
/// # Safety
/// `spec_ptr`/`spec_len`, `track_ptr`/`track_len`, and `path_ptr`/`path_len`
/// must name valid host-written buffers (see the module memory protocol);
/// all three are copied before this call returns.
#[no_mangle]
pub unsafe extern "C" fn kart_wasm_run_replay(
    spec_ptr: *const u8,
    spec_len: usize,
    track_ptr: *const u8,
    track_len: usize,
    path_ptr: *const u8,
    path_len: usize,
) -> i32 {
    SESSION.with(|cell| {
        let mut session = cell.borrow_mut();
        let spec = match unsafe { host_slice(spec_ptr, spec_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty spec bytes"),
            Err(code) => return fail(&mut session, code, "null spec pointer"),
        };
        let spec_text = match std::str::from_utf8(spec) {
            Ok(text) => text,
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_INPUT,
                    format!("spec is not UTF-8: {error}"),
                )
            }
        };
        let track = match unsafe { host_slice(track_ptr, track_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty track bytes"),
            Err(code) => return fail(&mut session, code, "null track bytes pointer"),
        };
        let path_bytes = match unsafe { host_slice(path_ptr, path_len) } {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => return fail(&mut session, KART_WASM_ERR_NULL, "empty track path"),
            Err(code) => return fail(&mut session, code, "null track path pointer"),
        };
        let path = match std::str::from_utf8(path_bytes) {
            Ok(text) => PathBuf::from(text),
            Err(error) => {
                return fail(
                    &mut session,
                    KART_WASM_ERR_INVALID_INPUT,
                    format!("track path is not UTF-8: {error}"),
                )
            }
        };
        if let Err(message) = kart_track_world::track::register_virtual_bytes(&path, track) {
            return fail(&mut session, KART_WASM_ERR_WORLD, message);
        }
        let output = kart_track_world::replay::run_live_world_replay(spec_text);
        session.result = output.into_bytes();
        set_error(&mut session, String::new());
        KART_WASM_OK
    })
}

/// Pointer to the last result document (`kart_wasm_run_replay` output).
/// Session-owned; valid until the next replay call.
#[no_mangle]
pub extern "C" fn kart_wasm_result_ptr() -> *const u8 {
    SESSION.with(|cell| cell.borrow().result.as_ptr())
}

/// Length of the last result document in bytes.
#[no_mangle]
pub extern "C" fn kart_wasm_result_len() -> usize {
    SESSION.with(|cell| cell.borrow().result.len())
}

/// Pointer to the last error message (UTF-8). Session-owned.
#[no_mangle]
pub extern "C" fn kart_wasm_last_error_ptr() -> *const u8 {
    SESSION.with(|cell| cell.borrow().last_error.as_ptr())
}

/// Length of the last error message in bytes.
#[no_mangle]
pub extern "C" fn kart_wasm_last_error_len() -> usize {
    SESSION.with(|cell| cell.borrow().last_error.len())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The snapshot POD layout is contractual for W4 — pin every offset.
    #[test]
    fn snapshot_layout() {
        assert_eq!(size_of::<KartWasmSnapshot>(), 312);
        assert_eq!(KART_WASM_SNAPSHOT_FIELD_LEN, 312);
        assert_eq!(offset_of!(KartWasmSnapshot, frame_index), 0);
        assert_eq!(offset_of!(KartWasmSnapshot, position), 8);
        assert_eq!(offset_of!(KartWasmSnapshot, velocity), 32);
        assert_eq!(offset_of!(KartWasmSnapshot, angular_velocity), 56);
        assert_eq!(offset_of!(KartWasmSnapshot, matrix), 80);
        assert_eq!(offset_of!(KartWasmSnapshot, time_ms), 152);
        assert_eq!(offset_of!(KartWasmSnapshot, rail_state_elapsed_ms), 160);
        assert_eq!(offset_of!(KartWasmSnapshot, transition_index), 168);
        assert_eq!(offset_of!(KartWasmSnapshot, domain_step_index), 176);
        assert_eq!(offset_of!(KartWasmSnapshot, rail_state), 184);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_state), 192);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_remaining_ms), 200);
        assert_eq!(offset_of!(KartWasmSnapshot, domain), 208);
        assert_eq!(offset_of!(KartWasmSnapshot, has_previous_domain), 209);
        assert_eq!(offset_of!(KartWasmSnapshot, previous_domain), 210);
        assert_eq!(offset_of!(KartWasmSnapshot, has_return_domain), 211);
        assert_eq!(offset_of!(KartWasmSnapshot, return_domain), 212);
        assert_eq!(offset_of!(KartWasmSnapshot, selector_b08), 213);
        assert_eq!(offset_of!(KartWasmSnapshot, item_use_key_latched), 214);
        assert_eq!(offset_of!(KartWasmSnapshot, reserved_v3_alignment), 215);
        assert_eq!(
            offset_of!(KartWasmSnapshot, booster_inventory_item_ids),
            216
        );
        assert_eq!(
            offset_of!(KartWasmSnapshot, booster_inventory_quantities),
            232
        );
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_committed_b10), 248);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_progress_b20), 256);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_pending_b2c), 264);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_history_b38), 272);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_display), 280);
        assert_eq!(offset_of!(KartWasmSnapshot, boost_gauge_cap), 288);
        assert_eq!(
            offset_of!(KartWasmSnapshot, banana_effect_remaining_ms),
            296
        );
        assert_eq!(offset_of!(KartWasmSnapshot, banana_effect_event_index), 304);
    }

    #[test]
    fn domain_ids_match_motion_domain_declaration_order() {
        for (domain, id) in [
            (MotionDomain::Ground, KART_WASM_DOMAIN_GND),
            (MotionDomain::GroundSlip, KART_WASM_DOMAIN_GND_SLIP),
            (MotionDomain::Contact, KART_WASM_DOMAIN_CONTACT),
            (MotionDomain::AirEntry, KART_WASM_DOMAIN_AIR_ENTRY),
            (MotionDomain::Air, KART_WASM_DOMAIN_AIR),
            (MotionDomain::Land, KART_WASM_DOMAIN_LAND),
            (MotionDomain::Rail, KART_WASM_DOMAIN_RAIL),
            (MotionDomain::MovingContact, KART_WASM_DOMAIN_MOVING_CONTACT),
            (MotionDomain::Reset, KART_WASM_DOMAIN_RESET),
            (MotionDomain::Water, KART_WASM_DOMAIN_WATER),
            (
                MotionDomain::NetworkCorrection,
                KART_WASM_DOMAIN_NET_CORRECTION,
            ),
        ] {
            assert_eq!(domain_to_u8(domain), id);
        }
    }

    #[test]
    fn key_mask_mapping_matches_engine_names() {
        assert!(keys_from_mask(KART_WASM_KEY_MASK_ALL + 1).is_err());
        assert_eq!(
            keys_from_mask(KART_WASM_KEY_UP | KART_WASM_KEY_LEFT).unwrap(),
            ["UP", "LEFT"]
        );
        assert_eq!(
            keys_from_mask(KART_WASM_KEY_MASK_ALL).unwrap(),
            ["UP", "DOWN", "LEFT", "RIGHT", "R", "CTRL", "ITEM_USE"]
        );
        assert!(keys_from_mask(0).unwrap().is_empty());
    }

    #[test]
    fn snapshot_exposes_drift_and_current_family_n2o_tail() {
        let mut state = VehicleState {
            selector_b08: true,
            item_use_key_latched: true,
            boost_gauge_committed_b10: 60.0,
            boost_gauge_progress_b20: 0.25,
            boost_gauge_pending_b2c: 50.0,
            boost_gauge_history_b38: 7.0,
            ..Default::default()
        };
        state.booster_inventory_slots[0].item_id = 6;
        state.booster_inventory_slots[0].quantity = 1;
        let engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let mut session = Session {
            engine: Some(engine),
            parameters: Some(VehicleParameters {
                boost_gauge_raw_760: 100.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        refresh_snapshot(&mut session).unwrap();
        assert_eq!(session.snapshot.selector_b08, 1);
        assert_eq!(session.snapshot.item_use_key_latched, 1);
        assert_eq!(session.snapshot.reserved_v3_alignment, 0);
        assert_eq!(session.snapshot.booster_inventory_item_ids, [6, 0xffff]);
        assert_eq!(session.snapshot.booster_inventory_quantities, [1, 0]);
        assert_eq!(session.snapshot.boost_gauge_committed_b10, 60.0);
        assert_eq!(session.snapshot.boost_gauge_progress_b20, 0.25);
        assert_eq!(session.snapshot.boost_gauge_pending_b2c, 50.0);
        assert_eq!(session.snapshot.boost_gauge_history_b38, 7.0);
        assert_eq!(session.snapshot.boost_gauge_display, 100.0);
        assert_eq!(session.snapshot.boost_gauge_cap, 100.0);
    }

    #[test]
    fn alloc_free_roundtrip() {
        unsafe {
            let ptr = kart_wasm_alloc(4096);
            assert!(!ptr.is_null());
            std::ptr::write_bytes(ptr, 0xAB, 4096);
            kart_wasm_free(ptr, 4096);
            assert!(kart_wasm_alloc(0).is_null());
            kart_wasm_free(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn session_call_order_is_enforced() {
        assert_eq!(kart_wasm_step(0, 0), KART_WASM_ERR_NO_ENGINE);
        assert!(kart_wasm_snapshot_ptr().is_null());
        assert_eq!(kart_wasm_snapshot_len(), size_of::<KartWasmSnapshot>());
        assert_eq!(kart_wasm_snapshot_field_len(), KART_WASM_SNAPSHOT_FIELD_LEN);
        let state = serde_json::to_vec(&VehicleState::default()).unwrap();
        // No world loaded yet.
        assert_eq!(
            unsafe {
                kart_wasm_create_engine(state.as_ptr(), state.len(), KART_WASM_NUMERIC_MODE_FLOAT32)
            },
            KART_WASM_ERR_INVALID_INPUT
        );
        assert_eq!(
            unsafe { kart_wasm_create_engine(state.as_ptr(), state.len(), 99) },
            KART_WASM_ERR_INVALID_INPUT
        );
    }
}
