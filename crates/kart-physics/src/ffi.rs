//! Frozen C ABI (milestone M4, docs/kart-physics-rust-handoff-2026-07-28.md).
//!
//! Everything that crosses the boundary is `#[repr(C)]` POD; the cdylib stays
//! dependency-free (no serde). The header `include/kart_physics.h` is
//! generated with cbindgen (see `cbindgen.toml`) and checked in.
//!
//! # Design decisions
//!
//! - **Error model.** Functions return a status code (`KP_OK` / `KP_ERR_*`);
//!   a thread-local message is readable via
//!   [`kart_physics_last_error_message`]. No Rust panic may cross the FFI:
//!   every exported function wraps its body in
//!   `std::panic::catch_unwind` and reports `KP_ERR_PANIC`.
//! - **Bools** cross as `u8` (0/1), **`Option<T>`** as a `has_*: u8`
//!   presence tag next to the payload (outer-Option presence semantics
//!   preserved exactly; absent slots carry zeroed payloads).
//! - **Strings** cross as [`KpString64`]: `len` (bytes) + 64-byte inline
//!   buffer, no NUL required. Strings longer than 64 bytes are rejected on
//!   import (`KP_ERR_STRING_TOO_LONG`); export fails the same way rather than
//!   truncating. All strings the engine itself produces (mode families,
//!   authorities, marker/contact/object ids) are short ASCII identifiers.
//! - **Dict-typed latches** are full POD mirrors: `rail_selector_runtime`
//!   (11 known keys + `path_context_id` int-or-string),
//!   `moving_contact_anchor`, and `network_correction` (including the nested
//!   `resume_topology`) have fixed key sets and mirror directly. The single
//!   genuinely open-ended payload — `rail_capture_latch.selector` (a verbatim
//!   `world.query_rail_contact` mapping) — crosses as a JSON document inside
//!   a fixed [`KpJsonBlob`] ([`KP_JSON_CAPACITY`] bytes), serialized/parsed
//!   by the small dependency-free codec in this module; f64 numbers
//!   round-trip bit-exact (Rust's shortest `{:?}` formatting +
//!   correctly-rounded parsing), and non-finite numbers are rejected
//!   (`KP_ERR_JSON`). The same blob carries
//!   `CheckpointRespawnCandidate.course_context_id` and the open-mapping
//!   world callback payloads below.
//! - **World** crosses as a function-pointer vtable ([`KpWorldVtable`]) plus
//!   a `user_data` void pointer. `query_ground` is required (passing a
//!   vtable with a NULL `query_ground` is `KP_ERR_INVALID_INPUT`); every
//!   optional capability is an `Option<fn>` where NULL means *capability
//!   absent*, preserving the contractual `hasattr` semantics (handoff §4).
//!   Capabilities whose payloads are open JSON mappings — `sweep_point`,
//!   `query_rail_contact`, `rail_query_contexts`, `rail_entry_contexts`,
//!   `query_moving_contact`, `query_network_correction`,
//!   `query_distinct_water_volume` — cross as JSON documents in
//!   [`KpJsonBlob`] out-params; scalar arguments (position, time_ms,
//!   plane_tolerance, …) cross as POD parameters. The closed
//!   `query_rail_entry_boundary` match crosses as the typed
//!   [`KpRailEntryBoundaryMatch`] mirror. List-returning callbacks
//!   (`sweep_obb_rotating_all`, `overlap_obb_all`, `rail_query_contexts`,
//!   `rail_entry_contexts`) follow the two-call protocol documented on
//!   [`KpWorldVtable`], with the callback-reported count clamped to the
//!   documented ceilings ([`KP_LIST_MAX_COUNT`] /
//!   [`KP_LIST_MAX_BUFFER_BYTES`]) — a garbage count used to abort the host
//!   via `handle_alloc_error`; now the adapter reports the query as failed
//!   instead (the engine surfaces its usual "capability reported present
//!   but returned no result" error).
//! - **ABI versioning.** [`KP_ABI_VERSION`] is bumped on any layout or
//!   signature change. The vtable is host-allocated, so the library cannot
//!   trust a host to pass the layout its own header describes: a host
//!   compiled against the v1 header owns only the 80-byte v1 prefix of the
//!   144-byte v2 [`KpWorldVtable`], the 6448-byte v1 state, and the 536-byte
//!   v1 parameter block. Reading current layouts from any of those pointers
//!   would be out of bounds. The handshake is therefore negotiated at
//!   creation: current hosts MUST create engines with
//!   [`kart_physics_create_v2`], passing their header's `KP_ABI_VERSION`
//!   (mismatch is refused with `KP_ERR_ABI_VERSION`); the returned engine
//!   then reads current layouts in every state/parameter/vtable operation.
//!   Plain [`kart_physics_create`] is the documented v1 compatibility shim:
//!   its engines translate exactly [`KP_VEHICLE_STATE_V1_BYTES`] of state,
//!   copy exactly [`KP_VEHICLE_PARAMETERS_V1_BYTES`] of parameters, and read
//!   exactly [`KP_WORLD_VTABLE_V1_BYTES`] of the world vtable. The larger
//!   JSON slot and current-only tails are never touched in v1 host memory.
//!   Current hosts use the versioned default writers
//!   [`kart_physics_state_default_v2`] and
//!   [`kart_physics_parameters_default_v2`]; the unsuffixed writers are
//!   frozen v1-only entry points. Hosts SHOULD additionally keep the cheap
//!   self-check (`kart_physics_abi_version()` equals their header's
//!   `KP_ABI_VERSION`; `examples/c_smoke` fails loudly).
//! - **Inputs** cross as [`KpInputs`]: `tick_ms` (u32 game clock, wraps at
//!   2^32 like the engine scheduler) plus a key bitmask (`KP_KEY_*`; the
//!   engine's recognized key set used at the ABI is
//!   UP/DOWN/LEFT/RIGHT/R/CTRL/ITEM_USE).
//!   Unknown bits are rejected (`KP_ERR_INVALID_INPUT`).
//! - **Entry points.** `create` mirrors `KartRiderVehicleEngine::new`
//!   (initial state + numeric mode); `kart_physics_step` mirrors
//!   `step_raw_frame(tick_ms, keys, parameters, world)` with the frame
//!   parameters passed per call, exactly like the Rust API. Passing a NULL
//!   world is the legal `world=None` path and surfaces the same engine
//!   errors as Rust.
//! - **Threading.** Handles are not thread-safe; one engine per thread, no
//!   concurrent calls on the same handle. World callbacks run on the calling
//!   thread and must not longjmp/throw back into Rust.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::domains::booster_resource::InventorySlot;
use crate::engine::{KartRiderVehicleEngine, NumericMode};
use crate::params::{BoostParameters, VehicleParameters};
use crate::state::{
    JsonValue, MovingContactAnchor, NetworkCorrectionState, RailCaptureLatch, RailPathContextId,
    RailSelectorRuntime, RigidStateSnapshot, VehicleState,
};
use crate::topology::{MotionDomain, TopologyState};
use crate::world::{
    CheckpointRespawnCandidate, CheckpointTargetCandidate, RailEntryBoundaryMatch,
    TakeoffLandingContact, World, WorldCapabilities, WorldGroundResult, WorldObbOverlap,
};

// ---------------------------------------------------------------------------
// Constants (mirrored into the C header by cbindgen)
// ---------------------------------------------------------------------------

/// ABI version; bumped on any layout or signature change. v4 (2026-08-01)
/// tail-appends the current-family ordinary-N2O inventory/gauge state and
/// tuning fields and adds `KP_KEY_ITEM_USE`. Existing v3 offsets remain
/// unchanged.
///
/// v3 (2026-07-30)
/// tail-appends the drift selector/timer state and tuning fields and adds
/// `KP_KEY_CTRL`. Existing v2 field offsets remain unchanged. ABI v1 used a
/// 4096-byte JSON slot, so the compatibility shim translates its historical
/// state layout rather than treating it as a current-struct prefix.
///
/// v2 (2026-07-29):
/// `KP_JSON_CAPACITY` 4096 -> 65536 (real `query_rail_contact` selectors
/// embed the full `registered_query_contexts` list — 40354 bytes measured
/// on factory_I02) and `KpWorldVtable` grew the 8 open-mapping capability
/// slots. v2 hosts MUST create engines with
/// [`kart_physics_create_v2`]`(initial_state, numeric_mode, KP_ABI_VERSION)`;
/// plain [`kart_physics_create`] is the v1 compatibility shim whose engines
/// never read past the v1 vtable prefix.
pub const KP_ABI_VERSION: u32 = 5;

/// Status: success.
pub const KP_OK: i32 = 0;
/// Status: a required pointer argument was NULL.
pub const KP_ERR_NULL: i32 = 1;
/// Status: state blob failed conversion or `VehicleState.validate`.
pub const KP_ERR_INVALID_STATE: i32 = 2;
/// Status: parameters failed conversion or `VehicleParameters.validate`.
pub const KP_ERR_INVALID_PARAMS: i32 = 3;
/// Status: inputs/vtable invalid (unknown key bits, NULL `query_ground`,
/// unknown numeric mode).
pub const KP_ERR_INVALID_INPUT: i32 = 4;
/// Status: a string exceeded [`KP_STRING_CAPACITY`] bytes.
pub const KP_ERR_STRING_TOO_LONG: i32 = 5;
/// Status: a JSON blob exceeded [`KP_JSON_CAPACITY`] bytes, held invalid
/// JSON, or a non-finite number.
pub const KP_ERR_JSON: i32 = 6;
/// Status: the engine step itself returned an error (message has details).
pub const KP_ERR_STEP: i32 = 7;
/// Status: a Rust panic was caught at the boundary (message has details).
pub const KP_ERR_PANIC: i32 = 8;
/// Status: [`kart_physics_create_v2`] was called with an `abi_version`
/// other than [`KP_ABI_VERSION`] (host/library ABI mismatch).
pub const KP_ERR_ABI_VERSION: i32 = 9;

/// `KpString64` inline capacity in bytes.
pub const KP_STRING_CAPACITY: u32 = 64;
/// `KpJsonBlob` inline capacity in bytes. Sized against real world
/// payloads: the largest observed `query_rail_contact` selector (which
/// embeds the full `registered_query_contexts` list) measures 40354 bytes
/// on factory_I02 (largest single `rail_query_contexts` element: 2492
/// bytes; `rail_entry_contexts` element: 532 bytes); 65536 leaves ~1.6x
/// headroom. Payloads exceeding the capacity fail with `KP_ERR_JSON`,
/// never truncate.
pub const KP_JSON_CAPACITY: u32 = 65536;

/// Exact byte size of the frozen ABI v1 `KpVehicleState`.
///
/// ABI v1 used a 4096-byte inline JSON slot. Its fields after
/// `rail_capture_latch` therefore do not share offsets with the current
/// struct and are translated by the v1 compatibility shim.
pub const KP_VEHICLE_STATE_V1_BYTES: usize = 6448;
/// Exact byte size of the ABI v2 `KpVehicleState` (the current state before
/// the ABI v3 drift tail was appended).
pub const KP_VEHICLE_STATE_V2_BYTES: usize = 67888;
/// Exact byte size of the frozen ABI v1/v2 `KpVehicleParameters` prefix.
pub const KP_VEHICLE_PARAMETERS_V1_BYTES: usize = 536;
/// Exact byte size of the frozen ABI v3 `KpVehicleState` prefix.
pub const KP_VEHICLE_STATE_V3_BYTES: usize = 67984;
/// Exact byte size of the frozen ABI v3 `KpVehicleParameters` prefix.
pub const KP_VEHICLE_PARAMETERS_V3_BYTES: usize = 592;

/// Key bit: throttle.
pub const KP_KEY_UP: u32 = 1 << 0;
/// Key bit: brake/reverse.
pub const KP_KEY_DOWN: u32 = 1 << 1;
/// Key bit: steer left.
pub const KP_KEY_LEFT: u32 = 1 << 2;
/// Key bit: steer right.
pub const KP_KEY_RIGHT: u32 = 1 << 3;
/// Key bit: manual restart (R).
pub const KP_KEY_R: u32 = 1 << 4;
/// Key bit: drift selector (either Control key).
pub const KP_KEY_CTRL: u32 = 1 << 5;
/// Key bit: consume one current-family ordinary-N2O inventory item.
pub const KP_KEY_ITEM_USE: u32 = 1 << 6;
/// All recognized key bits; anything outside is `KP_ERR_INVALID_INPUT`.
pub const KP_KEY_MASK_ALL: u32 =
    KP_KEY_UP | KP_KEY_DOWN | KP_KEY_LEFT | KP_KEY_RIGHT | KP_KEY_R | KP_KEY_CTRL | KP_KEY_ITEM_USE;

/// Numeric mode: production float32 (bit-exact with the Python engine).
pub const KP_NUMERIC_MODE_FLOAT32: u32 = 0;
/// Numeric mode: research float64.
pub const KP_NUMERIC_MODE_FLOAT64: u32 = 1;

/// Motion domain ids (match `MotionDomain` declaration order).
pub const KP_DOMAIN_GND: u8 = 0;
pub const KP_DOMAIN_GND_SLIP: u8 = 1;
pub const KP_DOMAIN_CONTACT: u8 = 2;
pub const KP_DOMAIN_AIR_ENTRY: u8 = 3;
pub const KP_DOMAIN_AIR: u8 = 4;
pub const KP_DOMAIN_LAND: u8 = 5;
pub const KP_DOMAIN_RAIL: u8 = 6;
pub const KP_DOMAIN_MOVING_CONTACT: u8 = 7;
pub const KP_DOMAIN_RESET: u8 = 8;
pub const KP_DOMAIN_WATER: u8 = 9;
pub const KP_DOMAIN_NET_CORRECTION: u8 = 10;

/// Stack fast-path candidate capacity for the list-returning world
/// callbacks (`sweep_obb_rotating_all`, `overlap_obb_all`).
const LIST_STACK_CAPACITY: usize = 64;

/// First-call buffer capacity (in elements) for the JSON-blob list
/// callbacks (`rail_query_contexts`, `rail_entry_contexts`); real maps
/// register a few dozen contexts (factory_I02: 38).
const LIST_STACK_CAPACITY_JSON: usize = 8;

/// Documented ceilings for the two-call list protocol (handoff
/// 2026-07-29 §3.1): a callback-reported total count above
/// `KP_LIST_MAX_COUNT`, or one whose element buffer would exceed
/// `KP_LIST_MAX_BUFFER_BYTES`, is rejected — the adapter reports the query
/// as failed (outer `None`), which surfaces the engine's usual "capability
/// reported present but returned no result" error instead of aborting the
/// host process on a garbage count (`handle_alloc_error` is not covered by
/// `catch_unwind`).
const KP_LIST_MAX_COUNT: usize = 1 << 20;
const KP_LIST_MAX_BUFFER_BYTES: usize = 1 << 28;

/// Returns the element count to allocate for the second call of the
/// two-call list protocol, or `None` when the callback-reported `count`
/// exceeds the documented ceilings.
fn clamped_list_count<T>(count: usize) -> Option<usize> {
    if count > KP_LIST_MAX_COUNT {
        return None;
    }
    let bytes = count.checked_mul(std::mem::size_of::<T>())?;
    if bytes > KP_LIST_MAX_BUFFER_BYTES {
        return None;
    }
    Some(count)
}

fn domain_to_u8(domain: MotionDomain) -> u8 {
    match domain {
        MotionDomain::Ground => KP_DOMAIN_GND,
        MotionDomain::GroundSlip => KP_DOMAIN_GND_SLIP,
        MotionDomain::Contact => KP_DOMAIN_CONTACT,
        MotionDomain::AirEntry => KP_DOMAIN_AIR_ENTRY,
        MotionDomain::Air => KP_DOMAIN_AIR,
        MotionDomain::Land => KP_DOMAIN_LAND,
        MotionDomain::Rail => KP_DOMAIN_RAIL,
        MotionDomain::MovingContact => KP_DOMAIN_MOVING_CONTACT,
        MotionDomain::Reset => KP_DOMAIN_RESET,
        MotionDomain::Water => KP_DOMAIN_WATER,
        MotionDomain::NetworkCorrection => KP_DOMAIN_NET_CORRECTION,
    }
}

fn domain_from_u8(value: u8) -> Result<MotionDomain, String> {
    Ok(match value {
        KP_DOMAIN_GND => MotionDomain::Ground,
        KP_DOMAIN_GND_SLIP => MotionDomain::GroundSlip,
        KP_DOMAIN_CONTACT => MotionDomain::Contact,
        KP_DOMAIN_AIR_ENTRY => MotionDomain::AirEntry,
        KP_DOMAIN_AIR => MotionDomain::Air,
        KP_DOMAIN_LAND => MotionDomain::Land,
        KP_DOMAIN_RAIL => MotionDomain::Rail,
        KP_DOMAIN_MOVING_CONTACT => MotionDomain::MovingContact,
        KP_DOMAIN_RESET => MotionDomain::Reset,
        KP_DOMAIN_WATER => MotionDomain::Water,
        KP_DOMAIN_NET_CORRECTION => MotionDomain::NetworkCorrection,
        other => return Err(format!("unknown motion domain id {other}")),
    })
}

fn bool_u8(value: bool) -> u8 {
    u8::from(value)
}

fn u8_bool(value: u8) -> bool {
    value != 0
}

// ---------------------------------------------------------------------------
// POD primitives
// ---------------------------------------------------------------------------

/// Fixed-capacity inline string: `len` bytes in `data` (no NUL required).
/// Maximum 64 bytes; longer strings are rejected, never truncated.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpString64 {
    pub len: u8,
    pub data: [u8; KP_STRING_CAPACITY as usize],
}

impl Default for KpString64 {
    fn default() -> Self {
        KpString64 {
            len: 0,
            data: [0; KP_STRING_CAPACITY as usize],
        }
    }
}

impl KpString64 {
    fn set(text: &str) -> Result<KpString64, String> {
        if text.len() > KP_STRING_CAPACITY as usize {
            return Err(format!(
                "string exceeds {} bytes: {text:?}",
                KP_STRING_CAPACITY
            ));
        }
        let mut result = KpString64 {
            len: text.len() as u8,
            ..Default::default()
        };
        result.data[..text.len()].copy_from_slice(text.as_bytes());
        Ok(result)
    }

    fn get(&self) -> Result<String, String> {
        if self.len as usize > KP_STRING_CAPACITY as usize {
            return Err(format!("KpString64 len {} out of range", self.len));
        }
        std::str::from_utf8(&self.data[..self.len as usize])
            .map(str::to_string)
            .map_err(|error| format!("KpString64 is not valid UTF-8: {error}"))
    }

    /// C-world import path: invalid UTF-8 degrades to lossy instead of an
    /// error (world trait methods cannot fail).
    fn get_lossy(&self) -> String {
        let len = (self.len as usize).min(KP_STRING_CAPACITY as usize);
        String::from_utf8_lossy(&self.data[..len]).into_owned()
    }
}

/// Fixed-capacity inline JSON document (see module docs). Holds
/// `rail_capture_latch.selector` and
/// `CheckpointRespawnCandidate.course_context_id`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpJsonBlob {
    pub len: u32,
    pub data: [u8; KP_JSON_CAPACITY as usize],
}

impl Default for KpJsonBlob {
    fn default() -> Self {
        KpJsonBlob {
            len: 0,
            data: [0; KP_JSON_CAPACITY as usize],
        }
    }
}

impl KpJsonBlob {
    fn set_json(value: &JsonValue) -> Result<KpJsonBlob, String> {
        let mut bytes = Vec::new();
        write_json_value(value, &mut bytes)?;
        if bytes.len() > KP_JSON_CAPACITY as usize {
            return Err(format!(
                "JSON payload exceeds {} bytes ({} required)",
                KP_JSON_CAPACITY,
                bytes.len()
            ));
        }
        let mut result = KpJsonBlob {
            len: bytes.len() as u32,
            ..Default::default()
        };
        result.data[..bytes.len()].copy_from_slice(&bytes);
        Ok(result)
    }

    fn get_json(&self) -> Result<JsonValue, String> {
        if self.len as usize > KP_JSON_CAPACITY as usize {
            return Err(format!("KpJsonBlob len {} out of range", self.len));
        }
        parse_json_value(&self.data[..self.len as usize])
    }
}

// ---------------------------------------------------------------------------
// Dependency-free JSON codec for `JsonValue`
// ---------------------------------------------------------------------------
//
// Numbers: written with Rust's shortest-round-trip `{:?}` formatting and
// parsed with the correctly-rounded `str::parse::<f64>`, so finite f64
// values round-trip bit-exact (including -0.0 and subnormals). Non-finite
// values are rejected.

fn write_json_string(text: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for ch in text.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0C}' => out.extend_from_slice(b"\\f"),
            ch if (ch as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", ch as u32).as_bytes());
            }
            ch => {
                let mut buffer = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
            }
        }
    }
    out.push(b'"');
}

fn write_json_value(value: &JsonValue, out: &mut Vec<u8>) -> Result<(), String> {
    match value {
        JsonValue::Null => out.extend_from_slice(b"null"),
        JsonValue::Bool(true) => out.extend_from_slice(b"true"),
        JsonValue::Bool(false) => out.extend_from_slice(b"false"),
        JsonValue::Number(number) => {
            if !number.is_finite() {
                return Err("non-finite number is not representable as JSON".to_string());
            }
            out.extend_from_slice(format!("{number:?}").as_bytes());
        }
        JsonValue::String(text) => write_json_string(text, out),
        JsonValue::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_json_value(item, out)?;
            }
            out.push(b']');
        }
        JsonValue::Object(entries) => {
            out.push(b'{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_json_string(key, out);
                out.push(b':');
                write_json_value(item, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn error(&self, message: &str) -> String {
        format!("JSON parse error at byte {}: {message}", self.pos)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected {:?}", byte as char)))
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.bytes.get(self.pos) {
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_whitespace();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                loop {
                    items.push(self.parse_value()?);
                    self.skip_whitespace();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(JsonValue::Array(items));
                        }
                        _ => return Err(self.error("expected ',' or ']'")),
                    }
                }
            }
            Some(b'{') => {
                self.pos += 1;
                let mut entries = BTreeMap::new();
                self.skip_whitespace();
                if self.bytes.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                loop {
                    self.skip_whitespace();
                    let key = self.parse_string()?;
                    self.skip_whitespace();
                    self.expect(b':')?;
                    let value = self.parse_value()?;
                    entries.insert(key, value);
                    self.skip_whitespace();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(JsonValue::Object(entries));
                        }
                        _ => return Err(self.error("expected ',' or '}'")),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            _ => Err(self.error("unexpected byte")),
        }
    }

    fn parse_literal(&mut self, literal: &[u8], value: JsonValue) -> Result<JsonValue, String> {
        if self.bytes.len() - self.pos >= literal.len()
            && &self.bytes[self.pos..self.pos + literal.len()] == literal
        {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.error("truncated \\u escape"));
        }
        let text = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| self.error("invalid \\u escape"))?;
        let value =
            u32::from_str_radix(text, 16).map_err(|_| self.error("invalid \\u escape digits"))?;
        self.pos += 4;
        Ok(value)
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut bytes: Vec<u8> = Vec::new();
        loop {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return Err(self.error("unterminated string"));
            };
            self.pos += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return Err(self.error("unterminated escape"));
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => bytes.push(b'"'),
                        b'\\' => bytes.push(b'\\'),
                        b'/' => bytes.push(b'/'),
                        b'b' => bytes.push(0x08),
                        b'f' => bytes.push(0x0C),
                        b'n' => bytes.push(b'\n'),
                        b'r' => bytes.push(b'\r'),
                        b't' => bytes.push(b'\t'),
                        b'u' => {
                            let first = self.parse_hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                // High surrogate: require a following
                                // low-surrogate escape.
                                if self.bytes.get(self.pos) == Some(&b'\\')
                                    && self.bytes.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let second = self.parse_hex4()?;
                                    if !(0xDC00..0xE000).contains(&second) {
                                        return Err(
                                            self.error("invalid low surrogate in \\u escape")
                                        );
                                    }
                                    0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                                } else {
                                    return Err(self.error("unpaired high surrogate"));
                                }
                            } else if (0xDC00..0xE000).contains(&first) {
                                return Err(self.error("unpaired low surrogate"));
                            } else {
                                first
                            };
                            let ch = char::from_u32(code)
                                .ok_or_else(|| self.error("invalid unicode scalar"))?;
                            let mut buffer = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                0x00..=0x1F => return Err(self.error("unescaped control byte in string")),
                _ => bytes.push(byte),
            }
        }
        String::from_utf8(bytes).map_err(|_| self.error("string is not valid UTF-8"))
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        while matches!(
            self.bytes.get(self.pos),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(self.error("empty number"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error("invalid number"))?;
        let value: f64 = text
            .parse()
            .map_err(|_| self.error(&format!("invalid number {text:?}")))?;
        if !value.is_finite() {
            return Err(self.error("non-finite number"));
        }
        Ok(JsonValue::Number(value))
    }
}

fn parse_json_value(bytes: &[u8]) -> Result<JsonValue, String> {
    let mut parser = JsonParser { bytes, pos: 0 };
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos != bytes.len() {
        return Err(parser.error("trailing bytes after JSON value"));
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Typed rich -> POD export errors (no string sniffing at the boundary)
// ---------------------------------------------------------------------------

/// Typed failure of a rich -> POD export ([`KpVehicleState::to_pod`] and the
/// nested latch mirrors). The FFI boundary maps the variant to a status
/// code (`KP_ERR_INVALID_STATE` / `KP_ERR_STRING_TOO_LONG` / `KP_ERR_JSON`)
/// instead of sniffing substrings out of the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodExportError {
    /// The rich state violates a fixed-shape POD invariant.
    InvalidState(String),
    /// A string exceeded [`KP_STRING_CAPACITY`] bytes.
    StringTooLong(String),
    /// A JSON payload exceeded [`KP_JSON_CAPACITY`] bytes or held a
    /// non-finite number.
    Json(String),
}

impl PodExportError {
    fn status_code(&self) -> i32 {
        match self {
            PodExportError::InvalidState(_) => KP_ERR_INVALID_STATE,
            PodExportError::StringTooLong(_) => KP_ERR_STRING_TOO_LONG,
            PodExportError::Json(_) => KP_ERR_JSON,
        }
    }

    fn into_message(self) -> String {
        match self {
            PodExportError::InvalidState(message)
            | PodExportError::StringTooLong(message)
            | PodExportError::Json(message) => message,
        }
    }
}

impl std::fmt::Display for PodExportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PodExportError::InvalidState(message)
            | PodExportError::StringTooLong(message)
            | PodExportError::Json(message) => formatter.write_str(message),
        }
    }
}

// ---------------------------------------------------------------------------
// POD mirrors: topology + dict-typed latches
// ---------------------------------------------------------------------------

/// POD mirror of [`TopologyState`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpTopologyState {
    /// `KP_DOMAIN_*` id.
    pub domain: u8,
    pub has_previous_domain: u8,
    /// `KP_DOMAIN_*` id (valid when `has_previous_domain != 0`).
    pub previous_domain: u8,
    pub has_return_domain: u8,
    /// `KP_DOMAIN_*` id (valid when `has_return_domain != 0`).
    pub return_domain: u8,
    pub transition_index: i64,
    pub domain_step_index: i64,
    pub rail_state_elapsed_ms: f64,
    pub entry_reason: KpString64,
    pub has_constraint_owner_id: u8,
    pub constraint_owner_id: KpString64,
    pub rail_state: i64,
}

impl KpTopologyState {
    fn to_pod(state: &TopologyState) -> Result<KpTopologyState, String> {
        Ok(KpTopologyState {
            domain: domain_to_u8(state.domain),
            has_previous_domain: bool_u8(state.previous_domain.is_some()),
            previous_domain: state.previous_domain.map_or(0, domain_to_u8),
            has_return_domain: bool_u8(state.return_domain.is_some()),
            return_domain: state.return_domain.map_or(0, domain_to_u8),
            transition_index: state.transition_index,
            domain_step_index: state.domain_step_index,
            rail_state_elapsed_ms: state.rail_state_elapsed_ms,
            entry_reason: KpString64::set(&state.entry_reason)?,
            has_constraint_owner_id: bool_u8(state.constraint_owner_id.is_some()),
            constraint_owner_id: match &state.constraint_owner_id {
                Some(owner) => KpString64::set(owner)?,
                None => KpString64::default(),
            },
            rail_state: state.rail_state,
        })
    }

    fn to_rich(self) -> Result<TopologyState, String> {
        Ok(TopologyState {
            domain: domain_from_u8(self.domain)?,
            previous_domain: if u8_bool(self.has_previous_domain) {
                Some(domain_from_u8(self.previous_domain)?)
            } else {
                None
            },
            return_domain: if u8_bool(self.has_return_domain) {
                Some(domain_from_u8(self.return_domain)?)
            } else {
                None
            },
            transition_index: self.transition_index,
            domain_step_index: self.domain_step_index,
            rail_state_elapsed_ms: self.rail_state_elapsed_ms,
            entry_reason: self.entry_reason.get()?,
            constraint_owner_id: if u8_bool(self.has_constraint_owner_id) {
                Some(self.constraint_owner_id.get()?)
            } else {
                None
            },
            rail_state: self.rail_state,
        })
    }
}

impl Default for KpTopologyState {
    fn default() -> Self {
        KpTopologyState::to_pod(&TopologyState::default()).expect("default topology is POD-able")
    }
}

/// POD mirror of [`RailSelectorRuntime`] (the `rail_selector_runtime`
/// latch): all 11 engine-known keys plus the int-or-string
/// `path_context_id`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpRailSelectorRuntime {
    pub has_alignment_primary_remaining_ms: u8,
    pub alignment_primary_remaining_ms: f64,
    pub has_alignment_stable_remaining_ms: u8,
    pub alignment_stable_remaining_ms: f64,
    /// 0 = absent, 1 = integer context index, 2 = string context id.
    pub path_context_id_tag: u8,
    pub path_context_index: i64,
    pub path_context_name: KpString64,
    pub has_state6_contact_remaining_ms: u8,
    pub state6_contact_remaining_ms: f64,
    pub has_state6_contact_candidate_latched: u8,
    pub state6_contact_candidate_latched: f64,
    pub has_state6_zero_boundary_release_delay_substeps: u8,
    pub state6_zero_boundary_release_delay_substeps: f64,
    pub has_state6_entry_force_pending_x: u8,
    pub state6_entry_force_pending_x: f64,
    pub has_state6_entry_force_pending_y: u8,
    pub state6_entry_force_pending_y: f64,
    pub has_state6_entry_force_pending_z: u8,
    pub state6_entry_force_pending_z: f64,
    pub has_state6_entry_force_delay_substeps: u8,
    pub state6_entry_force_delay_substeps: f64,
    pub has_state6_entry_common_tail_delay_substeps: u8,
    pub state6_entry_common_tail_delay_substeps: f64,
}

macro_rules! option_f64_slot {
    ($pod:ident, $rich:ident, $field:ident, $has:ident) => {
        $pod.$has = bool_u8($rich.$field.is_some());
        $pod.$field = $rich.$field.unwrap_or(0.0);
    };
}

macro_rules! read_option_f64_slot {
    ($pod:ident, $field:ident, $has:ident) => {
        if u8_bool($pod.$has) {
            Some($pod.$field)
        } else {
            None
        }
    };
}

impl KpRailSelectorRuntime {
    fn to_pod(runtime: &RailSelectorRuntime) -> Result<KpRailSelectorRuntime, String> {
        let mut pod = KpRailSelectorRuntime {
            has_alignment_primary_remaining_ms: 0,
            alignment_primary_remaining_ms: 0.0,
            has_alignment_stable_remaining_ms: 0,
            alignment_stable_remaining_ms: 0.0,
            path_context_id_tag: 0,
            path_context_index: 0,
            path_context_name: KpString64::default(),
            has_state6_contact_remaining_ms: 0,
            state6_contact_remaining_ms: 0.0,
            has_state6_contact_candidate_latched: 0,
            state6_contact_candidate_latched: 0.0,
            has_state6_zero_boundary_release_delay_substeps: 0,
            state6_zero_boundary_release_delay_substeps: 0.0,
            has_state6_entry_force_pending_x: 0,
            state6_entry_force_pending_x: 0.0,
            has_state6_entry_force_pending_y: 0,
            state6_entry_force_pending_y: 0.0,
            has_state6_entry_force_pending_z: 0,
            state6_entry_force_pending_z: 0.0,
            has_state6_entry_force_delay_substeps: 0,
            state6_entry_force_delay_substeps: 0.0,
            has_state6_entry_common_tail_delay_substeps: 0,
            state6_entry_common_tail_delay_substeps: 0.0,
        };
        option_f64_slot!(
            pod,
            runtime,
            alignment_primary_remaining_ms,
            has_alignment_primary_remaining_ms
        );
        option_f64_slot!(
            pod,
            runtime,
            alignment_stable_remaining_ms,
            has_alignment_stable_remaining_ms
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_contact_remaining_ms,
            has_state6_contact_remaining_ms
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_contact_candidate_latched,
            has_state6_contact_candidate_latched
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_zero_boundary_release_delay_substeps,
            has_state6_zero_boundary_release_delay_substeps
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_entry_force_pending_x,
            has_state6_entry_force_pending_x
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_entry_force_pending_y,
            has_state6_entry_force_pending_y
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_entry_force_pending_z,
            has_state6_entry_force_pending_z
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_entry_force_delay_substeps,
            has_state6_entry_force_delay_substeps
        );
        option_f64_slot!(
            pod,
            runtime,
            state6_entry_common_tail_delay_substeps,
            has_state6_entry_common_tail_delay_substeps
        );
        match &runtime.path_context_id {
            None => {}
            Some(RailPathContextId::Index(index)) => {
                pod.path_context_id_tag = 1;
                pod.path_context_index = *index;
            }
            Some(RailPathContextId::Name(name)) => {
                pod.path_context_id_tag = 2;
                pod.path_context_name = KpString64::set(name)?;
            }
        }
        Ok(pod)
    }

    fn to_rich(self) -> Result<RailSelectorRuntime, String> {
        let path_context_id = match self.path_context_id_tag {
            0 => None,
            1 => Some(RailPathContextId::Index(self.path_context_index)),
            2 => Some(RailPathContextId::Name(self.path_context_name.get()?)),
            other => return Err(format!("unknown path_context_id tag {other}")),
        };
        Ok(RailSelectorRuntime {
            alignment_primary_remaining_ms: read_option_f64_slot!(
                self,
                alignment_primary_remaining_ms,
                has_alignment_primary_remaining_ms
            ),
            alignment_stable_remaining_ms: read_option_f64_slot!(
                self,
                alignment_stable_remaining_ms,
                has_alignment_stable_remaining_ms
            ),
            path_context_id,
            state6_contact_remaining_ms: read_option_f64_slot!(
                self,
                state6_contact_remaining_ms,
                has_state6_contact_remaining_ms
            ),
            state6_contact_candidate_latched: read_option_f64_slot!(
                self,
                state6_contact_candidate_latched,
                has_state6_contact_candidate_latched
            ),
            state6_zero_boundary_release_delay_substeps: read_option_f64_slot!(
                self,
                state6_zero_boundary_release_delay_substeps,
                has_state6_zero_boundary_release_delay_substeps
            ),
            state6_entry_force_pending_x: read_option_f64_slot!(
                self,
                state6_entry_force_pending_x,
                has_state6_entry_force_pending_x
            ),
            state6_entry_force_pending_y: read_option_f64_slot!(
                self,
                state6_entry_force_pending_y,
                has_state6_entry_force_pending_y
            ),
            state6_entry_force_pending_z: read_option_f64_slot!(
                self,
                state6_entry_force_pending_z,
                has_state6_entry_force_pending_z
            ),
            state6_entry_force_delay_substeps: read_option_f64_slot!(
                self,
                state6_entry_force_delay_substeps,
                has_state6_entry_force_delay_substeps
            ),
            state6_entry_common_tail_delay_substeps: read_option_f64_slot!(
                self,
                state6_entry_common_tail_delay_substeps,
                has_state6_entry_common_tail_delay_substeps
            ),
        })
    }
}

impl Default for KpRailSelectorRuntime {
    fn default() -> Self {
        KpRailSelectorRuntime::to_pod(&RailSelectorRuntime::default())
            .expect("default rail selector runtime is POD-able")
    }
}

/// POD mirror of [`RailCaptureLatch`]. The open-ended `selector` mapping
/// crosses as a JSON document in `selector_json` (see module docs).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpRailCaptureLatch {
    pub contact_active: u8,
    pub has_rail_state: u8,
    pub rail_state: i64,
    pub has_authority: u8,
    pub authority: KpString64,
    /// JSON encoding of the open `selector` mapping (empty object `{}`
    /// when the mapping is empty).
    pub selector_json: KpJsonBlob,
}

impl KpRailCaptureLatch {
    fn to_pod(latch: &RailCaptureLatch) -> Result<KpRailCaptureLatch, PodExportError> {
        Ok(KpRailCaptureLatch {
            contact_active: bool_u8(latch.contact_active),
            has_rail_state: bool_u8(latch.rail_state.is_some()),
            rail_state: latch.rail_state.unwrap_or(0),
            has_authority: bool_u8(latch.authority.is_some()),
            authority: match &latch.authority {
                Some(authority) => {
                    KpString64::set(authority).map_err(PodExportError::StringTooLong)?
                }
                None => KpString64::default(),
            },
            selector_json: KpJsonBlob::set_json(&JsonValue::Object(latch.selector.clone()))
                .map_err(PodExportError::Json)?,
        })
    }

    fn to_rich(self) -> Result<RailCaptureLatch, String> {
        let selector = match self.selector_json.get_json()? {
            JsonValue::Object(entries) => entries,
            _ => return Err("rail_capture_latch selector_json must hold an object".to_string()),
        };
        Ok(RailCaptureLatch {
            contact_active: u8_bool(self.contact_active),
            rail_state: if u8_bool(self.has_rail_state) {
                Some(self.rail_state)
            } else {
                None
            },
            selector,
            authority: if u8_bool(self.has_authority) {
                Some(self.authority.get()?)
            } else {
                None
            },
        })
    }
}

impl Default for KpRailCaptureLatch {
    fn default() -> Self {
        KpRailCaptureLatch::to_pod(&RailCaptureLatch::default())
            .expect("default rail capture latch is POD-able")
    }
}

/// POD mirror of [`MovingContactAnchor`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpMovingContactAnchor {
    pub object_id: KpString64,
    pub local_position: [f64; 3],
    pub local_velocity: [f64; 3],
    pub relative_matrix: [f64; 9],
    pub local_angular_velocity: [f64; 3],
    pub local_contact_normal: [f64; 3],
    pub local_plane_offset: f64,
    pub has_discarded_normal_velocity: u8,
    pub discarded_normal_velocity: f64,
    pub has_object_point_velocity: u8,
    pub object_point_velocity: [f64; 3],
    pub has_kinematic_contract: u8,
    pub kinematic_contract: KpString64,
}

impl KpMovingContactAnchor {
    fn to_pod(anchor: &MovingContactAnchor) -> Result<KpMovingContactAnchor, String> {
        Ok(KpMovingContactAnchor {
            object_id: KpString64::set(&anchor.object_id)?,
            local_position: anchor.local_position,
            local_velocity: anchor.local_velocity,
            relative_matrix: anchor.relative_matrix,
            local_angular_velocity: anchor.local_angular_velocity,
            local_contact_normal: anchor.local_contact_normal,
            local_plane_offset: anchor.local_plane_offset,
            has_discarded_normal_velocity: bool_u8(anchor.discarded_normal_velocity.is_some()),
            discarded_normal_velocity: anchor.discarded_normal_velocity.unwrap_or(0.0),
            has_object_point_velocity: bool_u8(anchor.object_point_velocity.is_some()),
            object_point_velocity: anchor.object_point_velocity.unwrap_or([0.0; 3]),
            has_kinematic_contract: bool_u8(anchor.kinematic_contract.is_some()),
            kinematic_contract: match &anchor.kinematic_contract {
                Some(contract) => KpString64::set(contract)?,
                None => KpString64::default(),
            },
        })
    }

    fn to_rich(self) -> Result<MovingContactAnchor, String> {
        Ok(MovingContactAnchor {
            object_id: self.object_id.get()?,
            local_position: self.local_position,
            local_velocity: self.local_velocity,
            relative_matrix: self.relative_matrix,
            local_angular_velocity: self.local_angular_velocity,
            local_contact_normal: self.local_contact_normal,
            local_plane_offset: self.local_plane_offset,
            discarded_normal_velocity: if u8_bool(self.has_discarded_normal_velocity) {
                Some(self.discarded_normal_velocity)
            } else {
                None
            },
            object_point_velocity: if u8_bool(self.has_object_point_velocity) {
                Some(self.object_point_velocity)
            } else {
                None
            },
            kinematic_contract: if u8_bool(self.has_kinematic_contract) {
                Some(self.kinematic_contract.get()?)
            } else {
                None
            },
        })
    }
}

impl Default for KpMovingContactAnchor {
    fn default() -> Self {
        KpMovingContactAnchor::to_pod(&MovingContactAnchor::default())
            .expect("default moving contact anchor is POD-able")
    }
}

/// POD mirror of [`RigidStateSnapshot`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KpRigidStateSnapshot {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub matrix: [f64; 9],
}

impl KpRigidStateSnapshot {
    fn to_pod(snapshot: &RigidStateSnapshot) -> KpRigidStateSnapshot {
        KpRigidStateSnapshot {
            position: snapshot.position,
            velocity: snapshot.velocity,
            angular_velocity: snapshot.angular_velocity,
            matrix: snapshot.matrix,
        }
    }

    fn to_rich(self) -> RigidStateSnapshot {
        RigidStateSnapshot {
            position: self.position,
            velocity: self.velocity,
            angular_velocity: self.angular_velocity,
            matrix: self.matrix,
        }
    }
}

/// POD mirror of [`NetworkCorrectionState`] (the `network_correction`
/// latch), including the nested `resume_topology`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpNetworkCorrectionState {
    /// `"snap"` or `"blend"`.
    pub mode: KpString64,
    pub duration_ms: f64,
    pub elapsed_ms: f64,
    pub has_sequence: u8,
    pub sequence: u32,
    pub initial_state: KpRigidStateSnapshot,
    pub target_state: KpRigidStateSnapshot,
    pub complete: u8,
    pub has_policy_authority: u8,
    pub policy_authority: KpString64,
    pub has_resume_topology: u8,
    pub resume_topology: KpTopologyState,
}

impl KpNetworkCorrectionState {
    fn to_pod(correction: &NetworkCorrectionState) -> Result<KpNetworkCorrectionState, String> {
        Ok(KpNetworkCorrectionState {
            mode: KpString64::set(&correction.mode)?,
            duration_ms: correction.duration_ms,
            elapsed_ms: correction.elapsed_ms,
            has_sequence: bool_u8(correction.sequence.is_some()),
            sequence: correction.sequence.unwrap_or(0),
            initial_state: KpRigidStateSnapshot::to_pod(&correction.initial_state),
            target_state: KpRigidStateSnapshot::to_pod(&correction.target_state),
            complete: bool_u8(correction.complete),
            has_policy_authority: bool_u8(correction.policy_authority.is_some()),
            policy_authority: match &correction.policy_authority {
                Some(authority) => KpString64::set(authority)?,
                None => KpString64::default(),
            },
            has_resume_topology: bool_u8(correction.resume_topology.is_some()),
            resume_topology: match &correction.resume_topology {
                Some(topology) => KpTopologyState::to_pod(topology)?,
                None => KpTopologyState::default(),
            },
        })
    }

    fn to_rich(self) -> Result<NetworkCorrectionState, String> {
        Ok(NetworkCorrectionState {
            mode: self.mode.get()?,
            duration_ms: self.duration_ms,
            elapsed_ms: self.elapsed_ms,
            sequence: if u8_bool(self.has_sequence) {
                Some(self.sequence)
            } else {
                None
            },
            initial_state: self.initial_state.to_rich(),
            target_state: self.target_state.to_rich(),
            complete: u8_bool(self.complete),
            policy_authority: if u8_bool(self.has_policy_authority) {
                Some(self.policy_authority.get()?)
            } else {
                None
            },
            resume_topology: if u8_bool(self.has_resume_topology) {
                Some(self.resume_topology.to_rich()?)
            } else {
                None
            },
        })
    }
}

impl Default for KpNetworkCorrectionState {
    fn default() -> Self {
        KpNetworkCorrectionState::to_pod(&NetworkCorrectionState::default())
            .expect("default network correction state is POD-able")
    }
}

// ---------------------------------------------------------------------------
// POD mirror: VehicleState
// ---------------------------------------------------------------------------

/// POD mirror of [`VehicleState`] — the frozen state blob returned by
/// [`kart_physics_get_state`] and accepted by [`kart_physics_create`] /
/// [`kart_physics_set_state`]. See module docs for the Option/bool/string
/// conventions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpVehicleState {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub matrix: [f64; 9],
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub boost_state: i64,
    pub boost_remaining_ms: i64,
    pub drive_aa4: f64,
    pub drive_aa8: f64,
    pub drive_selected_slot: f64,
    pub drive_raw_aa0: f64,
    pub drive_sign_aac: u8,
    pub ground_timer_a80: f64,
    pub contact_compression: [f64; 4],
    pub collision_arg_threshold: f64,
    pub collision_e38: u8,
    pub collision_e4c: u8,
    pub soft_selector_af4: u8,
    pub soft_flag_188: u8,
    pub takeoff_separation_pending: u8,
    pub has_surface_mode_latched: u8,
    pub surface_mode_latched: i64,
    pub surface_selector_elapsed_ms: f64,
    /// `"c016"` or `"c72c"`.
    pub checkpoint_mode_family: KpString64,
    pub checkpoint_target: [f64; 3],
    pub checkpoint_gate: u8,
    pub has_checkpoint_gate_armed_tick_ms: u8,
    pub checkpoint_gate_armed_tick_ms: i64,
    pub has_checkpoint_marker_active_id: u8,
    pub checkpoint_marker_active_id: KpString64,
    pub manual_restart_key_latched: u8,
    pub manual_restart_count: i64,
    pub special_trigger_cooldown_ms: f64,
    pub has_authored_trigger_contact_id: u8,
    pub authored_trigger_contact_id: KpString64,
    pub has_rail_selector_runtime: u8,
    pub rail_selector_runtime: KpRailSelectorRuntime,
    pub has_rail_base_bf8: u8,
    pub rail_base_bf8: [f64; 9],
    pub has_rail_constraint_c1c: u8,
    pub rail_constraint_c1c: [f64; 9],
    pub has_rail_orientation_dt_latch: u8,
    pub rail_orientation_dt_latch: f64,
    pub has_rail_capture_latch: u8,
    pub rail_capture_latch: KpRailCaptureLatch,
    pub has_moving_contact_anchor: u8,
    pub moving_contact_anchor: KpMovingContactAnchor,
    pub has_network_correction: u8,
    pub network_correction: KpNetworkCorrectionState,
    pub has_last_network_sequence: u8,
    pub last_network_sequence: u32,
    pub network_packets_accepted: i64,
    pub network_packets_rejected: i64,
    pub network_packets_lost_estimate: i64,
    pub topology: KpTopologyState,
    pub time_ms: u64,
    pub step_index: u64,
    pub has_scheduler_previous_tick_ms: u8,
    pub scheduler_previous_tick_ms: u32,
    // ABI v3 tail: do not insert fields above this point.
    pub selector_b08: u8,
    pub selector_b5c: u8,
    pub selector_f08: u8,
    pub selector_fec: i64,
    pub selector_ff8: i64,
    pub field_af8: f64,
    pub recovery_timer: f64,
    pub drift_timer_b0c: f64,
    pub drift_field_b00: f64,
    pub drift_snapshot_5d0: f64,
    pub drift_b1c_charge_settlement: u8,
    pub drift_blend_ab0: f64,
    pub drift_blend_ab4: f64,
    pub drift_key_latched: u8,
    // ABI v4 tail: current-family ordinary-N2O state. Keep the fixed slots
    // as parallel POD arrays; rich InventorySlot validation remains the one
    // authority after conversion.
    pub booster_inventory_item_ids: [i64; 2],
    pub booster_inventory_quantities: [i64; 2],
    pub item_use_key_latched: u8,
    pub boost_gauge_committed_b10: f64,
    pub boost_gauge_progress_b20: f64,
    pub boost_gauge_pending_b2c: f64,
    pub boost_gauge_history_b38: f64,
    // ABI v5 tail: GoItemBanana +0xB9C timed item-effect state.
    pub banana_effect_remaining_ms: i64,
    pub banana_effect_owner_id: KpString64,
}

impl KpVehicleState {
    /// Rich -> POD. Fixed-shape state violations and over-capacity
    /// strings/JSON are returned as typed [`PodExportError`] values.
    pub fn to_pod(state: &VehicleState) -> Result<KpVehicleState, PodExportError> {
        let [booster_slot_0, booster_slot_1] = state.booster_inventory_slots.as_slice() else {
            return Err(PodExportError::InvalidState(format!(
                "booster_inventory_slots must contain exactly 2 slots, got {}",
                state.booster_inventory_slots.len()
            )));
        };
        Ok(KpVehicleState {
            position: state.position,
            velocity: state.velocity,
            angular_velocity: state.angular_velocity,
            matrix: state.matrix,
            force_accumulator: state.force_accumulator,
            torque_accumulator: state.torque_accumulator,
            boost_state: state.boost_state,
            boost_remaining_ms: state.boost_remaining_ms,
            drive_aa4: state.drive_aa4,
            drive_aa8: state.drive_aa8,
            drive_selected_slot: state.drive_selected_slot,
            drive_raw_aa0: state.drive_raw_aa0,
            drive_sign_aac: bool_u8(state.drive_sign_aac),
            ground_timer_a80: state.ground_timer_a80,
            contact_compression: state.contact_compression,
            collision_arg_threshold: state.collision_arg_threshold,
            collision_e38: bool_u8(state.collision_e38),
            collision_e4c: bool_u8(state.collision_e4c),
            soft_selector_af4: bool_u8(state.soft_selector_af4),
            soft_flag_188: bool_u8(state.soft_flag_188),
            takeoff_separation_pending: bool_u8(state.takeoff_separation_pending),
            has_surface_mode_latched: bool_u8(state.surface_mode_latched.is_some()),
            surface_mode_latched: state.surface_mode_latched.unwrap_or(0),
            surface_selector_elapsed_ms: state.surface_selector_elapsed_ms,
            checkpoint_mode_family: KpString64::set(&state.checkpoint_mode_family)
                .map_err(PodExportError::StringTooLong)?,
            checkpoint_target: state.checkpoint_target,
            checkpoint_gate: bool_u8(state.checkpoint_gate),
            has_checkpoint_gate_armed_tick_ms: bool_u8(
                state.checkpoint_gate_armed_tick_ms.is_some(),
            ),
            checkpoint_gate_armed_tick_ms: state.checkpoint_gate_armed_tick_ms.unwrap_or(0),
            has_checkpoint_marker_active_id: bool_u8(state.checkpoint_marker_active_id.is_some()),
            checkpoint_marker_active_id: match &state.checkpoint_marker_active_id {
                Some(id) => KpString64::set(id).map_err(PodExportError::StringTooLong)?,
                None => KpString64::default(),
            },
            manual_restart_key_latched: bool_u8(state.manual_restart_key_latched),
            manual_restart_count: state.manual_restart_count,
            special_trigger_cooldown_ms: state.special_trigger_cooldown_ms,
            has_authored_trigger_contact_id: bool_u8(state.authored_trigger_contact_id.is_some()),
            authored_trigger_contact_id: match &state.authored_trigger_contact_id {
                Some(id) => KpString64::set(id).map_err(PodExportError::StringTooLong)?,
                None => KpString64::default(),
            },
            has_rail_selector_runtime: bool_u8(state.rail_selector_runtime.is_some()),
            rail_selector_runtime: match &state.rail_selector_runtime {
                Some(runtime) => {
                    KpRailSelectorRuntime::to_pod(runtime).map_err(PodExportError::StringTooLong)?
                }
                None => KpRailSelectorRuntime::default(),
            },
            has_rail_base_bf8: bool_u8(state.rail_base_bf8.is_some()),
            rail_base_bf8: state.rail_base_bf8.unwrap_or([0.0; 9]),
            has_rail_constraint_c1c: bool_u8(state.rail_constraint_c1c.is_some()),
            rail_constraint_c1c: state.rail_constraint_c1c.unwrap_or([0.0; 9]),
            has_rail_orientation_dt_latch: bool_u8(state.rail_orientation_dt_latch.is_some()),
            rail_orientation_dt_latch: state.rail_orientation_dt_latch.unwrap_or(0.0),
            has_rail_capture_latch: bool_u8(state.rail_capture_latch.is_some()),
            rail_capture_latch: match &state.rail_capture_latch {
                Some(latch) => KpRailCaptureLatch::to_pod(latch)?,
                None => KpRailCaptureLatch::default(),
            },
            has_moving_contact_anchor: bool_u8(state.moving_contact_anchor.is_some()),
            moving_contact_anchor: match &state.moving_contact_anchor {
                Some(anchor) => {
                    KpMovingContactAnchor::to_pod(anchor).map_err(PodExportError::StringTooLong)?
                }
                None => KpMovingContactAnchor::default(),
            },
            has_network_correction: bool_u8(state.network_correction.is_some()),
            network_correction: match &state.network_correction {
                Some(correction) => KpNetworkCorrectionState::to_pod(correction)
                    .map_err(PodExportError::StringTooLong)?,
                None => KpNetworkCorrectionState::default(),
            },
            has_last_network_sequence: bool_u8(state.last_network_sequence.is_some()),
            last_network_sequence: state.last_network_sequence.unwrap_or(0),
            network_packets_accepted: state.network_packets_accepted,
            network_packets_rejected: state.network_packets_rejected,
            network_packets_lost_estimate: state.network_packets_lost_estimate,
            topology: KpTopologyState::to_pod(&state.topology)
                .map_err(PodExportError::StringTooLong)?,
            time_ms: state.time_ms,
            step_index: state.step_index,
            has_scheduler_previous_tick_ms: bool_u8(state.scheduler_previous_tick_ms.is_some()),
            scheduler_previous_tick_ms: state.scheduler_previous_tick_ms.unwrap_or(0),
            selector_b08: bool_u8(state.selector_b08),
            selector_b5c: bool_u8(state.selector_b5c),
            selector_f08: bool_u8(state.selector_f08),
            selector_fec: state.selector_fec,
            selector_ff8: state.selector_ff8,
            field_af8: state.field_af8,
            recovery_timer: state.recovery_timer,
            drift_timer_b0c: state.drift_timer_b0c,
            drift_field_b00: state.drift_field_b00,
            drift_snapshot_5d0: state.drift_snapshot_5d0,
            drift_b1c_charge_settlement: bool_u8(state.drift_b1c_charge_settlement),
            drift_blend_ab0: state.drift_blend_ab0,
            drift_blend_ab4: state.drift_blend_ab4,
            drift_key_latched: bool_u8(state.drift_key_latched),
            booster_inventory_item_ids: [booster_slot_0.item_id, booster_slot_1.item_id],
            booster_inventory_quantities: [booster_slot_0.quantity, booster_slot_1.quantity],
            item_use_key_latched: bool_u8(state.item_use_key_latched),
            boost_gauge_committed_b10: state.boost_gauge_committed_b10,
            boost_gauge_progress_b20: state.boost_gauge_progress_b20,
            boost_gauge_pending_b2c: state.boost_gauge_pending_b2c,
            boost_gauge_history_b38: state.boost_gauge_history_b38,
            banana_effect_remaining_ms: state.banana_effect_remaining_ms,
            banana_effect_owner_id: match &state.banana_effect_owner_id {
                Some(id) => KpString64::set(id).map_err(PodExportError::StringTooLong)?,
                None => KpString64::default(),
            },
        })
    }

    /// POD -> rich. Fails on malformed tags/strings/JSON; the result is NOT
    /// re-validated here (callers run `VehicleState.validate`).
    pub fn to_rich(self) -> Result<VehicleState, String> {
        Ok(VehicleState {
            position: self.position,
            velocity: self.velocity,
            angular_velocity: self.angular_velocity,
            matrix: self.matrix,
            force_accumulator: self.force_accumulator,
            torque_accumulator: self.torque_accumulator,
            boost_state: self.boost_state,
            boost_remaining_ms: self.boost_remaining_ms,
            booster_inventory_slots: vec![
                InventorySlot {
                    item_id: self.booster_inventory_item_ids[0],
                    quantity: self.booster_inventory_quantities[0],
                },
                InventorySlot {
                    item_id: self.booster_inventory_item_ids[1],
                    quantity: self.booster_inventory_quantities[1],
                },
            ],
            item_use_key_latched: u8_bool(self.item_use_key_latched),
            banana_effect_remaining_ms: self.banana_effect_remaining_ms,
            banana_effect_owner_id: if self.banana_effect_owner_id.len == 0 {
                None
            } else {
                Some(self.banana_effect_owner_id.get()?)
            },
            drive_aa4: self.drive_aa4,
            drive_aa8: self.drive_aa8,
            drive_selected_slot: self.drive_selected_slot,
            drive_raw_aa0: self.drive_raw_aa0,
            drive_sign_aac: u8_bool(self.drive_sign_aac),
            ground_timer_a80: self.ground_timer_a80,
            contact_compression: self.contact_compression,
            collision_arg_threshold: self.collision_arg_threshold,
            collision_e38: u8_bool(self.collision_e38),
            collision_e4c: u8_bool(self.collision_e4c),
            soft_selector_af4: u8_bool(self.soft_selector_af4),
            soft_flag_188: u8_bool(self.soft_flag_188),
            takeoff_separation_pending: u8_bool(self.takeoff_separation_pending),
            surface_mode_latched: if u8_bool(self.has_surface_mode_latched) {
                Some(self.surface_mode_latched)
            } else {
                None
            },
            surface_selector_elapsed_ms: self.surface_selector_elapsed_ms,
            checkpoint_mode_family: self.checkpoint_mode_family.get()?,
            checkpoint_target: self.checkpoint_target,
            checkpoint_gate: u8_bool(self.checkpoint_gate),
            checkpoint_gate_armed_tick_ms: if u8_bool(self.has_checkpoint_gate_armed_tick_ms) {
                Some(self.checkpoint_gate_armed_tick_ms)
            } else {
                None
            },
            checkpoint_marker_active_id: if u8_bool(self.has_checkpoint_marker_active_id) {
                Some(self.checkpoint_marker_active_id.get()?)
            } else {
                None
            },
            manual_restart_key_latched: u8_bool(self.manual_restart_key_latched),
            manual_restart_count: self.manual_restart_count,
            special_trigger_cooldown_ms: self.special_trigger_cooldown_ms,
            authored_trigger_contact_id: if u8_bool(self.has_authored_trigger_contact_id) {
                Some(self.authored_trigger_contact_id.get()?)
            } else {
                None
            },
            rail_selector_runtime: if u8_bool(self.has_rail_selector_runtime) {
                Some(self.rail_selector_runtime.to_rich()?)
            } else {
                None
            },
            rail_base_bf8: if u8_bool(self.has_rail_base_bf8) {
                Some(self.rail_base_bf8)
            } else {
                None
            },
            rail_constraint_c1c: if u8_bool(self.has_rail_constraint_c1c) {
                Some(self.rail_constraint_c1c)
            } else {
                None
            },
            rail_orientation_dt_latch: if u8_bool(self.has_rail_orientation_dt_latch) {
                Some(self.rail_orientation_dt_latch)
            } else {
                None
            },
            rail_capture_latch: if u8_bool(self.has_rail_capture_latch) {
                Some(self.rail_capture_latch.to_rich()?)
            } else {
                None
            },
            moving_contact_anchor: if u8_bool(self.has_moving_contact_anchor) {
                Some(self.moving_contact_anchor.to_rich()?)
            } else {
                None
            },
            network_correction: if u8_bool(self.has_network_correction) {
                Some(self.network_correction.to_rich()?)
            } else {
                None
            },
            last_network_sequence: if u8_bool(self.has_last_network_sequence) {
                Some(self.last_network_sequence)
            } else {
                None
            },
            network_packets_accepted: self.network_packets_accepted,
            network_packets_rejected: self.network_packets_rejected,
            network_packets_lost_estimate: self.network_packets_lost_estimate,
            topology: self.topology.to_rich()?,
            time_ms: self.time_ms,
            step_index: self.step_index,
            scheduler_previous_tick_ms: if u8_bool(self.has_scheduler_previous_tick_ms) {
                Some(self.scheduler_previous_tick_ms)
            } else {
                None
            },
            selector_b08: u8_bool(self.selector_b08),
            selector_b5c: u8_bool(self.selector_b5c),
            selector_f08: u8_bool(self.selector_f08),
            selector_fec: self.selector_fec,
            selector_ff8: self.selector_ff8,
            field_af8: self.field_af8,
            recovery_timer: self.recovery_timer,
            drift_timer_b0c: self.drift_timer_b0c,
            drift_field_b00: self.drift_field_b00,
            drift_snapshot_5d0: self.drift_snapshot_5d0,
            drift_b1c_charge_settlement: u8_bool(self.drift_b1c_charge_settlement),
            boost_gauge_committed_b10: self.boost_gauge_committed_b10,
            boost_gauge_progress_b20: self.boost_gauge_progress_b20,
            boost_gauge_pending_b2c: self.boost_gauge_pending_b2c,
            boost_gauge_history_b38: self.boost_gauge_history_b38,
            drift_blend_ab0: self.drift_blend_ab0,
            drift_blend_ab4: self.drift_blend_ab4,
            drift_key_latched: u8_bool(self.drift_key_latched),
        })
    }
}

impl Default for KpVehicleState {
    fn default() -> Self {
        KpVehicleState::to_pod(&VehicleState::default()).expect("default state is POD-able")
    }
}

// ---------------------------------------------------------------------------
// POD mirrors: VehicleParameters (+ boost)
// ---------------------------------------------------------------------------

/// POD mirror of [`BoostParameters`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpBoostParameters {
    pub correction_dir: [f64; 3],
    pub direction_gain: f64,
    pub selected_slot: f64,
    pub attached_state: i64,
    pub e8c: u8,
    pub efc: u8,
    pub flag_5bc: u8,
    pub raw_144: f64,
    pub obf_808: f64,
    pub obf_814: f64,
    pub obf_69c: f64,
    pub obf_684: f64,
    pub obf_8ac: f64,
    pub obf_8e8: f64,
    pub obf_94c: f64,
}

impl KpBoostParameters {
    fn to_pod(boost: &BoostParameters) -> KpBoostParameters {
        KpBoostParameters {
            correction_dir: boost.correction_dir,
            direction_gain: boost.direction_gain,
            selected_slot: boost.selected_slot,
            attached_state: boost.attached_state,
            e8c: bool_u8(boost.e8c),
            efc: bool_u8(boost.efc),
            flag_5bc: bool_u8(boost.flag_5bc),
            raw_144: boost.raw_144,
            obf_808: boost.obf_808,
            obf_814: boost.obf_814,
            obf_69c: boost.obf_69c,
            obf_684: boost.obf_684,
            obf_8ac: boost.obf_8ac,
            obf_8e8: boost.obf_8e8,
            obf_94c: boost.obf_94c,
        }
    }

    fn to_rich(self) -> BoostParameters {
        BoostParameters {
            correction_dir: self.correction_dir,
            direction_gain: self.direction_gain,
            selected_slot: self.selected_slot,
            attached_state: self.attached_state,
            e8c: u8_bool(self.e8c),
            efc: u8_bool(self.efc),
            flag_5bc: u8_bool(self.flag_5bc),
            raw_144: self.raw_144,
            obf_808: self.obf_808,
            obf_814: self.obf_814,
            obf_69c: self.obf_69c,
            obf_684: self.obf_684,
            obf_8ac: self.obf_8ac,
            obf_8e8: self.obf_8e8,
            obf_94c: self.obf_94c,
        }
    }
}

impl Default for KpBoostParameters {
    fn default() -> Self {
        KpBoostParameters::to_pod(&BoostParameters::default())
    }
}

/// POD mirror of [`VehicleParameters`] — flat scalars, the boost struct,
/// and `inverse_inertia_matrix`. `topology_selector_delta_ms` /
/// `initial_rail_capture_commit` are the engine-derived per-substep slots;
/// callers normally leave their `has_*` tags 0 (the frame loop fills them).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpVehicleParameters {
    pub substep_ms: i64,
    pub has_topology_selector_delta_ms: u8,
    pub topology_selector_delta_ms: f64,
    pub has_initial_rail_capture_commit: u8,
    pub initial_rail_capture_commit: u8,
    pub mass: f64,
    pub common_damping: f64,
    pub state6_residual_angular_damping: f64,
    pub directional_damping: f64,
    pub damping_runtime_multiplier: f64,
    pub surface_mode: i64,
    pub forward_axis_column: i64,
    pub forward_axis_sign: f64,
    pub ground_projection_floor: f64,
    pub ground_angular_correction_scale: f64,
    pub primary_ground_direction_gain: f64,
    pub fallback_ground_direction_gain: f64,
    pub negative_adc_force_scale: f64,
    pub lateral_high_dot_scale: f64,
    pub lateral_low_dot_scale: f64,
    pub ground_force_multiplier: f64,
    pub ground_longitudinal_gain: f64,
    pub ground_lateral_gain: f64,
    pub ground_axis_long: f64,
    pub ground_axis_lat: f64,
    pub ground_settlement_multiplier: f64,
    pub ground_torque_y_gain: f64,
    pub generic_slip_force_scale: f64,
    pub generic_slip_steering_basis: f64,
    pub contact_rest_compression: f64,
    pub contact_spring: f64,
    pub contact_rebound: f64,
    pub contact_bump: f64,
    pub contact_compression_quantum: f64,
    pub contact_probe_span_scale: f64,
    pub contact_gravity_per_mass: f64,
    pub drive_tune_718: f64,
    pub drive_speed_scale_71c: f64,
    pub drive_attenuation_bb8: f64,
    pub inverse_inertia_matrix: [f64; 9],
    pub boost: KpBoostParameters,
    pub collision_half_extents: [f64; 3],
    pub collision_center_up: f64,
    pub state6_contact_half_span: f64,
    // ABI v3 tail: do not insert fields above this point.
    pub drift_window_duration_734: f64,
    pub drift_gain_728: f64,
    pub drift_torque_y_gain_758: f64,
    pub drift_speed_profile_denominator: f64,
    pub drift_active_ab0_rate: f64,
    pub drift_active_ab4_rate: f64,
    pub drift_timer_ab0_rate: f64,
    // ABI v4 tail.
    pub item6_boost_duration_ms: i64,
    pub boost_gauge_raw_760: f64,
}

impl KpVehicleParameters {
    pub fn to_pod(parameters: &VehicleParameters) -> KpVehicleParameters {
        KpVehicleParameters {
            substep_ms: parameters.substep_ms,
            has_topology_selector_delta_ms: bool_u8(
                parameters.topology_selector_delta_ms.is_some(),
            ),
            topology_selector_delta_ms: parameters.topology_selector_delta_ms.unwrap_or(0.0),
            has_initial_rail_capture_commit: bool_u8(
                parameters.initial_rail_capture_commit.is_some(),
            ),
            initial_rail_capture_commit: bool_u8(
                parameters.initial_rail_capture_commit.unwrap_or(false),
            ),
            mass: parameters.mass,
            common_damping: parameters.common_damping,
            state6_residual_angular_damping: parameters.state6_residual_angular_damping,
            directional_damping: parameters.directional_damping,
            damping_runtime_multiplier: parameters.damping_runtime_multiplier,
            surface_mode: parameters.surface_mode,
            forward_axis_column: parameters.forward_axis_column,
            forward_axis_sign: parameters.forward_axis_sign,
            ground_projection_floor: parameters.ground_projection_floor,
            ground_angular_correction_scale: parameters.ground_angular_correction_scale,
            primary_ground_direction_gain: parameters.primary_ground_direction_gain,
            fallback_ground_direction_gain: parameters.fallback_ground_direction_gain,
            negative_adc_force_scale: parameters.negative_adc_force_scale,
            lateral_high_dot_scale: parameters.lateral_high_dot_scale,
            lateral_low_dot_scale: parameters.lateral_low_dot_scale,
            ground_force_multiplier: parameters.ground_force_multiplier,
            ground_longitudinal_gain: parameters.ground_longitudinal_gain,
            ground_lateral_gain: parameters.ground_lateral_gain,
            ground_axis_long: parameters.ground_axis_long,
            ground_axis_lat: parameters.ground_axis_lat,
            ground_settlement_multiplier: parameters.ground_settlement_multiplier,
            ground_torque_y_gain: parameters.ground_torque_y_gain,
            generic_slip_force_scale: parameters.generic_slip_force_scale,
            generic_slip_steering_basis: parameters.generic_slip_steering_basis,
            contact_rest_compression: parameters.contact_rest_compression,
            contact_spring: parameters.contact_spring,
            contact_rebound: parameters.contact_rebound,
            contact_bump: parameters.contact_bump,
            contact_compression_quantum: parameters.contact_compression_quantum,
            contact_probe_span_scale: parameters.contact_probe_span_scale,
            contact_gravity_per_mass: parameters.contact_gravity_per_mass,
            drive_tune_718: parameters.drive_tune_718,
            drive_speed_scale_71c: parameters.drive_speed_scale_71c,
            drive_attenuation_bb8: parameters.drive_attenuation_bb8,
            inverse_inertia_matrix: parameters.inverse_inertia_matrix,
            boost: KpBoostParameters::to_pod(&parameters.boost),
            collision_half_extents: parameters.collision_half_extents,
            collision_center_up: parameters.collision_center_up,
            state6_contact_half_span: parameters.state6_contact_half_span,
            drift_window_duration_734: parameters.drift_window_duration_734,
            drift_gain_728: parameters.drift_gain_728,
            drift_torque_y_gain_758: parameters.drift_torque_y_gain_758,
            drift_speed_profile_denominator: parameters.drift_speed_profile_denominator,
            drift_active_ab0_rate: parameters.drift_active_ab0_rate,
            drift_active_ab4_rate: parameters.drift_active_ab4_rate,
            drift_timer_ab0_rate: parameters.drift_timer_ab0_rate,
            item6_boost_duration_ms: parameters.item6_boost_duration_ms,
            boost_gauge_raw_760: parameters.boost_gauge_raw_760,
        }
    }

    pub fn to_rich(self) -> VehicleParameters {
        VehicleParameters {
            substep_ms: self.substep_ms,
            topology_selector_delta_ms: if u8_bool(self.has_topology_selector_delta_ms) {
                Some(self.topology_selector_delta_ms)
            } else {
                None
            },
            initial_rail_capture_commit: if u8_bool(self.has_initial_rail_capture_commit) {
                Some(u8_bool(self.initial_rail_capture_commit))
            } else {
                None
            },
            mass: self.mass,
            common_damping: self.common_damping,
            state6_residual_angular_damping: self.state6_residual_angular_damping,
            directional_damping: self.directional_damping,
            damping_runtime_multiplier: self.damping_runtime_multiplier,
            surface_mode: self.surface_mode,
            forward_axis_column: self.forward_axis_column,
            forward_axis_sign: self.forward_axis_sign,
            ground_projection_floor: self.ground_projection_floor,
            ground_angular_correction_scale: self.ground_angular_correction_scale,
            primary_ground_direction_gain: self.primary_ground_direction_gain,
            fallback_ground_direction_gain: self.fallback_ground_direction_gain,
            negative_adc_force_scale: self.negative_adc_force_scale,
            lateral_high_dot_scale: self.lateral_high_dot_scale,
            lateral_low_dot_scale: self.lateral_low_dot_scale,
            ground_force_multiplier: self.ground_force_multiplier,
            ground_longitudinal_gain: self.ground_longitudinal_gain,
            ground_lateral_gain: self.ground_lateral_gain,
            ground_axis_long: self.ground_axis_long,
            ground_axis_lat: self.ground_axis_lat,
            ground_settlement_multiplier: self.ground_settlement_multiplier,
            ground_torque_y_gain: self.ground_torque_y_gain,
            generic_slip_force_scale: self.generic_slip_force_scale,
            generic_slip_steering_basis: self.generic_slip_steering_basis,
            contact_rest_compression: self.contact_rest_compression,
            contact_spring: self.contact_spring,
            contact_rebound: self.contact_rebound,
            contact_bump: self.contact_bump,
            contact_compression_quantum: self.contact_compression_quantum,
            contact_probe_span_scale: self.contact_probe_span_scale,
            contact_gravity_per_mass: self.contact_gravity_per_mass,
            drive_tune_718: self.drive_tune_718,
            drive_speed_scale_71c: self.drive_speed_scale_71c,
            drive_attenuation_bb8: self.drive_attenuation_bb8,
            inverse_inertia_matrix: self.inverse_inertia_matrix,
            boost: self.boost.to_rich(),
            item6_boost_duration_ms: self.item6_boost_duration_ms,
            boost_gauge_raw_760: self.boost_gauge_raw_760,
            // ABI v4 predates the run25 producer tune fields. Keep its
            // frozen layout and supply the current-family modal defaults;
            // exposing custom profiles requires an intentional ABI bump.
            boost_gauge_primary_rate_6b4: 350.0,
            boost_gauge_primary_override_100c: 0.0,
            boost_gauge_window_add_988: 0.0,
            boost_gauge_window_factor_994: 0.0,
            boost_gauge_speed_threshold_54c: 160.0,
            collision_half_extents: self.collision_half_extents,
            collision_center_up: self.collision_center_up,
            state6_contact_half_span: self.state6_contact_half_span,
            drift_window_duration_734: self.drift_window_duration_734,
            drift_gain_728: self.drift_gain_728,
            drift_torque_y_gain_758: self.drift_torque_y_gain_758,
            drift_speed_profile_denominator: self.drift_speed_profile_denominator,
            drift_active_ab0_rate: self.drift_active_ab0_rate,
            drift_active_ab4_rate: self.drift_active_ab4_rate,
            drift_timer_ab0_rate: self.drift_timer_ab0_rate,
        }
    }
}

impl Default for KpVehicleParameters {
    fn default() -> Self {
        KpVehicleParameters::to_pod(&VehicleParameters::default())
    }
}

// ---------------------------------------------------------------------------
// Historical state/parameter layout translation
// ---------------------------------------------------------------------------

const KP_JSON_CAPACITY_V1: usize = 4096;
const KP_JSON_CAPACITY_DELTA_V2: usize = KP_JSON_CAPACITY as usize - KP_JSON_CAPACITY_V1;
const KP_STATE_RAIL_JSON_OFFSET: usize = std::mem::offset_of!(KpVehicleState, rail_capture_latch)
    + std::mem::offset_of!(KpRailCaptureLatch, selector_json);
const KP_STATE_CURRENT_TAIL_OFFSET: usize =
    std::mem::offset_of!(KpVehicleState, has_moving_contact_anchor);
const KP_STATE_V1_TAIL_OFFSET: usize = KP_STATE_CURRENT_TAIL_OFFSET - KP_JSON_CAPACITY_DELTA_V2;
const KP_JSON_BLOB_V1_BYTES: usize = std::mem::size_of::<u32>() + KP_JSON_CAPACITY_V1;

fn unsupported_negotiated_abi(abi: u32) -> String {
    format!("unsupported negotiated ABI version {abi}")
}

/// Read a host state using only the bytes owned by its negotiated ABI.
///
/// ABI v1 is not a simple prefix: increasing the inline rail-selector JSON
/// slot from 4096 to 65536 bytes in ABI v2 shifted every later field. Copy
/// the shared regions around that slot into a current default value, leaving
/// every tail introduced after the negotiated version at its defaults.
///
/// # Safety
/// `state` must point at a readable state object of the negotiated layout.
unsafe fn read_state_for_abi(
    state: *const KpVehicleState,
    negotiated_abi: u32,
) -> Result<KpVehicleState, String> {
    match negotiated_abi {
        1 => {
            let source = state.cast::<u8>();
            // Read the v1 length without ever reaching beyond the v1 object.
            let json_len = unsafe {
                std::ptr::read_unaligned(source.add(KP_STATE_RAIL_JSON_OFFSET).cast::<u32>())
            } as usize;
            if json_len > KP_JSON_CAPACITY_V1 {
                return Err(format!(
                    "ABI v1 rail selector JSON len {json_len} exceeds {KP_JSON_CAPACITY_V1}"
                ));
            }

            let mut current = KpVehicleState::default();
            let destination = (&mut current as *mut KpVehicleState).cast::<u8>();
            unsafe {
                std::ptr::copy_nonoverlapping(source, destination, KP_STATE_RAIL_JSON_OFFSET);
                std::ptr::copy_nonoverlapping(
                    source.add(KP_STATE_RAIL_JSON_OFFSET),
                    destination.add(KP_STATE_RAIL_JSON_OFFSET),
                    KP_JSON_BLOB_V1_BYTES,
                );
                std::ptr::copy_nonoverlapping(
                    source.add(KP_STATE_V1_TAIL_OFFSET),
                    destination.add(KP_STATE_CURRENT_TAIL_OFFSET),
                    KP_VEHICLE_STATE_V1_BYTES - KP_STATE_V1_TAIL_OFFSET,
                );
            }
            Ok(current)
        }
        2 => {
            let mut current = KpVehicleState::default();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.cast::<u8>(),
                    (&mut current as *mut KpVehicleState).cast::<u8>(),
                    KP_VEHICLE_STATE_V2_BYTES,
                );
            }
            Ok(current)
        }
        3 => {
            let mut current = KpVehicleState::default();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.cast::<u8>(),
                    (&mut current as *mut KpVehicleState).cast::<u8>(),
                    KP_VEHICLE_STATE_V3_BYTES,
                );
            }
            Ok(current)
        }
        KP_ABI_VERSION => Ok(unsafe { std::ptr::read(state) }),
        other => Err(unsupported_negotiated_abi(other)),
    }
}

/// Write a state using only the bytes owned by its negotiated ABI.
///
/// # Safety
/// `out` must point at writable storage for the negotiated state layout.
unsafe fn write_state_for_abi(
    out: *mut KpVehicleState,
    state: &KpVehicleState,
    negotiated_abi: u32,
) -> Result<(), PodExportError> {
    match negotiated_abi {
        1 => {
            let json_len = state.rail_capture_latch.selector_json.len as usize;
            if json_len > KP_JSON_CAPACITY_V1 {
                return Err(PodExportError::Json(format!(
                    "ABI v1 rail selector JSON exceeds {KP_JSON_CAPACITY_V1} bytes \
                     ({json_len} required)"
                )));
            }
            let destination = out.cast::<u8>();
            let source = (state as *const KpVehicleState).cast::<u8>();
            unsafe {
                // Initialize the complete historical object, including padding,
                // before copying its three layout-compatible regions.
                std::ptr::write_bytes(destination, 0, KP_VEHICLE_STATE_V1_BYTES);
                std::ptr::copy_nonoverlapping(source, destination, KP_STATE_RAIL_JSON_OFFSET);
                std::ptr::copy_nonoverlapping(
                    source.add(KP_STATE_RAIL_JSON_OFFSET),
                    destination.add(KP_STATE_RAIL_JSON_OFFSET),
                    KP_JSON_BLOB_V1_BYTES,
                );
                std::ptr::copy_nonoverlapping(
                    source.add(KP_STATE_CURRENT_TAIL_OFFSET),
                    destination.add(KP_STATE_V1_TAIL_OFFSET),
                    KP_VEHICLE_STATE_V1_BYTES - KP_STATE_V1_TAIL_OFFSET,
                );
            }
            Ok(())
        }
        2 => {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (state as *const KpVehicleState).cast::<u8>(),
                    out.cast::<u8>(),
                    KP_VEHICLE_STATE_V2_BYTES,
                );
            }
            Ok(())
        }
        3 => {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (state as *const KpVehicleState).cast::<u8>(),
                    out.cast::<u8>(),
                    KP_VEHICLE_STATE_V3_BYTES,
                );
            }
            Ok(())
        }
        KP_ABI_VERSION => {
            unsafe { std::ptr::write(out, *state) };
            Ok(())
        }
        other => Err(PodExportError::Json(unsupported_negotiated_abi(other))),
    }
}

/// Read parameters using only the frozen prefix for old engines.
///
/// # Safety
/// `parameters` must point at a readable parameter object of the negotiated
/// layout.
unsafe fn read_parameters_for_abi(
    parameters: *const KpVehicleParameters,
    negotiated_abi: u32,
) -> Result<KpVehicleParameters, String> {
    match negotiated_abi {
        1 | 2 => {
            let mut current = KpVehicleParameters::default();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    parameters.cast::<u8>(),
                    (&mut current as *mut KpVehicleParameters).cast::<u8>(),
                    KP_VEHICLE_PARAMETERS_V1_BYTES,
                );
            }
            Ok(current)
        }
        3 => {
            let mut current = KpVehicleParameters::default();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    parameters.cast::<u8>(),
                    (&mut current as *mut KpVehicleParameters).cast::<u8>(),
                    KP_VEHICLE_PARAMETERS_V3_BYTES,
                );
            }
            Ok(current)
        }
        KP_ABI_VERSION => Ok(unsafe { std::ptr::read(parameters) }),
        other => Err(unsupported_negotiated_abi(other)),
    }
}

/// Write the frozen ABI v1/v2 parameter prefix.
///
/// # Safety
/// `out` must point at at least [`KP_VEHICLE_PARAMETERS_V1_BYTES`] writable
/// bytes.
unsafe fn write_parameters_v1_prefix(
    out: *mut KpVehicleParameters,
    parameters: &KpVehicleParameters,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            (parameters as *const KpVehicleParameters).cast::<u8>(),
            out.cast::<u8>(),
            KP_VEHICLE_PARAMETERS_V1_BYTES,
        );
    }
}

/// One frame of input: the u32 game-clock tick (wraps at 2^32 exactly like
/// the engine scheduler) and the key bitmask (`KP_KEY_*`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KpInputs {
    pub tick_ms: u32,
    pub keys: u32,
}

fn keys_from_mask(mask: u32, negotiated_abi: u32) -> Result<Vec<String>, String> {
    let recognized_mask = if negotiated_abi >= 4 {
        KP_KEY_MASK_ALL
    } else if negotiated_abi >= 3 {
        KP_KEY_MASK_ALL & !KP_KEY_ITEM_USE
    } else {
        KP_KEY_MASK_ALL & !(KP_KEY_CTRL | KP_KEY_ITEM_USE)
    };
    if mask & !recognized_mask != 0 {
        return Err(format!(
            "unknown key bits 0x{:x} (recognized mask 0x{recognized_mask:x})",
            mask & !recognized_mask
        ));
    }
    let mut keys = Vec::new();
    for (bit, name) in [
        (KP_KEY_UP, "UP"),
        (KP_KEY_DOWN, "DOWN"),
        (KP_KEY_LEFT, "LEFT"),
        (KP_KEY_RIGHT, "RIGHT"),
        (KP_KEY_R, "R"),
        (KP_KEY_CTRL, "CTRL"),
        (KP_KEY_ITEM_USE, "ITEM_USE"),
    ] {
        if mask & bit != 0 {
            keys.push(name.to_string());
        }
    }
    Ok(keys)
}

// ---------------------------------------------------------------------------
// World vtable (C mapping of the `World` trait)
// ---------------------------------------------------------------------------

/// POD mirror of [`WorldGroundResult`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpWorldGroundResult {
    pub track_id: KpString64,
    pub node: KpString64,
    pub tri_index: i64,
    pub road_surface: KpString64,
    pub surface_id: i64,
    pub road_rail: KpString64,
    pub distance: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub has_triangle: u8,
    /// Three packed triangle vertices (valid when `has_triangle != 0`).
    pub triangle: [f64; 9],
    pub has_contact_basis_44: u8,
    pub contact_basis_44: [f64; 3],
}

impl Default for KpWorldGroundResult {
    fn default() -> Self {
        KpWorldGroundResult {
            track_id: KpString64::default(),
            node: KpString64::default(),
            tri_index: 0,
            road_surface: KpString64::default(),
            surface_id: 0,
            road_rail: KpString64::default(),
            distance: 0.0,
            point: [0.0; 3],
            normal: [0.0; 3],
            has_triangle: 0,
            triangle: [0.0; 9],
            has_contact_basis_44: 0,
            contact_basis_44: [0.0; 3],
        }
    }
}

impl KpWorldGroundResult {
    /// C callback output -> rich for `query_ground`. Strings decode lossily
    /// because this is the one world query whose trait signature cannot
    /// report failure (it returns the row unconditionally); every failable
    /// query imports strictly instead (see the other `to_rich` mirrors).
    fn to_rich_lossy(self) -> WorldGroundResult {
        WorldGroundResult {
            track_id: self.track_id.get_lossy(),
            node: self.node.get_lossy(),
            tri_index: self.tri_index,
            road_surface: self.road_surface.get_lossy(),
            surface_id: self.surface_id,
            road_rail: self.road_rail.get_lossy(),
            distance: self.distance,
            point: self.point,
            normal: self.normal,
            triangle: if u8_bool(self.has_triangle) {
                Some([
                    [self.triangle[0], self.triangle[1], self.triangle[2]],
                    [self.triangle[3], self.triangle[4], self.triangle[5]],
                    [self.triangle[6], self.triangle[7], self.triangle[8]],
                ])
            } else {
                None
            },
            contact_basis_44: if u8_bool(self.has_contact_basis_44) {
                Some(self.contact_basis_44)
            } else {
                None
            },
        }
    }
}

/// POD mirror of [`WorldObbOverlap`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpWorldObbOverlap {
    pub track_id: KpString64,
    pub hit: u8,
    pub loop_index: i64,
    pub fraction: f64,
    pub node: KpString64,
    pub tri_index: i64,
    pub road_surface: KpString64,
    pub surface_id: i64,
    pub road_rail: KpString64,
    pub center: [f64; 3],
    pub support_point: [f64; 3],
    pub ccd_normal: [f64; 3],
    pub normal: [f64; 3],
    pub normal_authority: KpString64,
    pub has_contact_basis_44: u8,
    pub contact_basis_44: [f64; 3],
    pub boost_multiplier_64c: u8,
}

impl Default for KpWorldObbOverlap {
    fn default() -> Self {
        KpWorldObbOverlap {
            track_id: KpString64::default(),
            hit: 0,
            loop_index: 0,
            fraction: 0.0,
            node: KpString64::default(),
            tri_index: 0,
            road_surface: KpString64::default(),
            surface_id: 0,
            road_rail: KpString64::default(),
            center: [0.0; 3],
            support_point: [0.0; 3],
            ccd_normal: [0.0; 3],
            normal: [0.0; 3],
            normal_authority: KpString64::default(),
            has_contact_basis_44: 0,
            contact_basis_44: [0.0; 3],
            boost_multiplier_64c: 0,
        }
    }
}

impl KpWorldObbOverlap {
    /// C callback output -> rich. Strict: invalid UTF-8 fails the query
    /// (the adapter records the detail) instead of silently altering the
    /// bytes the collision response reads.
    fn to_rich(self) -> Result<WorldObbOverlap, String> {
        Ok(WorldObbOverlap {
            track_id: self.track_id.get()?,
            hit: u8_bool(self.hit),
            loop_index: self.loop_index,
            fraction: self.fraction,
            node: self.node.get()?,
            tri_index: self.tri_index,
            road_surface: self.road_surface.get()?,
            surface_id: self.surface_id,
            road_rail: self.road_rail.get()?,
            center: self.center,
            support_point: self.support_point,
            ccd_normal: self.ccd_normal,
            normal: self.normal,
            normal_authority: self.normal_authority.get()?,
            contact_basis_44: if u8_bool(self.has_contact_basis_44) {
                Some(self.contact_basis_44)
            } else {
                None
            },
            boost_multiplier_64c: u8_bool(self.boost_multiplier_64c),
        })
    }
}

/// POD mirror of [`TakeoffLandingContact`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpTakeoffLandingContact {
    pub has_selected: u8,
    pub selected: u8,
    pub has_hit: u8,
    pub hit: u8,
    pub has_normal: u8,
    pub normal: [f64; 3],
    pub has_face_id: u8,
    pub face_id: KpString64,
    pub has_source: u8,
    pub source: KpString64,
}

impl Default for KpTakeoffLandingContact {
    fn default() -> Self {
        KpTakeoffLandingContact {
            has_selected: 0,
            selected: 0,
            has_hit: 0,
            hit: 0,
            has_normal: 0,
            normal: [0.0; 3],
            has_face_id: 0,
            face_id: KpString64::default(),
            has_source: 0,
            source: KpString64::default(),
        }
    }
}

impl KpTakeoffLandingContact {
    /// C callback output -> rich. Strict: `face_id`/`source` can be latched
    /// into state (`authored_trigger_contact_id`), so invalid UTF-8 fails
    /// the query instead of silently lossy-converting state bytes.
    fn to_rich(self) -> Result<TakeoffLandingContact, String> {
        Ok(TakeoffLandingContact {
            selected: if u8_bool(self.has_selected) {
                Some(u8_bool(self.selected))
            } else {
                None
            },
            hit: if u8_bool(self.has_hit) {
                Some(u8_bool(self.hit))
            } else {
                None
            },
            normal: if u8_bool(self.has_normal) {
                Some(self.normal)
            } else {
                None
            },
            face_id: if u8_bool(self.has_face_id) {
                Some(self.face_id.get()?)
            } else {
                None
            },
            source: if u8_bool(self.has_source) {
                Some(self.source.get()?)
            } else {
                None
            },
        })
    }
}

/// POD mirror of [`CheckpointTargetCandidate`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpCheckpointTargetCandidate {
    pub marker_id: KpString64,
    pub marker_ordinal: i64,
    pub position: [f64; 3],
    pub distance: f64,
    pub authority: KpString64,
}

impl Default for KpCheckpointTargetCandidate {
    fn default() -> Self {
        KpCheckpointTargetCandidate {
            marker_id: KpString64::default(),
            marker_ordinal: 0,
            position: [0.0; 3],
            distance: 0.0,
            authority: KpString64::default(),
        }
    }
}

impl KpCheckpointTargetCandidate {
    /// C callback output -> rich. Strict: `marker_id` can be latched into
    /// state, so invalid UTF-8 fails the query.
    fn to_rich(self) -> Result<CheckpointTargetCandidate, String> {
        Ok(CheckpointTargetCandidate {
            marker_id: self.marker_id.get()?,
            marker_ordinal: self.marker_ordinal,
            position: self.position,
            distance: self.distance,
            authority: self.authority.get()?,
        })
    }
}

/// POD mirror of [`CheckpointRespawnCandidate`]. `course_context_id` is an
/// open JSON value and crosses inside `course_context_id_json`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpCheckpointRespawnCandidate {
    pub course_reference_position: [f64; 3],
    pub sample_result_zero: u8,
    /// JSON encoding of the open `course_context_id` value.
    pub course_context_id_json: KpJsonBlob,
    pub course_road: KpString64,
    pub course_segment_index: i64,
    pub course_segment_fraction: f64,
    pub course_projection: f64,
    pub course_distance: f64,
    pub course_tangent: [f64; 3],
    pub has_authority: u8,
    pub authority: KpString64,
}

impl Default for KpCheckpointRespawnCandidate {
    fn default() -> Self {
        KpCheckpointRespawnCandidate {
            course_reference_position: [0.0; 3],
            sample_result_zero: 0,
            course_context_id_json: KpJsonBlob::default(),
            course_road: KpString64::default(),
            course_segment_index: 0,
            course_segment_fraction: 0.0,
            course_projection: 0.0,
            course_distance: 0.0,
            course_tangent: [0.0; 3],
            has_authority: 0,
            authority: KpString64::default(),
        }
    }
}

impl KpCheckpointRespawnCandidate {
    /// C callback output -> rich. Strict: the respawn candidate feeds the
    /// checkpoint state, so an unparseable `course_context_id` blob or
    /// invalid UTF-8 fails the query (detail recorded by the adapter)
    /// instead of degrading to `null`/lossy bytes.
    fn to_rich(self) -> Result<CheckpointRespawnCandidate, String> {
        Ok(CheckpointRespawnCandidate {
            course_reference_position: self.course_reference_position,
            sample_result_zero: u8_bool(self.sample_result_zero),
            course_context_id: self.course_context_id_json.get_json()?,
            course_road: self.course_road.get()?,
            course_segment_index: self.course_segment_index,
            course_segment_fraction: self.course_segment_fraction,
            course_projection: self.course_projection,
            course_distance: self.course_distance,
            course_tangent: self.course_tangent,
            authority: if u8_bool(self.has_authority) {
                Some(self.authority.get()?)
            } else {
                None
            },
        })
    }
}

/// Scalar [`JsonValue`] mirror (null/bool/number/string) for the closed
/// [`KpRailEntryBoundaryMatch`] fields whose runtime evidence carries more
/// than one scalar type (`context_id` is int-or-string). Arrays/objects are
/// not representable; a `tag` outside `KP_JSON_SCALAR_*` makes the
/// containing query fail.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpJsonScalar {
    /// `KP_JSON_SCALAR_*` tag.
    pub tag: u8,
    /// Valid when `tag == KP_JSON_SCALAR_BOOL`.
    pub bool_value: u8,
    /// Valid when `tag == KP_JSON_SCALAR_NUMBER`.
    pub number_value: f64,
    /// Valid when `tag == KP_JSON_SCALAR_STRING`.
    pub string_value: KpString64,
}

/// `KpJsonScalar` tag: JSON null.
pub const KP_JSON_SCALAR_NULL: u8 = 0;
/// `KpJsonScalar` tag: JSON true/false.
pub const KP_JSON_SCALAR_BOOL: u8 = 1;
/// `KpJsonScalar` tag: JSON number (f64).
pub const KP_JSON_SCALAR_NUMBER: u8 = 2;
/// `KpJsonScalar` tag: JSON string.
pub const KP_JSON_SCALAR_STRING: u8 = 3;

impl Default for KpJsonScalar {
    fn default() -> Self {
        KpJsonScalar {
            tag: KP_JSON_SCALAR_NULL,
            bool_value: 0,
            number_value: 0.0,
            string_value: KpString64::default(),
        }
    }
}

impl KpJsonScalar {
    /// Strict import: an out-of-range tag or invalid UTF-8 string fails the
    /// containing query (the rail entry match can be latched into state).
    fn to_rich(self) -> Result<JsonValue, String> {
        match self.tag {
            KP_JSON_SCALAR_NULL => Ok(JsonValue::Null),
            KP_JSON_SCALAR_BOOL => Ok(JsonValue::Bool(u8_bool(self.bool_value))),
            KP_JSON_SCALAR_NUMBER => Ok(JsonValue::Number(self.number_value)),
            KP_JSON_SCALAR_STRING => Ok(JsonValue::String(self.string_value.get()?)),
            other => Err(format!("unknown JSON scalar tag {other}")),
        }
    }
}

/// POD mirror of [`RailEntryBoundaryMatch`] (a closed struct, not an open
/// mapping): one authored ToRoad entry-rectangle match.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KpRailEntryBoundaryMatch {
    /// Authored context id (int or string on runtime evidence).
    pub context_id: KpJsonScalar,
    pub context_index: i64,
    pub road: KpString64,
    pub element: KpJsonScalar,
    pub marker: KpJsonScalar,
    pub signed_plane_distance: f64,
    pub plane_distance: f64,
    pub projected_coordinates: [f64; 2],
    pub path_start: [f64; 3],
    pub tangent: [f64; 3],
}

impl KpRailEntryBoundaryMatch {
    /// C callback output -> rich. Strict: an out-of-range scalar tag or
    /// invalid UTF-8 fails the query (the adapter turns the `Err` into the
    /// outer-`None` query failure and records the detail).
    fn to_rich(self) -> Result<RailEntryBoundaryMatch, String> {
        Ok(RailEntryBoundaryMatch {
            context_id: self.context_id.to_rich()?,
            context_index: self.context_index,
            road: self.road.get()?,
            element: self.element.to_rich()?,
            marker: self.marker.to_rich()?,
            signed_plane_distance: self.signed_plane_distance,
            plane_distance: self.plane_distance,
            projected_coordinates: self.projected_coordinates,
            path_start: self.path_start,
            tangent: self.tangent,
        })
    }
}

/// Required ground query. Fills `out` with the ground row for the probe
/// ray (`position`, `up`, `min_dot` — the engine's default is 0.65).
pub type KpQueryGroundFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    up: *const f64,
    min_dot: f64,
    out: *mut KpWorldGroundResult,
);

/// List-returning candidate query (`sweep_obb_rotating_all`,
/// `overlap_obb_all`). Writes up to `out_capacity` candidates into `out`
/// and returns the TOTAL candidate count. If the total exceeds
/// `out_capacity`, the engine allocates a larger buffer and calls the
/// callback a second time. NULL = capability absent.
pub type KpSweepObbRotatingAllFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    start: *const f64,
    end: *const f64,
    start_matrix: *const f64,
    end_matrix: *const f64,
    half_extents: *const f64,
    out: *mut KpWorldObbOverlap,
    out_capacity: usize,
) -> usize;

/// Single-candidate sweep fallback. Returns 1 and fills `out` on a hit,
/// 0 otherwise. NULL = capability absent.
pub type KpSweepObbRotatingFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    start: *const f64,
    end: *const f64,
    start_matrix: *const f64,
    end_matrix: *const f64,
    half_extents: *const f64,
    out: *mut KpWorldObbOverlap,
) -> i32;

/// `overlap_obb_all(center, matrix, half_extents)`; same two-call protocol
/// as [`KpSweepObbRotatingAllFn`]. NULL = capability absent (which
/// contractually releases `takeoff_separation_pending` immediately and
/// falls back to wheel-manifold landing — handoff §4).
pub type KpOverlapObbAllFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    center: *const f64,
    matrix: *const f64,
    half_extents: *const f64,
    out: *mut KpWorldObbOverlap,
    out_capacity: usize,
) -> usize;

/// `query_takeoff_contact` / `query_landing_contact`. Returns 1 and fills
/// `out` when a candidate exists, 0 for no candidate. `support_planes`
/// (count `support_plane_count`) and `active_slots` (count
/// `active_slot_count`, 0/1 values) may be NULL when their count is 0.
/// NULL = capability absent.
pub type KpQueryTakeoffLandingContactFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    velocity: *const f64,
    support_planes: *const KpWorldGroundResult,
    support_plane_count: usize,
    active_slots: *const i64,
    active_slot_count: usize,
    out: *mut KpTakeoffLandingContact,
) -> i32;

/// `query_checkpoint_target_candidate(position, radius)`. Returns 1 and
/// fills `out` on a candidate, 0 otherwise. NULL = capability absent.
pub type KpQueryCheckpointTargetCandidateFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    radius: f64,
    out: *mut KpCheckpointTargetCandidate,
) -> i32;

/// `query_checkpoint_respawn(position, checkpoint_target, checkpoint_gate,
/// mode_family, max_course_distance)`. `mode_family` is a NUL-terminated
/// string. Returns 1 and fills `out` on a candidate, 0 otherwise.
/// NULL = capability absent.
pub type KpQueryCheckpointRespawnFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    checkpoint_target: *const f64,
    checkpoint_gate: u8,
    mode_family: *const c_char,
    max_course_distance: f64,
    out: *mut KpCheckpointRespawnCandidate,
) -> i32;

/// `sweep_point(start, end)` — the A5B840 point-segment query. Returns
/// non-zero and fills `out` with the JSON-encoded candidate mapping, or 0
/// to report no result (the adapter then reports the query as failed and
/// the engine surfaces "sweep_point capability reported present but
/// returned no result"; Python requires a mapping return). The mapping
/// must be a JSON object. NULL = capability absent.
pub type KpSweepPointFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    start: *const f64,
    end: *const f64,
    out: *mut KpJsonBlob,
) -> i32;

/// `query_rail_contact(position, previous_rail_state, owner_id)`.
/// `owner_id` is a NUL-terminated string or NULL for None. Returns
/// non-zero and fills `out` with the JSON-encoded selector mapping (must
/// be a JSON object), or 0 when the query reports no rail contact (Python
/// `None`). NULL = capability absent.
pub type KpQueryRailContactFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    previous_rail_state: i64,
    owner_id: *const c_char,
    out: *mut KpJsonBlob,
) -> i32;

/// `query_rail_entry_boundary(position, road, plane_tolerance)`. `road` is
/// a NUL-terminated string or NULL for None. Returns non-zero and fills
/// `out` with the typed match mirror, or 0 when no entry rectangle matches
/// (Python `None`). NULL = capability absent.
pub type KpQueryRailEntryBoundaryFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    road: *const c_char,
    plane_tolerance: f64,
    out: *mut KpRailEntryBoundaryMatch,
) -> i32;

/// `rail_query_contexts()` / `rail_entry_contexts()`. Writes up to
/// `out_capacity` JSON-encoded context mappings (one JSON object per
/// element) into `out` and returns the TOTAL context count; the engine
/// calls a second time with a larger buffer when the total exceeds
/// `out_capacity` (the same two-call protocol as
/// [`KpSweepObbRotatingAllFn`], with the same documented count ceilings).
/// NULL = capability absent.
pub type KpRailContextListFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    out: *mut KpJsonBlob,
    out_capacity: usize,
) -> usize;

/// `query_moving_contact(position, time_ms, object_id, delta_ms,
/// max_distance)`. `object_id` is a NUL-terminated string or NULL for
/// None. Returns non-zero and fills `out` with the JSON-encoded record
/// mapping (must be a JSON object), or 0 when the world has no candidate
/// (Python `None`). NULL = capability absent.
pub type KpQueryMovingContactFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    position: *const f64,
    time_ms: i64,
    object_id: *const c_char,
    delta_ms: i64,
    max_distance: f64,
    out: *mut KpJsonBlob,
) -> i32;

/// `query_network_correction(position)`. Returns non-zero and fills `out`
/// with the JSON-encoded correction packet mapping (must be a JSON
/// object), or 0 when the world has none pending (Python `None`).
/// NULL = capability absent.
pub type KpQueryNetworkCorrectionFn =
    unsafe extern "C" fn(user_data: *mut c_void, position: *const f64, out: *mut KpJsonBlob) -> i32;

/// `query_distinct_water_volume(position)`. Returns non-zero and fills
/// `out` with the JSON-encoded selector value (any JSON value; Python only
/// takes `bool(...)` of it), or 0 for Python `None`.
/// NULL = capability absent.
pub type KpQueryDistinctWaterVolumeFn =
    unsafe extern "C" fn(user_data: *mut c_void, position: *const f64, out: *mut KpJsonBlob) -> i32;

/// The frozen world vtable: the C mapping of the [`World`] trait.
/// `query_ground` is required; every optional function pointer that is
/// NULL means *capability absent*, exactly like Python's `hasattr`
/// probes (the engine branches on presence, so keep the pointers in sync
/// with what the world actually implements). Open-mapping payloads cross
/// as JSON documents in [`KpJsonBlob`] out-params; the closed rail entry
/// match crosses as [`KpRailEntryBoundaryMatch`]. Layout is append-only:
/// new capability slots are added at the tail together with a
/// [`KP_ABI_VERSION`] bump (v1 ended at `query_checkpoint_respawn`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KpWorldVtable {
    /// Opaque pointer passed back to every callback.
    pub user_data: *mut c_void,
    /// Non-zero: sweep at the step-start pose instead of the step-end pose
    /// (Python `getattr(world, "query_at_step_start", False)`).
    pub query_at_step_start: u8,
    /// Required. Passing a vtable with NULL `query_ground` is
    /// `KP_ERR_INVALID_INPUT`.
    pub query_ground: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            up: *const f64,
            min_dot: f64,
            out: *mut KpWorldGroundResult,
        ),
    >,
    pub sweep_obb_rotating_all: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            start: *const f64,
            end: *const f64,
            start_matrix: *const f64,
            end_matrix: *const f64,
            half_extents: *const f64,
            out: *mut KpWorldObbOverlap,
            out_capacity: usize,
        ) -> usize,
    >,
    pub sweep_obb_rotating: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            start: *const f64,
            end: *const f64,
            start_matrix: *const f64,
            end_matrix: *const f64,
            half_extents: *const f64,
            out: *mut KpWorldObbOverlap,
        ) -> i32,
    >,
    pub overlap_obb_all: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            center: *const f64,
            matrix: *const f64,
            half_extents: *const f64,
            out: *mut KpWorldObbOverlap,
            out_capacity: usize,
        ) -> usize,
    >,
    pub query_takeoff_contact: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            velocity: *const f64,
            support_planes: *const KpWorldGroundResult,
            support_plane_count: usize,
            active_slots: *const i64,
            active_slot_count: usize,
            out: *mut KpTakeoffLandingContact,
        ) -> i32,
    >,
    pub query_landing_contact: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            velocity: *const f64,
            support_planes: *const KpWorldGroundResult,
            support_plane_count: usize,
            active_slots: *const i64,
            active_slot_count: usize,
            out: *mut KpTakeoffLandingContact,
        ) -> i32,
    >,
    pub query_checkpoint_target_candidate: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            radius: f64,
            out: *mut KpCheckpointTargetCandidate,
        ) -> i32,
    >,
    pub query_checkpoint_respawn: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            checkpoint_target: *const f64,
            checkpoint_gate: u8,
            mode_family: *const c_char,
            max_course_distance: f64,
            out: *mut KpCheckpointRespawnCandidate,
        ) -> i32,
    >,
    // -- ABI v2 tail: the eight open-mapping capabilities (all NULL in a
    // v1-initialized vtable means capability absent, exactly the v1
    // carve-out behavior). See the Kp*Fn typedefs for the contracts.
    pub sweep_point: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            start: *const f64,
            end: *const f64,
            out: *mut KpJsonBlob,
        ) -> i32,
    >,
    pub query_rail_contact: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            previous_rail_state: i64,
            owner_id: *const c_char,
            out: *mut KpJsonBlob,
        ) -> i32,
    >,
    pub query_rail_entry_boundary: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            road: *const c_char,
            plane_tolerance: f64,
            out: *mut KpRailEntryBoundaryMatch,
        ) -> i32,
    >,
    pub rail_query_contexts: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            out: *mut KpJsonBlob,
            out_capacity: usize,
        ) -> usize,
    >,
    pub rail_entry_contexts: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            out: *mut KpJsonBlob,
            out_capacity: usize,
        ) -> usize,
    >,
    pub query_moving_contact: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            time_ms: i64,
            object_id: *const c_char,
            delta_ms: i64,
            max_distance: f64,
            out: *mut KpJsonBlob,
        ) -> i32,
    >,
    pub query_network_correction: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            out: *mut KpJsonBlob,
        ) -> i32,
    >,
    pub query_distinct_water_volume: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            position: *const f64,
            out: *mut KpJsonBlob,
        ) -> i32,
    >,
}

/// Size in bytes of the ABI v1 [`KpWorldVtable`] prefix (everything up to
/// but excluding `sweep_point`): a host compiled against the v1 header owns
/// exactly this many bytes. A v1-negotiated engine ([`kart_physics_create`])
/// copies only this prefix out of the host vtable and leaves the v2-only
/// tail slots NULL, so no out-of-bounds read can happen. Pinned against the
/// real layout in `ffi::tests::layout_is_stable`.
pub const KP_WORLD_VTABLE_V1_BYTES: usize = 80;

impl std::fmt::Debug for KpWorldVtable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KpWorldVtable")
            .field("user_data", &self.user_data)
            .field("query_at_step_start", &self.query_at_step_start)
            .field("query_ground", &self.query_ground.is_some())
            .field(
                "sweep_obb_rotating_all",
                &self.sweep_obb_rotating_all.is_some(),
            )
            .field("sweep_obb_rotating", &self.sweep_obb_rotating.is_some())
            .field("overlap_obb_all", &self.overlap_obb_all.is_some())
            .field(
                "query_takeoff_contact",
                &self.query_takeoff_contact.is_some(),
            )
            .field(
                "query_landing_contact",
                &self.query_landing_contact.is_some(),
            )
            .field(
                "query_checkpoint_target_candidate",
                &self.query_checkpoint_target_candidate.is_some(),
            )
            .field(
                "query_checkpoint_respawn",
                &self.query_checkpoint_respawn.is_some(),
            )
            .field("sweep_point", &self.sweep_point.is_some())
            .field("query_rail_contact", &self.query_rail_contact.is_some())
            .field(
                "query_rail_entry_boundary",
                &self.query_rail_entry_boundary.is_some(),
            )
            .field("rail_query_contexts", &self.rail_query_contexts.is_some())
            .field("rail_entry_contexts", &self.rail_entry_contexts.is_some())
            .field("query_moving_contact", &self.query_moving_contact.is_some())
            .field(
                "query_network_correction",
                &self.query_network_correction.is_some(),
            )
            .field(
                "query_distinct_water_volume",
                &self.query_distinct_water_volume.is_some(),
            )
            .finish()
    }
}

/// [`World`] implementation that calls through a [`KpWorldVtable`].
struct CWorldAdapter {
    vtable: KpWorldVtable,
    /// Detail of the most recent adapter-side query failure (JSON codec
    /// offset/reason, invalid UTF-8, embedded NUL, clamp violation). The
    /// `World` trait has no error channel, so [`kart_physics_step`] appends
    /// this to the engine's generic "capability reported present but
    /// returned no result" `KP_ERR_STEP` message instead of letting the
    /// diagnostics die inside the adapter.
    last_query_error: RefCell<Option<String>>,
}

impl CWorldAdapter {
    fn new(vtable: KpWorldVtable) -> Self {
        CWorldAdapter {
            vtable,
            last_query_error: RefCell::new(None),
        }
    }

    /// Record why a callback payload was rejected (last failure wins: the
    /// engine raises at the first failed query, so the newest record is the
    /// relevant one).
    fn note_query_failure(&self, detail: String) {
        *self.last_query_error.borrow_mut() = Some(detail);
    }

    fn take_query_error(&self) -> Option<String> {
        self.last_query_error.borrow_mut().take()
    }

    /// Map a strict callback-payload import (`Result`) onto the trait's
    /// query-failure idiom (`None`), recording the detail on the way.
    fn strict_import<T>(&self, result: Result<T, String>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.note_query_failure(error);
                None
            }
        }
    }

    /// Parse a callback-filled JSON blob that must hold an object mapping.
    /// `None` (with the detail recorded) on any parse failure or non-object
    /// payload — the caller reports the query as failed.
    fn parse_blob_object(&self, blob: &KpJsonBlob) -> Option<BTreeMap<String, JsonValue>> {
        match blob.get_json() {
            Ok(JsonValue::Object(entries)) => Some(entries),
            Ok(_) => {
                self.note_query_failure("world callback JSON payload is not an object".to_string());
                None
            }
            Err(error) => {
                self.note_query_failure(error);
                None
            }
        }
    }

    /// Shared CString conversion for the engine -> callback string
    /// arguments: an embedded NUL fails the query (outer `None`, detail
    /// recorded) instead of looking like "query answered no result".
    fn c_string_arg(&self, what: &str, value: &str) -> Option<std::ffi::CString> {
        match std::ffi::CString::new(value) {
            Ok(converted) => Some(converted),
            Err(error) => {
                self.note_query_failure(format!("{what} contains an embedded NUL: {error}"));
                None
            }
        }
    }

    fn call_takeoff_landing(
        &self,
        callback: KpQueryTakeoffLandingContactFn,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<TakeoffLandingContact> {
        let planes: Vec<KpWorldGroundResult> = support_planes
            .iter()
            .map(rich_ground_result_to_pod_lossy)
            .collect();
        let mut out = KpTakeoffLandingContact::default();
        // Safety: slices are valid for their lengths; `out` is a valid
        // write target.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                velocity.as_ptr(),
                planes.as_ptr(),
                planes.len(),
                active_slots.as_ptr(),
                active_slots.len(),
                &mut out,
            )
        };
        if present != 0 {
            self.strict_import(out.to_rich())
        } else {
            None
        }
    }

    /// Shared tail of the two JSON-blob list capabilities: two-call
    /// protocol with the documented count clamp, then per-element object
    /// parsing. Any protocol violation (count over the ceiling, invalid
    /// JSON, non-object element) reports the query as failed (`None`,
    /// detail recorded), which surfaces the engine's usual capability
    /// error.
    fn call_json_blob_list(
        &self,
        callback: KpRailContextListFn,
    ) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        // Safety: `buffer` is valid for its capacity; the callback writes
        // at most `out_capacity` elements by contract.
        unsafe {
            let mut buffer = vec![KpJsonBlob::default(); LIST_STACK_CAPACITY_JSON];
            let count = callback(
                self.vtable.user_data,
                buffer.as_mut_ptr(),
                LIST_STACK_CAPACITY_JSON,
            );
            if count > LIST_STACK_CAPACITY_JSON {
                let count = self.clamped_list_count::<KpJsonBlob>(count)?;
                buffer = vec![KpJsonBlob::default(); count];
                let second = callback(self.vtable.user_data, buffer.as_mut_ptr(), count);
                buffer.truncate(second.min(count));
            } else {
                buffer.truncate(count);
            }
            buffer
                .iter()
                .map(|blob| self.parse_blob_object(blob))
                .collect::<Option<Vec<_>>>()
        }
    }

    /// [`clamped_list_count`] plus the query-failure detail record.
    fn clamped_list_count<T>(&self, count: usize) -> Option<usize> {
        match clamped_list_count::<T>(count) {
            Some(count) => Some(count),
            None => {
                self.note_query_failure(format!(
                    "callback-reported list count {count} exceeds the documented \
                     ceilings (KP_LIST_MAX_COUNT={KP_LIST_MAX_COUNT}, \
                     KP_LIST_MAX_BUFFER_BYTES={KP_LIST_MAX_BUFFER_BYTES})"
                ));
                None
            }
        }
    }
}

/// Rich -> POD for the support-plane arguments the engine passes back to
/// the world. Lossy only in the >64-byte string case (which the engine
/// never produces: world rows originate from the C side).
fn rich_ground_result_to_pod_lossy(result: &WorldGroundResult) -> KpWorldGroundResult {
    fn pod_string(text: &str) -> KpString64 {
        KpString64::set(text).unwrap_or_else(|_| {
            let mut truncated = KpString64::default();
            let cut = text
                .char_indices()
                .take_while(|(index, _)| *index + 4 <= KP_STRING_CAPACITY as usize)
                .last()
                .map_or(0, |(index, ch)| index + ch.len_utf8());
            truncated.data[..cut].copy_from_slice(&text.as_bytes()[..cut]);
            truncated.len = cut as u8;
            truncated
        })
    }
    KpWorldGroundResult {
        track_id: pod_string(&result.track_id),
        node: pod_string(&result.node),
        tri_index: result.tri_index,
        road_surface: pod_string(&result.road_surface),
        surface_id: result.surface_id,
        road_rail: pod_string(&result.road_rail),
        distance: result.distance,
        point: result.point,
        normal: result.normal,
        has_triangle: bool_u8(result.triangle.is_some()),
        triangle: match result.triangle {
            Some(vertices) => [
                vertices[0][0],
                vertices[0][1],
                vertices[0][2],
                vertices[1][0],
                vertices[1][1],
                vertices[1][2],
                vertices[2][0],
                vertices[2][1],
                vertices[2][2],
            ],
            None => [0.0; 9],
        },
        has_contact_basis_44: bool_u8(result.contact_basis_44.is_some()),
        contact_basis_44: result.contact_basis_44.unwrap_or([0.0; 3]),
    }
}

impl World for CWorldAdapter {
    fn capabilities(&self) -> WorldCapabilities {
        WorldCapabilities {
            overlap_obb_all: self.vtable.overlap_obb_all.is_some(),
            // The frozen C vtable predates scoped ToMovableObject queries.
            // TrackWorld owns this internal capability; exposing it through C
            // requires an intentional ABI-version bump.
            query_event_obstacle_contacts: false,
            // The frozen C vtable has no animated item-effect callback.
            query_banana_trigger: false,
            query_takeoff_contact: self.vtable.query_takeoff_contact.is_some(),
            query_landing_contact: self.vtable.query_landing_contact.is_some(),
            query_checkpoint_target_candidate: self
                .vtable
                .query_checkpoint_target_candidate
                .is_some(),
            query_checkpoint_respawn: self.vtable.query_checkpoint_respawn.is_some(),
            sweep_obb_rotating_all: self.vtable.sweep_obb_rotating_all.is_some(),
            sweep_obb_rotating: self.vtable.sweep_obb_rotating.is_some(),
            query_at_step_start: u8_bool(self.vtable.query_at_step_start),
            query_rail_contact: self.vtable.query_rail_contact.is_some(),
            query_rail_entry_boundary: self.vtable.query_rail_entry_boundary.is_some(),
            query_moving_contact: self.vtable.query_moving_contact.is_some(),
            sweep_point: self.vtable.sweep_point.is_some(),
            rail_query_contexts: self.vtable.rail_query_contexts.is_some(),
            rail_entry_contexts: self.vtable.rail_entry_contexts.is_some(),
            query_network_correction: self.vtable.query_network_correction.is_some(),
            query_distinct_water_volume: self.vtable.query_distinct_water_volume.is_some(),
        }
    }

    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult {
        let Some(callback) = self.vtable.query_ground else {
            // Unreachable: `kart_physics_step` rejects NULL query_ground.
            return WorldGroundResult::default();
        };
        let mut out = KpWorldGroundResult::default();
        // Safety: `out` is a valid write target; the input pointers are
        // valid 3-element arrays.
        unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                up.as_ptr(),
                min_dot,
                &mut out,
            );
        }
        out.to_rich_lossy()
    }

    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let callback = self.vtable.sweep_obb_rotating_all?;
        // Safety: all array pointers are valid for their element counts.
        unsafe {
            let mut buffer = vec![KpWorldObbOverlap::default(); LIST_STACK_CAPACITY];
            let count = callback(
                self.vtable.user_data,
                start.as_ptr(),
                end.as_ptr(),
                start_matrix.as_ptr(),
                end_matrix.as_ptr(),
                half_extents.as_ptr(),
                buffer.as_mut_ptr(),
                LIST_STACK_CAPACITY,
            );
            if count > LIST_STACK_CAPACITY {
                // Clamp the untrusted callback count (handoff §3.1): a
                // garbage count must fail the query, never allocate
                // unbounded memory (alloc failure aborts the host).
                let count = self.clamped_list_count::<KpWorldObbOverlap>(count)?;
                buffer = vec![KpWorldObbOverlap::default(); count];
                let second = callback(
                    self.vtable.user_data,
                    start.as_ptr(),
                    end.as_ptr(),
                    start_matrix.as_ptr(),
                    end_matrix.as_ptr(),
                    half_extents.as_ptr(),
                    buffer.as_mut_ptr(),
                    count,
                );
                buffer.truncate(second.min(count));
            } else {
                buffer.truncate(count);
            }
            self.strict_import(buffer.iter().map(|item| item.to_rich()).collect())
        }
    }

    fn sweep_obb_rotating(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<WorldObbOverlap> {
        let callback = self.vtable.sweep_obb_rotating?;
        let mut out = KpWorldObbOverlap::default();
        // Safety: all pointers are valid.
        let hit = unsafe {
            callback(
                self.vtable.user_data,
                start.as_ptr(),
                end.as_ptr(),
                start_matrix.as_ptr(),
                end_matrix.as_ptr(),
                half_extents.as_ptr(),
                &mut out,
            )
        };
        if hit != 0 {
            self.strict_import(out.to_rich())
        } else {
            None
        }
    }

    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let callback = self.vtable.overlap_obb_all?;
        // Safety: all array pointers are valid for their element counts.
        unsafe {
            let mut buffer = vec![KpWorldObbOverlap::default(); LIST_STACK_CAPACITY];
            let count = callback(
                self.vtable.user_data,
                center.as_ptr(),
                matrix.as_ptr(),
                half_extents.as_ptr(),
                buffer.as_mut_ptr(),
                LIST_STACK_CAPACITY,
            );
            if count > LIST_STACK_CAPACITY {
                // Clamp the untrusted callback count (handoff §3.1), same
                // as `sweep_obb_rotating_all`.
                let count = self.clamped_list_count::<KpWorldObbOverlap>(count)?;
                buffer = vec![KpWorldObbOverlap::default(); count];
                let second = callback(
                    self.vtable.user_data,
                    center.as_ptr(),
                    matrix.as_ptr(),
                    half_extents.as_ptr(),
                    buffer.as_mut_ptr(),
                    count,
                );
                buffer.truncate(second.min(count));
            } else {
                buffer.truncate(count);
            }
            self.strict_import(buffer.iter().map(|item| item.to_rich()).collect())
        }
    }

    fn query_takeoff_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let callback = self.vtable.query_takeoff_contact?;
        Some(self.call_takeoff_landing(callback, position, velocity, support_planes, active_slots))
    }

    fn query_landing_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let callback = self.vtable.query_landing_contact?;
        Some(self.call_takeoff_landing(callback, position, velocity, support_planes, active_slots))
    }

    fn query_checkpoint_target_candidate(
        &self,
        position: [f64; 3],
        radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        let callback = self.vtable.query_checkpoint_target_candidate?;
        let mut out = KpCheckpointTargetCandidate::default();
        // Safety: `out` is valid; `position` points at 3 f64.
        let present =
            unsafe { callback(self.vtable.user_data, position.as_ptr(), radius, &mut out) };
        if present != 0 {
            self.strict_import(out.to_rich().map(Some))
        } else {
            Some(None)
        }
    }

    fn query_checkpoint_respawn(
        &self,
        position: [f64; 3],
        checkpoint_target: [f64; 3],
        checkpoint_gate: bool,
        mode_family: &str,
        max_course_distance: f64,
    ) -> Option<Option<CheckpointRespawnCandidate>> {
        let callback = self.vtable.query_checkpoint_respawn?;
        // An embedded NUL fails the query (outer None), it does not look
        // like "query answered no result".
        let mode_family_c =
            self.c_string_arg("query_checkpoint_respawn mode_family", mode_family)?;
        let mut out = KpCheckpointRespawnCandidate::default();
        // Safety: `out` is valid; the array pointers are valid 3-element
        // arrays; `mode_family_c` is NUL-terminated.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                checkpoint_target.as_ptr(),
                bool_u8(checkpoint_gate),
                mode_family_c.as_ptr(),
                max_course_distance,
                &mut out,
            )
        };
        if present != 0 {
            self.strict_import(out.to_rich().map(Some))
        } else {
            Some(None)
        }
    }

    fn sweep_point(&self, start: [f64; 3], end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        let callback = self.vtable.sweep_point?;
        let mut out = KpJsonBlob::default();
        // Safety: `out` is a valid write target; the arrays are 3 f64 each.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                start.as_ptr(),
                end.as_ptr(),
                &mut out,
            )
        };
        if present == 0 {
            // Python requires a mapping return; a C callback reporting no
            // result fails the query (the engine raises "sweep_point
            // capability reported present but returned no result").
            return None;
        }
        self.parse_blob_object(&out)
    }

    fn query_rail_contact(
        &self,
        position: [f64; 3],
        previous_rail_state: i64,
        owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let callback = self.vtable.query_rail_contact?;
        let owner_id_c = match owner_id {
            Some(owner) => Some(self.c_string_arg("query_rail_contact owner_id", owner)?),
            None => None,
        };
        let mut out = KpJsonBlob::default();
        // Safety: `out` is valid; `position` points at 3 f64; `owner_id_c`
        // is NUL-terminated when present.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                previous_rail_state,
                owner_id_c
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                &mut out,
            )
        };
        if present == 0 {
            return Some(None);
        }
        self.parse_blob_object(&out).map(Some)
    }

    fn query_rail_entry_boundary(
        &self,
        position: [f64; 3],
        road: Option<&str>,
        plane_tolerance: f64,
    ) -> Option<Option<RailEntryBoundaryMatch>> {
        let callback = self.vtable.query_rail_entry_boundary?;
        let road_c = match road {
            Some(road) => Some(self.c_string_arg("query_rail_entry_boundary road", road)?),
            None => None,
        };
        let mut out = KpRailEntryBoundaryMatch::default();
        // Safety: `out` is valid; `position` points at 3 f64; `road_c` is
        // NUL-terminated when present.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                road_c
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                plane_tolerance,
                &mut out,
            )
        };
        if present == 0 {
            return Some(None);
        }
        self.strict_import(out.to_rich().map(Some))
    }

    fn rail_query_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        let callback = self.vtable.rail_query_contexts?;
        self.call_json_blob_list(callback)
    }

    fn rail_entry_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        let callback = self.vtable.rail_entry_contexts?;
        self.call_json_blob_list(callback)
    }

    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let callback = self.vtable.query_moving_contact?;
        let object_id_c = match object_id {
            Some(object_id) => {
                Some(self.c_string_arg("query_moving_contact object_id", object_id)?)
            }
            None => None,
        };
        let mut out = KpJsonBlob::default();
        // Safety: `out` is valid; `position` points at 3 f64; `object_id_c`
        // is NUL-terminated when present.
        let present = unsafe {
            callback(
                self.vtable.user_data,
                position.as_ptr(),
                time_ms,
                object_id_c
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                delta_ms,
                max_distance,
                &mut out,
            )
        };
        if present == 0 {
            return Some(None);
        }
        self.parse_blob_object(&out).map(Some)
    }

    fn query_network_correction(
        &self,
        position: [f64; 3],
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let callback = self.vtable.query_network_correction?;
        let mut out = KpJsonBlob::default();
        // Safety: `out` is valid; `position` points at 3 f64.
        let present = unsafe { callback(self.vtable.user_data, position.as_ptr(), &mut out) };
        if present == 0 {
            return Some(None);
        }
        self.parse_blob_object(&out).map(Some)
    }

    fn query_distinct_water_volume(&self, position: [f64; 3]) -> Option<Option<JsonValue>> {
        let callback = self.vtable.query_distinct_water_volume?;
        let mut out = KpJsonBlob::default();
        // Safety: `out` is valid; `position` points at 3 f64.
        let present = unsafe { callback(self.vtable.user_data, position.as_ptr(), &mut out) };
        if present == 0 {
            return Some(None);
        }
        // Any JSON value is legal here (Python only takes `bool(...)` of
        // it); a parse failure fails the query (detail recorded).
        self.strict_import(out.get_json().map(Some))
    }
}

// ---------------------------------------------------------------------------
// Engine handle + exported functions
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(message: String) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        format!("rust panic across FFI: {text}")
    } else if let Some(text) = payload.downcast_ref::<String>() {
        format!("rust panic across FFI: {text}")
    } else {
        "rust panic across FFI (unknown payload)".to_string()
    }
}

/// Run an FFI body returning a status code with panic containment.
/// `body` returns `Ok(KP_OK)` or `Err((status, message))`; panics become
/// `KP_ERR_PANIC`. The thread-local message is cleared on success.
fn ffi_guard_status(body: impl FnOnce() -> Result<i32, (i32, String)>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(status)) => {
            clear_last_error();
            status
        }
        Ok(Err((status, message))) => {
            debug_assert_ne!(status, KP_OK);
            set_last_error(message);
            status
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            KP_ERR_PANIC
        }
    }
}

/// Pointer-returning variant ([`kart_physics_create`]): errors and panics
/// both yield NULL with the thread-local message set.
fn ffi_guard_ptr<T>(body: impl FnOnce() -> Result<*mut T, (i32, String)>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(pointer)) => {
            clear_last_error();
            pointer
        }
        Ok(Err((_, message))) => {
            set_last_error(message);
            std::ptr::null_mut()
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            std::ptr::null_mut()
        }
    }
}

/// Opaque engine handle (`KartRiderVehicleEngine` + numeric mode + the M3
/// prediction history + the negotiated ABI version). Create with
/// [`kart_physics_create_v2`] (v2 hosts) or the v1 compatibility shim
/// [`kart_physics_create`], release with [`kart_physics_destroy`]. Not
/// thread-safe.
pub struct KartPhysicsEngine {
    engine: KartRiderVehicleEngine,
    /// ABI layout the host was created against (1 = frozen v1 shim;
    /// [`KP_ABI_VERSION`] = current). Every state, parameter, input, and
    /// vtable access is bounded by this value.
    negotiated_abi: u32,
}

/// ABI version (`KP_ABI_VERSION`).
#[no_mangle]
pub extern "C" fn kart_physics_abi_version() -> u32 {
    KP_ABI_VERSION
}

/// Copy the thread-local error message of the most recent failed call on
/// this thread into `buffer` (NUL-terminated when `capacity > 0`). Returns
/// the full message length in bytes, excluding the NUL; if the return
/// value is >= `capacity`, the message was truncated. `buffer` may be
/// NULL with `capacity == 0` to query the length.
///
/// # Safety
/// `buffer` must be writable for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_last_error_message(
    buffer: *mut c_char,
    capacity: usize,
) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let bytes = message.as_bytes();
        if !buffer.is_null() && capacity > 0 {
            let copy_len = bytes.len().min(capacity - 1);
            // Safety: caller guarantees `capacity` writable bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), copy_len);
                *buffer.add(copy_len) = 0;
            }
        }
        bytes.len()
    })
}

/// Frozen ABI v1 default-parameter writer.
///
/// This unsuffixed symbol predates ABI negotiation. It therefore writes
/// exactly [`KP_VEHICLE_PARAMETERS_V1_BYTES`] and is safe for a buffer
/// allocated by the ABI v1 header. Current hosts MUST call
/// [`kart_physics_parameters_default_v2`] instead.
///
/// # Safety
/// `out` must point at at least [`KP_VEHICLE_PARAMETERS_V1_BYTES`] writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_parameters_default(out: *mut KpVehicleParameters) -> i32 {
    ffi_guard_status(|| {
        if out.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_parameters_default: NULL out".to_string(),
            ));
        }
        let parameters = KpVehicleParameters::default();
        // Safety: checked non-null; this legacy entry point requires exactly
        // the frozen v1 prefix documented above.
        unsafe { write_parameters_v1_prefix(out, &parameters) };
        Ok(KP_OK)
    })
}

/// Current-ABI default-parameter writer.
///
/// `abi_version` must equal [`KP_ABI_VERSION`]; mismatch is refused before
/// `out` is touched.
///
/// # Safety
/// `out` must point at a writable current [`KpVehicleParameters`].
#[no_mangle]
pub unsafe extern "C" fn kart_physics_parameters_default_v2(
    out: *mut KpVehicleParameters,
    abi_version: u32,
) -> i32 {
    ffi_guard_status(|| {
        if out.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_parameters_default_v2: NULL out".to_string(),
            ));
        }
        if abi_version != KP_ABI_VERSION {
            return Err((
                KP_ERR_ABI_VERSION,
                format!(
                    "kart_physics_parameters_default_v2: host ABI version {abi_version} does not \
                     match library KP_ABI_VERSION {KP_ABI_VERSION} (KP_ERR_ABI_VERSION)"
                ),
            ));
        }
        unsafe { std::ptr::write(out, KpVehicleParameters::default()) };
        Ok(KP_OK)
    })
}

/// Frozen ABI v1 default-state writer.
///
/// This unsuffixed symbol writes the historical 4096-byte-JSON layout,
/// exactly [`KP_VEHICLE_STATE_V1_BYTES`] bytes. Current hosts MUST call
/// [`kart_physics_state_default_v2`] instead.
///
/// # Safety
/// `out` must point at at least [`KP_VEHICLE_STATE_V1_BYTES`] writable bytes
/// with 8-byte alignment.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_state_default(out: *mut KpVehicleState) -> i32 {
    ffi_guard_status(|| {
        if out.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_state_default: NULL out".to_string(),
            ));
        }
        let state = KpVehicleState::default();
        unsafe { write_state_for_abi(out, &state, 1) }
            .map_err(|error| (error.status_code(), error.into_message()))?;
        Ok(KP_OK)
    })
}

/// Current-ABI default-state writer.
///
/// `abi_version` must equal [`KP_ABI_VERSION`]; mismatch is refused before
/// `out` is touched.
///
/// # Safety
/// `out` must point at a writable current [`KpVehicleState`].
#[no_mangle]
pub unsafe extern "C" fn kart_physics_state_default_v2(
    out: *mut KpVehicleState,
    abi_version: u32,
) -> i32 {
    ffi_guard_status(|| {
        if out.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_state_default_v2: NULL out".to_string(),
            ));
        }
        if abi_version != KP_ABI_VERSION {
            return Err((
                KP_ERR_ABI_VERSION,
                format!(
                    "kart_physics_state_default_v2: host ABI version {abi_version} does not \
                     match library KP_ABI_VERSION {KP_ABI_VERSION} (KP_ERR_ABI_VERSION)"
                ),
            ));
        }
        unsafe { std::ptr::write(out, KpVehicleState::default()) };
        Ok(KP_OK)
    })
}

/// Shared body of the create entry points: build the engine and tag it
/// with the ABI layout the host negotiated.
///
/// # Safety
/// `initial_state`, when non-NULL, must point at a readable state object of
/// the negotiated layout.
unsafe fn create_engine_handle(
    initial_state: *const KpVehicleState,
    numeric_mode: u32,
    negotiated_abi: u32,
) -> Result<*mut KartPhysicsEngine, (i32, String)> {
    let mode = match numeric_mode {
        KP_NUMERIC_MODE_FLOAT32 => NumericMode::Float32,
        KP_NUMERIC_MODE_FLOAT64 => NumericMode::Float64,
        other => {
            return Err((
                KP_ERR_INVALID_INPUT,
                format!("unknown numeric mode {other}"),
            ))
        }
    };
    let state = if initial_state.is_null() {
        VehicleState::default()
    } else {
        // Safety: checked non-null; the public entry point defines which
        // historical layout the pointer owns.
        let pod = unsafe { read_state_for_abi(initial_state, negotiated_abi) }
            .map_err(|error| (KP_ERR_INVALID_STATE, error))?;
        pod.to_rich()
            .map_err(|error| (KP_ERR_INVALID_STATE, error))?
    };
    let engine =
        KartRiderVehicleEngine::new(&state, mode).map_err(|error| (KP_ERR_INVALID_STATE, error))?;
    Ok(Box::into_raw(Box::new(KartPhysicsEngine {
        engine,
        negotiated_abi,
    })))
}

/// Create an engine from an initial state (NULL = default state) and a
/// numeric mode (`KP_NUMERIC_MODE_FLOAT32` / `KP_NUMERIC_MODE_FLOAT64`).
/// Returns NULL on error (see `kart_physics_last_error_message`).
///
/// This is the **ABI v1 compatibility shim**: the returned engine is
/// v1-negotiated. All operations use the historical state/parameter sizes:
/// create/set/get translate exactly [`KP_VEHICLE_STATE_V1_BYTES`], step
/// copies exactly [`KP_VEHICLE_PARAMETERS_V1_BYTES`], and the world adapter
/// reads exactly [`KP_WORLD_VTABLE_V1_BYTES`]. The v2-only vtable tail is
/// treated as NULL (capability absent — exactly the v1 degradation
/// semantics). A host compiled against the v1 header can therefore run
/// against this library without any out-of-bounds read or write.
/// Current hosts MUST use [`kart_physics_create_v2`] instead; the v2-only
/// capabilities are invisible through this shim even when the vtable
/// slots are populated.
///
/// # Safety
/// `initial_state`, when non-NULL, must point at a valid ABI v1 state
/// (`KP_VEHICLE_STATE_V1_BYTES`, 8-byte aligned).
#[no_mangle]
pub unsafe extern "C" fn kart_physics_create(
    initial_state: *const KpVehicleState,
    numeric_mode: u32,
) -> *mut KartPhysicsEngine {
    ffi_guard_ptr(|| unsafe { create_engine_handle(initial_state, numeric_mode, 1) })
}

/// Create a current-ABI engine from an initial state (NULL = default state)
/// and a numeric mode (`KP_NUMERIC_MODE_FLOAT32` /
/// `KP_NUMERIC_MODE_FLOAT64`). `abi_version` must be the host header's
/// `KP_ABI_VERSION`; any other value is refused (NULL with
/// `KP_ERR_ABI_VERSION` detail in `kart_physics_last_error_message`), so a
/// host/library layout mismatch can never silently create an engine.
/// The legacy function name is frozen; engines it returns read the full v2
/// [`KpWorldVtable`] in [`kart_physics_step`]. Returns NULL on error (see
/// `kart_physics_last_error_message`).
///
/// # Safety
/// `initial_state`, when non-NULL, must point at a valid `KpVehicleState`.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_create_v2(
    initial_state: *const KpVehicleState,
    numeric_mode: u32,
    abi_version: u32,
) -> *mut KartPhysicsEngine {
    ffi_guard_ptr(|| {
        if abi_version != KP_ABI_VERSION {
            return Err((
                KP_ERR_ABI_VERSION,
                format!(
                    "kart_physics_create_v2: host ABI version {abi_version} does not \
                     match library KP_ABI_VERSION {KP_ABI_VERSION} (KP_ERR_ABI_VERSION)"
                ),
            ));
        }
        unsafe { create_engine_handle(initial_state, numeric_mode, KP_ABI_VERSION) }
    })
}

/// Release an engine created by [`kart_physics_create_v2`] or
/// [`kart_physics_create`]. NULL is a no-op.
///
/// # Safety
/// `engine` must be a handle from [`kart_physics_create_v2`] or
/// [`kart_physics_create`] that has not already been destroyed.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_destroy(engine: *mut KartPhysicsEngine) {
    if engine.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // Safety: the caller passes a live handle exactly once.
        drop(unsafe { Box::from_raw(engine) });
    }));
}

/// Snapshot the engine's current state into `out`. A v1-shim engine writes
/// exactly [`KP_VEHICLE_STATE_V1_BYTES`] in the historical layout; a current
/// engine writes the full current struct.
///
/// # Safety
/// `engine` must be a live handle; `out` must be writable for that engine's
/// negotiated state layout.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_get_state(
    engine: *const KartPhysicsEngine,
    out: *mut KpVehicleState,
) -> i32 {
    ffi_guard_status(|| {
        if engine.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_get_state: NULL engine".to_string(),
            ));
        }
        if out.is_null() {
            return Err((KP_ERR_NULL, "kart_physics_get_state: NULL out".to_string()));
        }
        // Safety: pointers checked non-null; `engine` is a live handle.
        let engine = unsafe { &*engine };
        let state = KpVehicleState::to_pod(&engine.engine.state)
            .map_err(|error| (error.status_code(), error.into_message()))?;
        unsafe { write_state_for_abi(out, &state, engine.negotiated_abi) }
            .map_err(|error| (error.status_code(), error.into_message()))?;
        Ok(KP_OK)
    })
}

/// Replace the engine's current state (the prediction history is kept,
/// mirroring direct `state` assignment in the Python/Rust API). The state
/// is validated (`VehicleState.validate`) before it is committed.
///
/// # Safety
/// `engine` must be a live handle; `state` must be readable for that
/// engine's negotiated state layout.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_set_state(
    engine: *mut KartPhysicsEngine,
    state: *const KpVehicleState,
) -> i32 {
    ffi_guard_status(|| {
        if engine.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_set_state: NULL engine".to_string(),
            ));
        }
        if state.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_set_state: NULL state".to_string(),
            ));
        }
        // Safety: pointers checked non-null; `engine` is a live handle and
        // selects the exact readable state layout.
        let engine = unsafe { &mut *engine };
        let state = unsafe { read_state_for_abi(state, engine.negotiated_abi) }
            .map_err(|error| (KP_ERR_INVALID_STATE, error))?;
        let rich = state
            .to_rich()
            .map_err(|error| (KP_ERR_INVALID_STATE, error))?;
        rich.validate()
            .map_err(|error| (KP_ERR_INVALID_STATE, error))?;
        engine.engine.state = rich;
        Ok(KP_OK)
    })
}

/// Advance one game frame: the C mapping of
/// `KartRiderVehicleEngine::step_raw_frame(inputs.tick_ms, keys,
/// parameters, world)`. `parameters` must be the frame parameters (use
/// [`kart_physics_parameters_default_v2`] for the current standard set; ABI
/// v1 hosts use the frozen unsuffixed default writer). The frame loop derives
/// the per-substep substep widths itself. `world` may be NULL (the legal
/// `world=None` path — the engine surfaces its usual error unless the
/// scenario is world-free). When non-NULL, `world->query_ground` must be set;
/// every NULL optional pointer means capability absent.
///
/// Only bytes the engine's negotiated ABI owns are read: a v1-negotiated
/// engine copies exactly [`KP_VEHICLE_PARAMETERS_V1_BYTES`] from
/// `parameters` and [`KP_WORLD_VTABLE_V1_BYTES`] from `world`; a current
/// engine reads the full current structs.
///
/// # Safety
/// `engine` must be a live handle; `inputs` and `parameters` must point at
/// valid structs; `world`, when non-NULL, must point at a valid
/// `KpWorldVtable` (at least the negotiated prefix) whose callbacks are
/// safe to call for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn kart_physics_step(
    engine: *mut KartPhysicsEngine,
    inputs: *const KpInputs,
    parameters: *const KpVehicleParameters,
    world: *const KpWorldVtable,
) -> i32 {
    ffi_guard_status(|| {
        if engine.is_null() {
            return Err((KP_ERR_NULL, "kart_physics_step: NULL engine".to_string()));
        }
        if inputs.is_null() {
            return Err((KP_ERR_NULL, "kart_physics_step: NULL inputs".to_string()));
        }
        if parameters.is_null() {
            return Err((
                KP_ERR_NULL,
                "kart_physics_step: NULL parameters".to_string(),
            ));
        }
        // Safety: pointers checked non-null; `engine` is a live handle and
        // `world` validity is a caller contract. KpInputs did not grow.
        let (engine, inputs) = unsafe { (&mut *engine, &*inputs) };
        let keys = keys_from_mask(inputs.keys, engine.negotiated_abi)
            .map_err(|error| (KP_ERR_INVALID_INPUT, error))?;
        let parameters = unsafe { read_parameters_for_abi(parameters, engine.negotiated_abi) }
            .map_err(|error| (KP_ERR_INVALID_PARAMS, error))?;
        let params = parameters.to_rich();
        params
            .validate()
            .map_err(|error| (KP_ERR_INVALID_PARAMS, error))?;
        let adapter = if world.is_null() {
            None
        } else {
            // Safety: `world` is non-null; a v1-negotiated engine reads
            // only the v1 prefix the v1-layout host owns.
            let vtable = unsafe { read_world_vtable(world, engine.negotiated_abi) };
            if vtable.query_ground.is_none() {
                return Err((
                    KP_ERR_INVALID_INPUT,
                    "world vtable requires query_ground (NULL means no world; \
                     pass a NULL vtable pointer for world=None)"
                        .to_string(),
                ));
            }
            Some(CWorldAdapter::new(vtable))
        };
        let world_ref: Option<&dyn World> = adapter.as_ref().map(|adapter| adapter as &dyn World);
        engine
            .engine
            .step_raw_frame(inputs.tick_ms, &keys, &params, world_ref, None, None)
            .map_err(|error| {
                // The engine's capability error is generic ("reported
                // present but returned no result"); append the adapter's
                // recorded detail (JSON codec offset/reason, invalid
                // UTF-8, embedded NUL, clamp violation) when there is one.
                match adapter.as_ref().and_then(CWorldAdapter::take_query_error) {
                    Some(detail) => (
                        KP_ERR_STEP,
                        format!("{error} (world callback import detail: {detail})"),
                    ),
                    None => (KP_ERR_STEP, error),
                }
            })?;
        Ok(KP_OK)
    })
}

/// Copy the host vtable, reading only the bytes the negotiated ABI owns.
/// `negotiated_abi >= 2`: the full v2 struct (the host created the engine
/// through [`kart_physics_create_v2`], which pins the layouts equal).
/// `negotiated_abi == 1`: exactly the v1 prefix
/// ([`KP_WORLD_VTABLE_V1_BYTES`]) into a zero-initialized vtable, so the
/// v2-only tail slots read as NULL (capability absent) and a host compiled
/// against the shorter v1 layout is never read out of bounds.
///
/// # Safety
/// `world` must be readable for at least the negotiated prefix
/// (`KP_WORLD_VTABLE_V1_BYTES` for v1, the full v2 size for v2).
unsafe fn read_world_vtable(world: *const KpWorldVtable, negotiated_abi: u32) -> KpWorldVtable {
    if negotiated_abi >= 2 {
        // Safety: the caller guarantees the full v2 struct is readable.
        unsafe { *world }
    } else {
        // Safety: `KpWorldVtable` is all-POD (a raw pointer, a u8, and
        // nullable function pointers), so the all-zero byte pattern is a
        // valid value (every slot = capability absent).
        let mut vtable: KpWorldVtable = unsafe { std::mem::zeroed() };
        // Safety: the caller guarantees the v1 prefix is readable; the
        // destination owns the full v2 size, so the copy is in bounds on
        // both sides.
        unsafe {
            std::ptr::copy_nonoverlapping(
                world.cast::<u8>(),
                (&mut vtable as *mut KpWorldVtable).cast::<u8>(),
                KP_WORLD_VTABLE_V1_BYTES,
            );
        }
        vtable
    }
}

// ---------------------------------------------------------------------------
// Tests: POD round-trips, layout stability, error model, end-to-end FFI
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[repr(C, align(8))]
    struct ExactV1State([u8; KP_VEHICLE_STATE_V1_BYTES]);

    impl Default for ExactV1State {
        fn default() -> Self {
            Self([0; KP_VEHICLE_STATE_V1_BYTES])
        }
    }

    impl ExactV1State {
        fn as_const_ptr(&self) -> *const KpVehicleState {
            self.0.as_ptr().cast::<KpVehicleState>()
        }

        fn as_mut_ptr(&mut self) -> *mut KpVehicleState {
            self.0.as_mut_ptr().cast::<KpVehicleState>()
        }

        fn to_current(&self) -> KpVehicleState {
            unsafe { read_state_for_abi(self.as_const_ptr(), 1) }.expect("valid ABI v1 state")
        }
    }

    #[repr(C, align(8))]
    struct ExactV1Parameters([u8; KP_VEHICLE_PARAMETERS_V1_BYTES]);

    impl Default for ExactV1Parameters {
        fn default() -> Self {
            Self([0; KP_VEHICLE_PARAMETERS_V1_BYTES])
        }
    }

    impl ExactV1Parameters {
        fn as_const_ptr(&self) -> *const KpVehicleParameters {
            self.0.as_ptr().cast::<KpVehicleParameters>()
        }

        fn as_mut_ptr(&mut self) -> *mut KpVehicleParameters {
            self.0.as_mut_ptr().cast::<KpVehicleParameters>()
        }
    }

    #[repr(C, align(8))]
    struct ExactV3State([u8; KP_VEHICLE_STATE_V3_BYTES]);

    impl Default for ExactV3State {
        fn default() -> Self {
            let mut storage = Self([0; KP_VEHICLE_STATE_V3_BYTES]);
            unsafe {
                write_state_for_abi(storage.as_mut_ptr(), &KpVehicleState::default(), 3)
                    .expect("default state fits ABI v3");
            }
            storage
        }
    }

    impl ExactV3State {
        fn as_const_ptr(&self) -> *const KpVehicleState {
            self.0.as_ptr().cast::<KpVehicleState>()
        }

        fn as_mut_ptr(&mut self) -> *mut KpVehicleState {
            self.0.as_mut_ptr().cast::<KpVehicleState>()
        }

        fn to_current(&self) -> KpVehicleState {
            unsafe { read_state_for_abi(self.as_const_ptr(), 3) }.expect("valid ABI v3 state")
        }
    }

    #[repr(C, align(8))]
    struct ExactV3Parameters([u8; KP_VEHICLE_PARAMETERS_V3_BYTES]);

    impl ExactV3Parameters {
        fn from_current(parameters: &KpVehicleParameters) -> Self {
            let mut storage = Self([0; KP_VEHICLE_PARAMETERS_V3_BYTES]);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (parameters as *const KpVehicleParameters).cast::<u8>(),
                    storage.0.as_mut_ptr(),
                    KP_VEHICLE_PARAMETERS_V3_BYTES,
                );
            }
            storage
        }

        fn as_const_ptr(&self) -> *const KpVehicleParameters {
            self.0.as_ptr().cast::<KpVehicleParameters>()
        }
    }

    #[repr(C, align(8))]
    struct ExactV1World([u8; KP_WORLD_VTABLE_V1_BYTES]);

    impl ExactV1World {
        fn from_current(world: &KpWorldVtable) -> Self {
            let mut storage = Self([0; KP_WORLD_VTABLE_V1_BYTES]);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (world as *const KpWorldVtable).cast::<u8>(),
                    storage.0.as_mut_ptr(),
                    KP_WORLD_VTABLE_V1_BYTES,
                );
            }
            storage
        }

        fn as_const_ptr(&self) -> *const KpWorldVtable {
            self.0.as_ptr().cast::<KpWorldVtable>()
        }
    }

    fn rich_selector() -> BTreeMap<String, JsonValue> {
        BTreeMap::from([
            ("contact_active".to_string(), JsonValue::Bool(true)),
            ("rail_state".to_string(), JsonValue::Number(3.0)),
            (
                "path_id".to_string(),
                JsonValue::String("rail:path-9/\"kp\"\n한".to_string()),
            ),
            (
                "runtime".to_string(),
                JsonValue::Object(BTreeMap::from([(
                    "rail_dir".to_string(),
                    JsonValue::Array(vec![
                        JsonValue::Number(0.0),
                        JsonValue::Number(-0.0),
                        JsonValue::Number(100.00000149011612),
                        JsonValue::Number(1e-7),
                    ]),
                )])),
            ),
            ("min_velocity".to_string(), JsonValue::Null),
        ])
    }

    /// RAIL-domain state with every rail latch populated.
    fn rail_populated_state() -> VehicleState {
        let state = VehicleState {
            position: [1.5, -2.25, 3.125],
            velocity: [4.0, 5.0, -6.0],
            angular_velocity: [0.1, -0.2, 0.3],
            boost_state: 3,
            boost_remaining_ms: 120,
            drive_aa4: 12.5,
            drive_sign_aac: true,
            ground_timer_a80: 0.75,
            contact_compression: [0.1, 0.2, 0.3, 0.4],
            collision_e38: true,
            soft_flag_188: true,
            takeoff_separation_pending: true,
            surface_mode_latched: Some(2),
            surface_selector_elapsed_ms: 8.5,
            checkpoint_mode_family: "c72c".to_string(),
            checkpoint_target: [7.0, 8.0, 9.0],
            checkpoint_gate: true,
            checkpoint_gate_armed_tick_ms: Some(4242),
            checkpoint_marker_active_id: Some("kap_m1".to_string()),
            manual_restart_key_latched: true,
            manual_restart_count: 2,
            special_trigger_cooldown_ms: 30.0,
            authored_trigger_contact_id: Some("trigger:9".to_string()),
            rail_selector_runtime: Some(RailSelectorRuntime {
                alignment_primary_remaining_ms: Some(11.0),
                alignment_stable_remaining_ms: Some(12.0),
                path_context_id: Some(RailPathContextId::Name("asset-context-0".to_string())),
                state6_contact_remaining_ms: Some(100.00000149011612),
                state6_contact_candidate_latched: Some(1.0),
                state6_zero_boundary_release_delay_substeps: Some(2.0),
                state6_entry_force_pending_x: Some(3.0),
                state6_entry_force_pending_y: Some(-4.0),
                state6_entry_force_pending_z: Some(5.0),
                state6_entry_force_delay_substeps: Some(6.0),
                state6_entry_common_tail_delay_substeps: Some(7.0),
            }),
            rail_base_bf8: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            rail_constraint_c1c: Some([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            rail_orientation_dt_latch: Some(2.0),
            rail_capture_latch: Some(RailCaptureLatch {
                contact_active: true,
                rail_state: Some(3),
                selector: rich_selector(),
                authority: Some("world_query_rail_contact".to_string()),
            }),
            last_network_sequence: Some(77),
            network_packets_accepted: 5,
            network_packets_rejected: 1,
            network_packets_lost_estimate: 2,
            topology: TopologyState {
                domain: MotionDomain::Rail,
                previous_domain: Some(MotionDomain::Ground),
                transition_index: 4,
                domain_step_index: 9,
                rail_state_elapsed_ms: 22.0,
                entry_reason: "rail_capture_or_sustain".to_string(),
                constraint_owner_id: Some("rail:path-9".to_string()),
                rail_state: 3,
                ..Default::default()
            },
            time_ms: 1234,
            step_index: 567,
            scheduler_previous_tick_ms: Some(0xFFFF_FFF0),
            ..Default::default()
        };
        state.validate().unwrap();
        state
    }

    /// NET-CORRECTION state with the anchor + transaction latches populated.
    fn correction_populated_state() -> VehicleState {
        let state = VehicleState {
            moving_contact_anchor: Some(MovingContactAnchor {
                object_id: "roadobj:42".to_string(),
                local_position: [1.0, 2.0, 3.0],
                local_velocity: [4.0, 5.0, 6.0],
                relative_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                local_angular_velocity: [0.5, 0.6, 0.7],
                local_contact_normal: [0.0, 0.0, 1.0],
                local_plane_offset: 0.25,
                discarded_normal_velocity: Some(-1.5),
                object_point_velocity: Some([9.0, 8.0, 7.0]),
                kinematic_contract: Some("kinematic".to_string()),
            }),
            network_correction: Some(NetworkCorrectionState {
                mode: "blend".to_string(),
                duration_ms: 120.0,
                elapsed_ms: 40.0,
                sequence: Some(0xABCD),
                initial_state: RigidStateSnapshot {
                    position: [1.0, 2.0, 3.0],
                    velocity: [4.0, 5.0, 6.0],
                    angular_velocity: [0.1, 0.2, 0.3],
                    matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                },
                target_state: RigidStateSnapshot {
                    position: [7.0, 8.0, 9.0],
                    velocity: [10.0, 11.0, 12.0],
                    angular_velocity: [0.4, 0.5, 0.6],
                    matrix: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                },
                complete: false,
                policy_authority: Some("explicit_network_reconciliation_policy".to_string()),
                resume_topology: Some(TopologyState {
                    domain: MotionDomain::MovingContact,
                    return_domain: Some(MotionDomain::Ground),
                    transition_index: 8,
                    domain_step_index: 3,
                    entry_reason: "moving_reference_frame_acquired".to_string(),
                    constraint_owner_id: Some("roadobj:42".to_string()),
                    ..Default::default()
                }),
            }),
            rail_selector_runtime: Some(RailSelectorRuntime {
                path_context_id: Some(RailPathContextId::Index(32)),
                ..Default::default()
            }),
            topology: TopologyState {
                domain: MotionDomain::NetworkCorrection,
                previous_domain: Some(MotionDomain::MovingContact),
                return_domain: Some(MotionDomain::MovingContact),
                transition_index: 9,
                entry_reason: "network_correction_active".to_string(),
                constraint_owner_id: Some("roadobj:42".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        state.validate().unwrap();
        state
    }

    #[test]
    fn json_codec_round_trips_bit_exact() {
        let value = JsonValue::Object(rich_selector());
        let mut bytes = Vec::new();
        write_json_value(&value, &mut bytes).unwrap();
        let parsed = parse_json_value(&bytes).unwrap();
        assert_eq!(parsed, value);
        // -0.0 must survive as -0.0 (f64 `==` would not catch a sign flip).
        let mut negative_zero = Vec::new();
        write_json_value(&JsonValue::Number(-0.0), &mut negative_zero).unwrap();
        match parse_json_value(&negative_zero).unwrap() {
            JsonValue::Number(number) => assert_eq!(number.to_bits(), (-0.0f64).to_bits()),
            other => panic!("expected number, got {other:?}"),
        }
        // Shortest-repr round-trip for awkward values.
        for sample in [1e-7, 100.00000149011612, 0.1, 1e300, 5e-324] {
            let mut encoded = Vec::new();
            write_json_value(&JsonValue::Number(sample), &mut encoded).unwrap();
            match parse_json_value(&encoded).unwrap() {
                JsonValue::Number(number) => assert_eq!(number.to_bits(), sample.to_bits()),
                other => panic!("expected number, got {other:?}"),
            }
        }
        // Non-finite values are rejected.
        assert!(write_json_value(&JsonValue::Number(f64::NAN), &mut Vec::new()).is_err());
        assert!(parse_json_value(b"1e999").is_err());
        // Garbage is rejected.
        assert!(parse_json_value(b"{\"a\": }").is_err());
        assert!(parse_json_value(b"[1,2").is_err());
        assert!(parse_json_value(b"null null").is_err());
        assert!(parse_json_value(b"\"\\ud800x\"").is_err());
    }

    #[test]
    fn string64_limits() {
        let exactly = "a".repeat(64);
        assert_eq!(KpString64::set(&exactly).unwrap().len, 64);
        assert!(KpString64::set(&"b".repeat(65)).is_err());
        let pod = KpString64::set("rail:path-9").unwrap();
        assert_eq!(pod.get().unwrap(), "rail:path-9");
        let mut invalid = KpString64::default();
        invalid.data[0] = 0xFF;
        invalid.len = 1;
        assert!(invalid.get().is_err());
        assert_eq!(invalid.get_lossy(), "\u{FFFD}");
    }

    #[test]
    fn default_state_pod_round_trip() {
        let state = VehicleState::default();
        let pod = KpVehicleState::to_pod(&state).unwrap();
        assert_eq!(pod.to_rich().unwrap(), state);
        // POD -> rich -> POD is also identity.
        assert_eq!(
            KpVehicleState::to_pod(&pod.to_rich().unwrap()).unwrap(),
            pod
        );
        pod.to_rich().unwrap().validate().unwrap();
    }

    #[test]
    fn rail_populated_state_pod_round_trip() {
        let state = rail_populated_state();
        let pod = KpVehicleState::to_pod(&state).unwrap();
        let rich = pod.to_rich().unwrap();
        assert_eq!(rich, state);
        rich.validate().unwrap();
        assert_eq!(KpVehicleState::to_pod(&rich).unwrap(), pod);
    }

    #[test]
    fn correction_populated_state_pod_round_trip() {
        let state = correction_populated_state();
        let pod = KpVehicleState::to_pod(&state).unwrap();
        let rich = pod.to_rich().unwrap();
        assert_eq!(rich, state);
        rich.validate().unwrap();
        assert_eq!(KpVehicleState::to_pod(&rich).unwrap(), pod);
    }

    #[test]
    fn parameters_pod_round_trip() {
        let parameters = VehicleParameters {
            topology_selector_delta_ms: Some(16.0),
            initial_rail_capture_commit: Some(true),
            ..Default::default()
        };
        let pod = KpVehicleParameters::to_pod(&parameters);
        assert_eq!(pod.to_rich(), parameters);
        let defaults = VehicleParameters::default();
        assert_eq!(KpVehicleParameters::to_pod(&defaults).to_rich(), defaults);
    }

    #[test]
    fn oversized_strings_are_rejected_not_truncated() {
        let state = VehicleState {
            checkpoint_marker_active_id: Some("x".repeat(65)),
            ..Default::default()
        };
        assert!(KpVehicleState::to_pod(&state).is_err());
    }

    #[test]
    fn malformed_inventory_shape_is_rejected_not_panicked() {
        let state = VehicleState {
            booster_inventory_slots: vec![InventorySlot {
                item_id: 6,
                quantity: 1,
            }],
            ..Default::default()
        };
        let error = KpVehicleState::to_pod(&state).unwrap_err();
        assert!(matches!(error, PodExportError::InvalidState(_)));
        assert_eq!(error.status_code(), KP_ERR_INVALID_STATE);
    }

    #[test]
    fn keys_mask_conversion() {
        assert_eq!(
            keys_from_mask(0, KP_ABI_VERSION).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            keys_from_mask(KP_KEY_UP | KP_KEY_LEFT, KP_ABI_VERSION).unwrap(),
            ["UP", "LEFT"]
        );
        assert_eq!(
            keys_from_mask(KP_KEY_MASK_ALL, KP_ABI_VERSION).unwrap(),
            ["UP", "DOWN", "LEFT", "RIGHT", "R", "CTRL", "ITEM_USE"]
        );
        assert!(keys_from_mask(KP_KEY_CTRL, 2).is_err());
        assert!(keys_from_mask(KP_KEY_ITEM_USE, 3).is_err());
        assert!(keys_from_mask(1 << 7, KP_ABI_VERSION).is_err());
    }

    /// Frozen layout notes (64-bit targets): every mirror is naturally
    /// aligned (f64/i64/pointer dominate); the sizes below are the frozen
    /// ABI and must only change together with `KP_ABI_VERSION`. ABI v5
    /// tail-appends GoItemBanana state; ABI v4 tail-appends current-family
    /// N2O state/parameters; ABI v3
    /// tail-appended drift state/parameters (old offsets stay pinned).
    /// ABI v2
    /// (2026-07-29): `KP_JSON_CAPACITY` 4096 -> 65536 (real factory_I02
    /// `query_rail_contact` selectors measure 40354 bytes) grew
    /// `KpJsonBlob` and every struct embedding it, and `KpWorldVtable`
    /// gained 8 tail slots (80 -> 144 bytes; v1 slot offsets unchanged).
    #[test]
    fn layout_is_stable() {
        assert_eq!(size_of::<KpString64>(), 65);
        assert_eq!(align_of::<KpString64>(), 1);
        assert_eq!(size_of::<KpJsonBlob>(), 65540);
        assert_eq!(align_of::<KpJsonBlob>(), 4);
        assert_eq!(size_of::<KpJsonScalar>(), 88);
        assert_eq!(align_of::<KpJsonScalar>(), 8);
        assert_eq!(size_of::<KpInputs>(), 8);
        assert_eq!(align_of::<KpInputs>(), 4);
        assert_eq!(size_of::<KpTopologyState>(), 176);
        assert_eq!(size_of::<KpRailSelectorRuntime>(), 240);
        assert_eq!(size_of::<KpRailCaptureLatch>(), 65624);
        assert_eq!(size_of::<KpMovingContactAnchor>(), 368);
        assert_eq!(size_of::<KpRigidStateSnapshot>(), 144);
        assert_eq!(size_of::<KpNetworkCorrectionState>(), 632);
        assert_eq!(size_of::<KpVehicleState>(), 68136);
        assert_eq!(align_of::<KpVehicleState>(), 8);
        assert_eq!(KP_VEHICLE_STATE_V1_BYTES, 6448);
        assert_eq!(KP_VEHICLE_STATE_V2_BYTES, 67888);
        assert_eq!(KP_VEHICLE_STATE_V3_BYTES, 67984);
        assert_eq!(KP_VEHICLE_PARAMETERS_V1_BYTES, 536);
        assert_eq!(KP_VEHICLE_PARAMETERS_V3_BYTES, 592);
        assert_eq!(
            KP_STATE_CURRENT_TAIL_OFFSET,
            KP_STATE_RAIL_JSON_OFFSET + size_of::<KpJsonBlob>()
        );
        assert_eq!(
            KP_STATE_V1_TAIL_OFFSET,
            KP_STATE_RAIL_JSON_OFFSET + KP_JSON_BLOB_V1_BYTES
        );
        assert_eq!(
            KP_VEHICLE_STATE_V1_BYTES,
            KP_VEHICLE_STATE_V2_BYTES - KP_JSON_CAPACITY_DELTA_V2
        );
        assert_eq!(size_of::<KpBoostParameters>(), 120);
        assert_eq!(size_of::<KpVehicleParameters>(), 608);
        assert_eq!(std::mem::offset_of!(KpVehicleState, selector_b08), 67888);
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, booster_inventory_item_ids),
            KP_VEHICLE_STATE_V3_BYTES
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, booster_inventory_quantities),
            68000
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, item_use_key_latched),
            68016
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, boost_gauge_committed_b10),
            68024
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, boost_gauge_progress_b20),
            68032
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, boost_gauge_pending_b2c),
            68040
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, boost_gauge_history_b38),
            68048
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, banana_effect_remaining_ms),
            68056
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleState, banana_effect_owner_id),
            68064
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleParameters, drift_window_duration_734),
            536
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleParameters, item6_boost_duration_ms),
            KP_VEHICLE_PARAMETERS_V3_BYTES
        );
        assert_eq!(
            std::mem::offset_of!(KpVehicleParameters, boost_gauge_raw_760),
            600
        );
        assert_eq!(size_of::<KpWorldGroundResult>(), 464);
        assert_eq!(size_of::<KpWorldObbOverlap>(), 520);
        assert_eq!(size_of::<KpTakeoffLandingContact>(), 168);
        assert_eq!(size_of::<KpCheckpointTargetCandidate>(), 184);
        assert_eq!(size_of::<KpCheckpointRespawnCandidate>(), 65768);
        assert_eq!(size_of::<KpRailEntryBoundaryMatch>(), 424);
        assert_eq!(align_of::<KpRailEntryBoundaryMatch>(), 8);
        assert_eq!(size_of::<KpWorldVtable>(), 144);
        assert_eq!(align_of::<KpWorldVtable>(), 8);
        // The v1 vtable prefix is unchanged: the last v1 slot still sits at
        // byte offset 72 (user_data 8 + query_at_step_start 8 + 7 slots).
        assert_eq!(
            std::mem::offset_of!(KpWorldVtable, query_checkpoint_respawn),
            72
        );
        // The v2 capability tail starts right after it; the v1 prefix size
        // constant is pinned against the real layout (N1: a v1-negotiated
        // engine copies exactly this many bytes out of the host vtable).
        assert_eq!(std::mem::offset_of!(KpWorldVtable, sweep_point), 80);
        assert_eq!(
            KP_WORLD_VTABLE_V1_BYTES,
            std::mem::offset_of!(KpWorldVtable, sweep_point)
        );
        assert_eq!(
            std::mem::offset_of!(KpWorldVtable, query_distinct_water_volume),
            136
        );
        // Domain ids stay in sync with `MotionDomain` declaration order.
        for (domain, id) in [
            (MotionDomain::Ground, KP_DOMAIN_GND),
            (MotionDomain::GroundSlip, KP_DOMAIN_GND_SLIP),
            (MotionDomain::Contact, KP_DOMAIN_CONTACT),
            (MotionDomain::AirEntry, KP_DOMAIN_AIR_ENTRY),
            (MotionDomain::Air, KP_DOMAIN_AIR),
            (MotionDomain::Land, KP_DOMAIN_LAND),
            (MotionDomain::Rail, KP_DOMAIN_RAIL),
            (MotionDomain::MovingContact, KP_DOMAIN_MOVING_CONTACT),
            (MotionDomain::Reset, KP_DOMAIN_RESET),
            (MotionDomain::Water, KP_DOMAIN_WATER),
            (MotionDomain::NetworkCorrection, KP_DOMAIN_NET_CORRECTION),
        ] {
            assert_eq!(domain_to_u8(domain), id);
            assert_eq!(domain_from_u8(id).unwrap(), domain);
        }
    }

    // -- End-to-end FFI drive (flat ground plane through the vtable) --

    unsafe extern "C" fn flat_query_ground(
        _user_data: *mut c_void,
        _position: *const f64,
        _up: *const f64,
        _min_dot: f64,
        out: *mut KpWorldGroundResult,
    ) {
        // The canonical parity mock plane row (z=0, up=+Z).
        let row = KpWorldGroundResult {
            track_id: KpString64::set("mock").unwrap(),
            node: KpString64::set("mock").unwrap(),
            tri_index: 7,
            surface_id: 3,
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        unsafe { *out = row };
    }

    unsafe extern "C" fn empty_sweep(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        _start_matrix: *const f64,
        _end_matrix: *const f64,
        _half_extents: *const f64,
        _out: *mut KpWorldObbOverlap,
        _out_capacity: usize,
    ) -> usize {
        0
    }

    fn flat_world_vtable() -> KpWorldVtable {
        KpWorldVtable {
            user_data: std::ptr::null_mut(),
            query_at_step_start: 0,
            query_ground: Some(flat_query_ground),
            sweep_obb_rotating_all: Some(empty_sweep),
            sweep_obb_rotating: None,
            overlap_obb_all: None,
            query_takeoff_contact: None,
            query_landing_contact: None,
            query_checkpoint_target_candidate: None,
            query_checkpoint_respawn: None,
            sweep_point: None,
            query_rail_contact: None,
            query_rail_entry_boundary: None,
            rail_query_contexts: None,
            rail_entry_contexts: None,
            query_moving_contact: None,
            query_network_correction: None,
            query_distinct_water_volume: None,
        }
    }

    #[test]
    fn ffi_end_to_end_flat_drive() {
        unsafe {
            let engine =
                kart_physics_create_v2(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION);
            assert!(!engine.is_null());
            let mut parameters = KpVehicleParameters::default();
            assert_eq!(
                kart_physics_parameters_default_v2(&mut parameters, KP_ABI_VERSION),
                KP_OK
            );
            let world = flat_world_vtable();
            let mut inputs = KpInputs {
                tick_ms: 1000,
                keys: 0,
            };
            // Clock-init frame.
            assert_eq!(
                kart_physics_step(engine, &inputs, &parameters, &world),
                KP_OK
            );
            for _ in 0..20 {
                inputs.tick_ms = inputs.tick_ms.wrapping_add(16);
                inputs.keys = KP_KEY_UP;
                assert_eq!(
                    kart_physics_step(engine, &inputs, &parameters, &world),
                    KP_OK
                );
            }
            let mut state = KpVehicleState::default();
            assert_eq!(kart_physics_get_state(engine, &mut state), KP_OK);
            assert_eq!(state.time_ms, 20 * 16);
            assert_eq!(state.step_index, 20 * 8);
            assert_eq!(state.topology.domain, KP_DOMAIN_GND);
            assert_eq!(state.has_scheduler_previous_tick_ms, 1);
            assert_eq!(state.scheduler_previous_tick_ms, 1000 + 20 * 16);
            // The kart must have moved forward (away from the origin) and
            // every component must be finite.
            assert!(state.position.iter().all(|value| value.is_finite()));
            assert!(state.position.iter().any(|value| *value != 0.0));
            kart_physics_destroy(engine);
        }
    }

    #[test]
    fn ffi_v1_shim_uses_exact_state_and_parameter_buffers() {
        unsafe {
            // These Boxes own exactly the historical byte counts. Any full
            // current-struct read/write here is an allocation-boundary OOB
            // (and is caught by Miri/sanitizers instead of being hidden by a
            // conveniently oversized current struct).
            let mut initial_state = Box::new(ExactV1State::default());
            assert_eq!(
                kart_physics_state_default(initial_state.as_mut_ptr()),
                KP_OK
            );
            let decoded_default = initial_state.to_current().to_rich().unwrap();
            assert_eq!(decoded_default, VehicleState::default());

            let engine = kart_physics_create(initial_state.as_const_ptr(), KP_NUMERIC_MODE_FLOAT32);
            assert!(!engine.is_null());

            let mut parameters = Box::new(ExactV1Parameters::default());
            assert_eq!(
                kart_physics_parameters_default(parameters.as_mut_ptr()),
                KP_OK
            );
            let decoded_parameters = read_parameters_for_abi(parameters.as_const_ptr(), 1).unwrap();
            assert_eq!(decoded_parameters.to_rich(), VehicleParameters::default());

            let world = ExactV1World::from_current(&flat_world_vtable());
            let mut inputs = KpInputs {
                tick_ms: 1000,
                keys: 0,
            };
            assert_eq!(
                kart_physics_step(
                    engine,
                    &inputs,
                    parameters.as_const_ptr(),
                    world.as_const_ptr(),
                ),
                KP_OK
            );
            inputs.tick_ms = 1016;
            inputs.keys = KP_KEY_UP;
            assert_eq!(
                kart_physics_step(
                    engine,
                    &inputs,
                    parameters.as_const_ptr(),
                    world.as_const_ptr(),
                ),
                KP_OK
            );

            let mut output_state = Box::new(ExactV1State::default());
            assert_eq!(
                kart_physics_get_state(engine, output_state.as_mut_ptr()),
                KP_OK
            );
            let decoded_output = output_state.to_current();
            assert_eq!(decoded_output.time_ms, 16);
            assert_eq!(decoded_output.step_index, 8);
            assert_eq!(decoded_output.topology.domain, KP_DOMAIN_GND);

            assert_eq!(
                kart_physics_set_state(engine, output_state.as_const_ptr()),
                KP_OK
            );
            kart_physics_destroy(engine);
        }
    }

    #[test]
    fn ffi_v3_translation_uses_exact_state_and_parameter_buffers() {
        std::thread::Builder::new()
            .name("ffi-v3-exact-buffers".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(exercise_ffi_v3_exact_buffers)
            .unwrap()
            .join()
            .unwrap();
    }

    fn exercise_ffi_v3_exact_buffers() {
        unsafe {
            // These allocations own exactly the frozen v3 prefixes. The v4
            // translator must neither read nor write the appended N2O tail.
            let initial_state = Box::new(ExactV3State::default());
            let decoded_default = initial_state.to_current().to_rich().unwrap();
            assert_eq!(decoded_default, VehicleState::default());

            let engine =
                create_engine_handle(initial_state.as_const_ptr(), KP_NUMERIC_MODE_FLOAT32, 3)
                    .unwrap();

            let current_parameters = KpVehicleParameters {
                item6_boost_duration_ms: 9999,
                boost_gauge_raw_760: 42.0,
                ..Default::default()
            };
            let parameters = Box::new(ExactV3Parameters::from_current(&current_parameters));
            let decoded_parameters = read_parameters_for_abi(parameters.as_const_ptr(), 3).unwrap();
            let default_parameters = VehicleParameters::default();
            assert_eq!(
                decoded_parameters.item6_boost_duration_ms,
                default_parameters.item6_boost_duration_ms
            );
            assert_eq!(
                decoded_parameters.boost_gauge_raw_760,
                default_parameters.boost_gauge_raw_760
            );

            let world = flat_world_vtable();
            let mut inputs = KpInputs {
                tick_ms: 1000,
                keys: 0,
            };
            assert_eq!(
                kart_physics_step(engine, &inputs, parameters.as_const_ptr(), &world),
                KP_OK
            );
            inputs.tick_ms = 1016;
            inputs.keys = KP_KEY_UP;
            assert_eq!(
                kart_physics_step(engine, &inputs, parameters.as_const_ptr(), &world),
                KP_OK
            );

            let mut output_state = Box::new(ExactV3State::default());
            assert_eq!(
                kart_physics_get_state(engine, output_state.as_mut_ptr()),
                KP_OK
            );
            let decoded_output = output_state.to_current().to_rich().unwrap();
            assert_eq!(decoded_output.time_ms, 16);
            assert_eq!(decoded_output.step_index, 8);
            assert_eq!(
                decoded_output.booster_inventory_slots,
                VehicleState::default().booster_inventory_slots
            );
            assert!(!decoded_output.item_use_key_latched);
            assert_eq!(decoded_output.boost_gauge_committed_b10, 0.0);
            assert_eq!(decoded_output.boost_gauge_progress_b20, 0.0);
            assert_eq!(decoded_output.boost_gauge_pending_b2c, 0.0);
            assert_eq!(decoded_output.boost_gauge_history_b38, 0.0);

            assert_eq!(
                kart_physics_set_state(engine, output_state.as_const_ptr()),
                KP_OK
            );
            kart_physics_destroy(engine);
        }
    }

    #[test]
    fn ffi_error_paths() {
        unsafe {
            // NULL arguments.
            assert_eq!(
                kart_physics_step(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null()
                ),
                KP_ERR_NULL
            );
            assert_eq!(
                kart_physics_get_state(std::ptr::null(), std::ptr::null_mut()),
                KP_ERR_NULL
            );
            // Unknown numeric mode -> NULL handle + message.
            let engine = kart_physics_create(std::ptr::null(), 99);
            assert!(engine.is_null());
            let mut buffer = [0 as c_char; 128];
            let needed = kart_physics_last_error_message(buffer.as_mut_ptr(), buffer.len());
            assert!(needed > 0);
            let message = std::ffi::CStr::from_ptr(buffer.as_ptr()).to_str().unwrap();
            assert!(message.contains("numeric mode"));
            // Vtable without query_ground -> KP_ERR_INVALID_INPUT.
            let engine =
                kart_physics_create_v2(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION);
            assert!(!engine.is_null());
            let parameters = KpVehicleParameters::default();
            let inputs = KpInputs::default();
            let mut world = flat_world_vtable();
            world.query_ground = None;
            assert_eq!(
                kart_physics_step(engine, &inputs, &parameters, &world),
                KP_ERR_INVALID_INPUT
            );
            // Unknown key bits -> KP_ERR_INVALID_INPUT.
            let bad_inputs = KpInputs {
                tick_ms: 5,
                keys: 1 << 9,
            };
            assert_eq!(
                kart_physics_step(engine, &bad_inputs, &parameters, &world),
                KP_ERR_INVALID_INPUT
            );
            // world=None: the first frame is the clock-init frame, which
            // succeeds without touching the world; the second frame hits
            // the engine's own world-required error.
            assert_eq!(
                kart_physics_step(engine, &inputs, &parameters, std::ptr::null()),
                KP_OK
            );
            let inputs2 = KpInputs {
                tick_ms: 16,
                keys: 0,
            };
            assert_eq!(
                kart_physics_step(engine, &inputs2, &parameters, std::ptr::null()),
                KP_ERR_STEP
            );
            // Invalid state import (bad mode family) -> KP_ERR_INVALID_STATE.
            let state = KpVehicleState {
                checkpoint_mode_family: KpString64::set("bogus").unwrap(),
                ..Default::default()
            };
            assert_eq!(kart_physics_set_state(engine, &state), KP_ERR_INVALID_STATE);
            kart_physics_destroy(engine);
            kart_physics_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn vtable_adapter_capability_absence_semantics() {
        // Only query_ground set: every optional capability must report
        // absent (the v1 all-NULL tail included).
        let adapter = CWorldAdapter::new(KpWorldVtable {
            user_data: std::ptr::null_mut(),
            query_at_step_start: 0,
            query_ground: Some(flat_query_ground),
            sweep_obb_rotating_all: None,
            sweep_obb_rotating: None,
            overlap_obb_all: None,
            query_takeoff_contact: None,
            query_landing_contact: None,
            query_checkpoint_target_candidate: None,
            query_checkpoint_respawn: None,
            sweep_point: None,
            query_rail_contact: None,
            query_rail_entry_boundary: None,
            rail_query_contexts: None,
            rail_entry_contexts: None,
            query_moving_contact: None,
            query_network_correction: None,
            query_distinct_water_volume: None,
        });
        assert_eq!(adapter.capabilities(), WorldCapabilities::default());
        // The optional methods return None (capability absent), matching
        // the `hasattr`-miss semantics.
        assert!(adapter
            .overlap_obb_all([0.0; 3], [0.0; 9], [0.0; 3])
            .is_none());
        assert!(adapter
            .query_checkpoint_target_candidate([0.0; 3], 2.0)
            .is_none());
        assert!(adapter.sweep_point([0.0; 3], [1.0; 3]).is_none());
        assert!(adapter.query_rail_contact([0.0; 3], 0, None).is_none());
        assert!(adapter
            .query_rail_entry_boundary([0.0; 3], None, 0.05)
            .is_none());
        assert!(adapter.rail_query_contexts().is_none());
        assert!(adapter.rail_entry_contexts().is_none());
        assert!(adapter
            .query_moving_contact([0.0; 3], 0, None, 1, 1.5)
            .is_none());
        assert!(adapter.query_network_correction([0.0; 3]).is_none());
        assert!(adapter.query_distinct_water_volume([0.0; 3]).is_none());
        // query_ground still works and converts the POD row.
        let result = adapter.query_ground([0.0; 3], [0.0, 0.0, 1.0], 0.65);
        assert_eq!(result.node, "mock");
        assert_eq!(result.tri_index, 7);
        assert_eq!(result.normal, [0.0, 0.0, 1.0]);
        assert_eq!(result.triangle, None);
    }

    // -- Typed rich -> POD export errors (handoff 2026-07-29 §3.2) --

    #[test]
    fn typed_pod_export_errors() {
        // Over-capacity string -> StringTooLong (KP_ERR_STRING_TOO_LONG).
        let state = VehicleState {
            checkpoint_marker_active_id: Some("x".repeat(65)),
            ..Default::default()
        };
        let error = KpVehicleState::to_pod(&state).unwrap_err();
        assert!(matches!(error, PodExportError::StringTooLong(_)));
        assert_eq!(error.status_code(), KP_ERR_STRING_TOO_LONG);
        // Over-capacity JSON selector -> Json (KP_ERR_JSON), without any
        // message-substring sniffing.
        let state = VehicleState {
            rail_capture_latch: Some(RailCaptureLatch {
                contact_active: true,
                rail_state: Some(1),
                selector: BTreeMap::from([(
                    "pad".to_string(),
                    JsonValue::String("y".repeat(KP_JSON_CAPACITY as usize + 16)),
                )]),
                authority: None,
            }),
            ..Default::default()
        };
        let error = KpVehicleState::to_pod(&state).unwrap_err();
        assert!(matches!(error, PodExportError::Json(_)));
        assert_eq!(error.status_code(), KP_ERR_JSON);
    }

    // -- Open-mapping capability adapters (present direction) --

    fn json_number(value: f64) -> JsonValue {
        JsonValue::Number(value)
    }

    fn write_json_out(out: *mut KpJsonBlob, value: &JsonValue) {
        let blob = KpJsonBlob::set_json(value).expect("test JSON fits");
        unsafe { *out = blob };
    }

    fn smoke_sweep_point_mapping() -> JsonValue {
        JsonValue::Object(BTreeMap::from([
            ("hit".to_string(), JsonValue::Bool(true)),
            ("road_rail".to_string(), JsonValue::String(String::new())),
        ]))
    }

    unsafe extern "C" fn ok_sweep_point(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        write_json_out(out, &smoke_sweep_point_mapping());
        1
    }

    unsafe extern "C" fn garbage_sweep_point(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        let mut blob = KpJsonBlob {
            len: 4,
            ..Default::default()
        };
        blob.data[..4].copy_from_slice(b"{nil");
        unsafe { *out = blob };
        1
    }

    unsafe extern "C" fn non_object_sweep_point(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        write_json_out(out, &JsonValue::Array(vec![json_number(1.0)]));
        1
    }

    unsafe extern "C" fn absent_result_sweep_point(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        _out: *mut KpJsonBlob,
    ) -> i32 {
        0
    }

    fn smoke_rail_selector(rail_state: i64) -> JsonValue {
        let vec3 = |x: f64, y: f64, z: f64| {
            JsonValue::Array(vec![json_number(x), json_number(y), json_number(z)])
        };
        let runtime = JsonValue::Object(BTreeMap::from([
            ("rail_dir".to_string(), vec3(1.0, 0.0, 0.0)),
            ("accel_be0".to_string(), json_number(4608.0)),
            ("obf_6e8".to_string(), json_number(1.0)),
            ("grip_720".to_string(), json_number(1.0)),
            ("min_velocity".to_string(), json_number(5.0)),
            ("max_velocity".to_string(), json_number(12.0)),
        ]));
        let point = |x: f64| {
            JsonValue::Object(BTreeMap::from([
                ("position".to_string(), vec3(x, 0.0, 0.0)),
                ("secondary".to_string(), vec3(1.0, 0.0, 0.0)),
                ("normal".to_string(), vec3(0.0, 0.0, 1.0)),
            ]))
        };
        let context = JsonValue::Object(BTreeMap::from([
            ("context_id".to_string(), json_number(0.0)),
            (
                "points".to_string(),
                JsonValue::Array(vec![point(0.0), point(1.4)]),
            ),
            (
                "road".to_string(),
                JsonValue::String("rail-road".to_string()),
            ),
            ("total_length".to_string(), json_number(1.4)),
        ]));
        let mut selector = BTreeMap::from([
            ("contact_active".to_string(), JsonValue::Bool(true)),
            ("rail_state".to_string(), json_number(rail_state as f64)),
            ("path_id".to_string(), json_number(7.0)),
            ("runtime".to_string(), runtime),
            (
                "registered_query_contexts".to_string(),
                JsonValue::Array(vec![context]),
            ),
        ]);
        if rail_state == 1 {
            selector.insert("contact_exists".to_string(), JsonValue::Bool(true));
            selector.insert(
                "contact_identifier_nonempty".to_string(),
                JsonValue::Bool(true),
            );
            selector.insert("alignment_base_ms".to_string(), json_number(1.0));
        }
        JsonValue::Object(selector)
    }

    unsafe extern "C" fn recording_rail_contact(
        _user_data: *mut c_void,
        position: *const f64,
        previous_rail_state: i64,
        owner_id: *const c_char,
        out: *mut KpJsonBlob,
    ) -> i32 {
        // The functional schedule from the recorded Python trace: negative
        // (None) once the kart passes the path end; mirror the current rail
        // state for sustain (the world must not publish transitions);
        // state-1 selector with the alignment gate inputs otherwise.
        let x = unsafe { *position };
        if x >= 1.35 {
            return 0;
        }
        let _ = owner_id; // the engine passes constraint_owner_id on sustain
        let rail_state = if previous_rail_state >= 2 {
            previous_rail_state
        } else {
            1
        };
        write_json_out(out, &smoke_rail_selector(rail_state));
        1
    }

    unsafe extern "C" fn negative_rail_contact(
        _user_data: *mut c_void,
        _position: *const f64,
        _previous_rail_state: i64,
        _owner_id: *const c_char,
        _out: *mut KpJsonBlob,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn no_entry_boundary_match(
        _user_data: *mut c_void,
        _position: *const f64,
        _road: *const c_char,
        _plane_tolerance: f64,
        _out: *mut KpRailEntryBoundaryMatch,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn entry_boundary_match(
        _user_data: *mut c_void,
        _position: *const f64,
        road: *const c_char,
        _plane_tolerance: f64,
        out: *mut KpRailEntryBoundaryMatch,
    ) -> i32 {
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(road) }.to_str().unwrap(),
            "rail-road"
        );
        let scalar = |tag: u8, number: f64, text: &str| KpJsonScalar {
            tag,
            bool_value: 0,
            number_value: number,
            string_value: KpString64::set(text).unwrap(),
        };
        unsafe {
            *out = KpRailEntryBoundaryMatch {
                context_id: scalar(KP_JSON_SCALAR_NUMBER, 3.0, ""),
                context_index: 3,
                road: KpString64::set("rail-road").unwrap(),
                element: scalar(KP_JSON_SCALAR_NUMBER, 11.0, ""),
                marker: scalar(KP_JSON_SCALAR_STRING, 0.0, "marker-9"),
                signed_plane_distance: -0.0125,
                plane_distance: 0.0125,
                projected_coordinates: [0.5, -0.25],
                path_start: [1.0, 2.0, 3.0],
                tangent: [1.0, 0.0, 0.0],
            }
        };
        1
    }

    unsafe extern "C" fn two_contexts(
        _user_data: *mut c_void,
        out: *mut KpJsonBlob,
        out_capacity: usize,
    ) -> usize {
        let contexts = [smoke_rail_selector(2), smoke_rail_selector(3)];
        if out_capacity >= contexts.len() {
            for (index, context) in contexts.iter().enumerate() {
                write_json_out(unsafe { out.add(index) }, context);
            }
        }
        contexts.len()
    }

    unsafe extern "C" fn ten_contexts(
        _user_data: *mut c_void,
        out: *mut KpJsonBlob,
        out_capacity: usize,
    ) -> usize {
        if out_capacity >= 10 {
            for index in 0..10 {
                write_json_out(unsafe { out.add(index) }, &smoke_rail_selector(2));
            }
        }
        10
    }

    unsafe extern "C" fn garbage_count_contexts(
        _user_data: *mut c_void,
        _out: *mut KpJsonBlob,
        _out_capacity: usize,
    ) -> usize {
        usize::MAX
    }

    unsafe extern "C" fn over_byte_ceiling_contexts(
        _user_data: *mut c_void,
        _out: *mut KpJsonBlob,
        _out_capacity: usize,
    ) -> usize {
        // 1<<20 elements of 65540 bytes = 68+ GB: under the count ceiling
        // but over the byte ceiling; must be rejected WITHOUT any
        // allocation attempt.
        KP_LIST_MAX_COUNT
    }

    unsafe extern "C" fn empty_contexts(
        _user_data: *mut c_void,
        _out: *mut KpJsonBlob,
        _out_capacity: usize,
    ) -> usize {
        0
    }

    unsafe extern "C" fn moving_contact_record(
        _user_data: *mut c_void,
        _position: *const f64,
        time_ms: i64,
        object_id: *const c_char,
        delta_ms: i64,
        max_distance: f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        assert_eq!(time_ms, 1234);
        assert_eq!(delta_ms, 1);
        assert_eq!(max_distance, 1.5);
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(object_id) }
                .to_str()
                .unwrap(),
            "roadobj:42"
        );
        write_json_out(
            out,
            &JsonValue::Object(BTreeMap::from([(
                "contact_active".to_string(),
                JsonValue::Bool(true),
            )])),
        );
        1
    }

    unsafe extern "C" fn network_correction_record(
        _user_data: *mut c_void,
        _position: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        write_json_out(
            out,
            &JsonValue::Object(BTreeMap::from([(
                "mode".to_string(),
                JsonValue::String("snap".to_string()),
            )])),
        );
        1
    }

    unsafe extern "C" fn no_network_correction(
        _user_data: *mut c_void,
        _position: *const f64,
        _out: *mut KpJsonBlob,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn water_volume_selector(
        _user_data: *mut c_void,
        _position: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        write_json_out(out, &JsonValue::Bool(true));
        1
    }

    unsafe extern "C" fn garbage_count_sweep(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        _start_matrix: *const f64,
        _end_matrix: *const f64,
        _half_extents: *const f64,
        _out: *mut KpWorldObbOverlap,
        _out_capacity: usize,
    ) -> usize {
        usize::MAX
    }

    unsafe extern "C" fn garbage_count_overlap(
        _user_data: *mut c_void,
        _center: *const f64,
        _matrix: *const f64,
        _half_extents: *const f64,
        _out: *mut KpWorldObbOverlap,
        _out_capacity: usize,
    ) -> usize {
        usize::MAX
    }

    fn open_world_vtable() -> KpWorldVtable {
        KpWorldVtable {
            sweep_point: Some(ok_sweep_point),
            query_rail_contact: Some(recording_rail_contact),
            query_rail_entry_boundary: Some(no_entry_boundary_match),
            rail_query_contexts: Some(two_contexts),
            rail_entry_contexts: Some(empty_contexts),
            query_moving_contact: Some(moving_contact_record),
            query_network_correction: Some(network_correction_record),
            query_distinct_water_volume: Some(water_volume_selector),
            ..flat_world_vtable()
        }
    }

    #[test]
    fn vtable_adapter_open_mapping_capabilities() {
        let adapter = CWorldAdapter::new(open_world_vtable());
        let capabilities = adapter.capabilities();
        assert!(capabilities.sweep_point);
        assert!(capabilities.query_rail_contact);
        assert!(capabilities.query_rail_entry_boundary);
        assert!(capabilities.rail_query_contexts);
        assert!(capabilities.rail_entry_contexts);
        assert!(capabilities.query_moving_contact);
        assert!(capabilities.query_network_correction);
        assert!(capabilities.query_distinct_water_volume);

        // sweep_point: JSON object -> mapping.
        let mapping = adapter.sweep_point([0.0; 3], [1.0; 3]).expect("present");
        assert_eq!(mapping.get("hit"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            mapping.get("road_rail"),
            Some(&JsonValue::String(String::new()))
        );

        // query_rail_contact below the release cut: state-1 selector with
        // the alignment gate inputs; ints cross as f64 JSON numbers.
        let selector = adapter
            .query_rail_contact([0.0, 0.0, 0.0], 0, None)
            .expect("capability present")
            .expect("selector present");
        assert_eq!(selector.get("rail_state"), Some(&JsonValue::Number(1.0)));
        assert!(selector.contains_key("registered_query_contexts"));
        // Past the path end the callback reports None (inner Option).
        assert_eq!(
            adapter.query_rail_contact([2.0, 0.0, 0.0], 2, None),
            Some(None)
        );

        // query_rail_entry_boundary: typed mirror, no match -> Some(None).
        assert_eq!(
            adapter.query_rail_entry_boundary([0.0; 3], Some("rail-road"), 0.05),
            Some(None)
        );

        // rail_query_contexts / rail_entry_contexts: list protocol.
        let contexts = adapter.rail_query_contexts().expect("present");
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0].get("rail_state"), Some(&JsonValue::Number(2.0)));
        assert_eq!(adapter.rail_entry_contexts(), Some(Vec::new()));

        // query_moving_contact: scalar args cross as POD; mapping payload.
        let record = adapter
            .query_moving_contact([0.0; 3], 1234, Some("roadobj:42"), 1, 1.5)
            .expect("capability present")
            .expect("record present");
        assert_eq!(record.get("contact_active"), Some(&JsonValue::Bool(true)));

        // query_network_correction: mapping payload.
        let record = adapter
            .query_network_correction([0.0; 3])
            .expect("capability present")
            .expect("record present");
        assert_eq!(
            record.get("mode"),
            Some(&JsonValue::String("snap".to_string()))
        );

        // query_distinct_water_volume: any JSON value is legal.
        assert_eq!(
            adapter.query_distinct_water_volume([0.0; 3]),
            Some(Some(JsonValue::Bool(true)))
        );
    }

    #[test]
    fn vtable_adapter_open_mapping_error_paths() {
        // Invalid JSON in a mapping-typed blob -> query failed (None), the
        // same outer-None the engine turns into its "capability reported
        // present but returned no result" error.
        let adapter = CWorldAdapter::new(KpWorldVtable {
            sweep_point: Some(garbage_sweep_point),
            ..flat_world_vtable()
        });
        assert!(adapter.sweep_point([0.0; 3], [1.0; 3]).is_none());
        // Valid JSON but not an object -> query failed.
        let adapter = CWorldAdapter::new(KpWorldVtable {
            sweep_point: Some(non_object_sweep_point),
            ..flat_world_vtable()
        });
        assert!(adapter.sweep_point([0.0; 3], [1.0; 3]).is_none());
        // A mapping-typed callback reporting no result -> query failed
        // (sweep_point has no inner-None in the trait).
        let adapter = CWorldAdapter::new(KpWorldVtable {
            sweep_point: Some(absent_result_sweep_point),
            ..flat_world_vtable()
        });
        assert!(adapter.sweep_point([0.0; 3], [1.0; 3]).is_none());
        // A double-Option callback reporting no result -> Some(None).
        let adapter = CWorldAdapter::new(KpWorldVtable {
            query_rail_contact: Some(negative_rail_contact),
            query_network_correction: Some(no_network_correction),
            ..flat_world_vtable()
        });
        assert_eq!(adapter.query_rail_contact([0.0; 3], 0, None), Some(None));
        assert_eq!(adapter.query_network_correction([0.0; 3]), Some(None));
    }

    #[test]
    fn vtable_adapter_entry_boundary_typed_mirror() {
        let adapter = CWorldAdapter::new(KpWorldVtable {
            query_rail_entry_boundary: Some(entry_boundary_match),
            ..flat_world_vtable()
        });
        let matched = adapter
            .query_rail_entry_boundary([0.0; 3], Some("rail-road"), 0.05)
            .expect("capability present")
            .expect("match present");
        assert_eq!(matched.context_id, JsonValue::Number(3.0));
        assert_eq!(matched.context_index, 3);
        assert_eq!(matched.road, "rail-road");
        assert_eq!(matched.element, JsonValue::Number(11.0));
        assert_eq!(matched.marker, JsonValue::String("marker-9".to_string()));
        assert_eq!(matched.signed_plane_distance, -0.0125);
        assert_eq!(matched.plane_distance, 0.0125);
        assert_eq!(matched.projected_coordinates, [0.5, -0.25]);
        assert_eq!(matched.path_start, [1.0, 2.0, 3.0]);
        assert_eq!(matched.tangent, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn list_callback_counts_are_clamped() {
        // handoff 2026-07-29 §3.1: a garbage first-call count must fail the
        // query instead of allocating unbounded memory (host abort).
        let adapter = CWorldAdapter::new(KpWorldVtable {
            sweep_obb_rotating_all: Some(garbage_count_sweep),
            overlap_obb_all: Some(garbage_count_overlap),
            rail_query_contexts: Some(garbage_count_contexts),
            ..flat_world_vtable()
        });
        assert!(adapter
            .sweep_obb_rotating_all([0.0; 3], [1.0; 3], [0.0; 9], [0.0; 9], [0.0; 3])
            .is_none());
        assert!(adapter
            .overlap_obb_all([0.0; 3], [0.0; 9], [0.0; 3])
            .is_none());
        assert!(adapter.rail_query_contexts().is_none());
        // A count under the element ceiling but over the byte ceiling is
        // also rejected without allocating.
        let adapter = CWorldAdapter::new(KpWorldVtable {
            rail_query_contexts: Some(over_byte_ceiling_contexts),
            ..flat_world_vtable()
        });
        assert!(adapter.rail_query_contexts().is_none());
        // A legitimate over-stack-capacity count still round-trips through
        // the second call.
        let adapter = CWorldAdapter::new(KpWorldVtable {
            rail_query_contexts: Some(ten_contexts),
            ..flat_world_vtable()
        });
        assert_eq!(
            adapter.rail_query_contexts().map(|list| list.len()),
            Some(10)
        );
    }

    // -- End-to-end FFI RAIL cycle (present) vs v1 degradation (absent) --

    unsafe extern "C" fn rail_smoke_query_ground(
        _user_data: *mut c_void,
        _position: *const f64,
        _up: *const f64,
        _min_dot: f64,
        out: *mut KpWorldGroundResult,
    ) {
        // The RAIL smoke plane row (identical to the C smoke / Python
        // reference world): z=0 with the authored triangle.
        unsafe {
            *out = KpWorldGroundResult {
                track_id: KpString64::set("ffi-smoke").unwrap(),
                node: KpString64::set("n0").unwrap(),
                tri_index: 0,
                surface_id: 0,
                point: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                has_triangle: 1,
                triangle: [
                    -500.0, -500.0, 0.0, //
                    500.0, -500.0, 0.0, //
                    0.0, 500.0, 0.0,
                ],
                ..Default::default()
            }
        };
    }

    unsafe extern "C" fn rail_smoke_sweep_point(
        _user_data: *mut c_void,
        _start: *const f64,
        _end: *const f64,
        out: *mut KpJsonBlob,
    ) -> i32 {
        write_json_out(out, &smoke_sweep_point_mapping());
        1
    }

    unsafe extern "C" fn rail_smoke_contexts(
        _user_data: *mut c_void,
        out: *mut KpJsonBlob,
        out_capacity: usize,
    ) -> usize {
        let contexts = smoke_rail_selector(2);
        if out_capacity >= 1 {
            if let JsonValue::Object(selector) = contexts {
                let context = selector
                    .get("registered_query_contexts")
                    .expect("contexts")
                    .clone();
                if let JsonValue::Array(list) = context {
                    write_json_out(out, &list[0]);
                    return 1;
                }
            }
        }
        0
    }

    fn rail_smoke_vtable(rail_capabilities: bool) -> KpWorldVtable {
        let mut world = KpWorldVtable {
            query_ground: Some(rail_smoke_query_ground),
            ..flat_world_vtable()
        };
        if rail_capabilities {
            world.sweep_point = Some(rail_smoke_sweep_point);
            world.query_rail_contact = Some(recording_rail_contact);
            world.query_rail_entry_boundary = Some(no_entry_boundary_match);
            world.rail_query_contexts = Some(rail_smoke_contexts);
            world.rail_entry_contexts = Some(empty_contexts);
        }
        world
    }

    /// Drive the RAIL smoke scenario (clock-init + 40 frames of 16 ms UP)
    /// and return the per-frame motion domain ids. `create_v2` selects the
    /// creation entry point (v2 handshake vs the v1 compatibility shim).
    fn drive_rail_smoke(world: &KpWorldVtable, create_v2: bool) -> Vec<u8> {
        unsafe {
            let engine = if create_v2 {
                kart_physics_create_v2(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION)
            } else {
                kart_physics_create(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32)
            };
            assert!(!engine.is_null());
            let mut current_parameters = KpVehicleParameters::default();
            let mut v1_parameters = ExactV1Parameters::default();
            let parameters: *const KpVehicleParameters = if create_v2 {
                assert_eq!(
                    kart_physics_parameters_default_v2(&mut current_parameters, KP_ABI_VERSION),
                    KP_OK
                );
                &current_parameters
            } else {
                assert_eq!(
                    kart_physics_parameters_default(v1_parameters.as_mut_ptr()),
                    KP_OK
                );
                v1_parameters.as_const_ptr()
            };
            let mut inputs = KpInputs {
                tick_ms: 1000,
                keys: 0,
            };
            assert_eq!(kart_physics_step(engine, &inputs, parameters, world), KP_OK);
            let mut domains = Vec::new();
            for _ in 0..40 {
                inputs.tick_ms = inputs.tick_ms.wrapping_add(16);
                inputs.keys = KP_KEY_UP;
                assert_eq!(kart_physics_step(engine, &inputs, parameters, world), KP_OK);
                if create_v2 {
                    let mut state = KpVehicleState::default();
                    assert_eq!(kart_physics_get_state(engine, &mut state), KP_OK);
                    domains.push(state.topology.domain);
                } else {
                    let mut state = ExactV1State::default();
                    assert_eq!(kart_physics_get_state(engine, state.as_mut_ptr()), KP_OK);
                    domains.push(state.to_current().topology.domain);
                }
            }
            kart_physics_destroy(engine);
            domains
        }
    }

    /// Assert the v1 degradation contract: every frame stays GND.
    fn assert_all_gnd(domains: &[u8], context: &str) {
        assert!(
            domains.iter().all(|domain| *domain == KP_DOMAIN_GND),
            "{context}: v1 degradation must keep GND ownership: {domains:?}"
        );
    }

    #[test]
    fn ffi_rail_cycle_present_vs_absent() {
        // Present direction: a v2-negotiated engine sees the v2 slots
        // carrying the recorded RAIL smoke world and enters AND exits the
        // RAIL domain (the recorded Python trace: RAIL from frame 1,
        // release at frame 33 once the kart passes the path end at
        // x >= 1.35).
        let domains = drive_rail_smoke(&rail_smoke_vtable(true), true);
        assert_eq!(domains.len(), 40);
        assert_eq!(domains[0], KP_DOMAIN_RAIL, "frame 1 must own RAIL");
        assert!(
            domains
                .iter()
                .all(|domain| *domain == KP_DOMAIN_RAIL || *domain == KP_DOMAIN_GND),
            "unexpected domains in the smoke cycle: {domains:?}"
        );
        assert!(
            domains[1..].iter().any(|domain| *domain != KP_DOMAIN_RAIL),
            "the engine must exit RAIL: {domains:?}"
        );
        assert_eq!(*domains.last().unwrap(), KP_DOMAIN_GND);
        // Absent direction (v1 regression): all v2 slots NULL keeps the v1
        // carve-out behavior — the GND capture probe sees no
        // query_rail_contact capability and the kart stays GND for the
        // whole scenario, exactly like the v1 flat smoke.
        let degraded = drive_rail_smoke(&rail_smoke_vtable(false), true);
        assert_all_gnd(&degraded, "v2 engine, NULL v2 slots");
    }

    #[test]
    fn ffi_rail_v1_shim_ignores_v2_slots() {
        // The N1 semantic gate: a v1-negotiated engine (the
        // kart_physics_create compatibility shim) stepped with a FULL v2
        // vtable whose 8 new slots are populated must NOT see them — the
        // RAIL-capture scenario degrades to GND for every frame, exactly
        // the v1 behavior. The same vtable on a v2-negotiated engine
        // captures RAIL (asserted in ffi_rail_cycle_present_vs_absent).
        let full_v2 = rail_smoke_vtable(true);
        let shimmed = drive_rail_smoke(&full_v2, false);
        assert_all_gnd(&shimmed, "v1 shim engine, full v2 vtable");

        // The exact old-host memory case: a host compiled against the v1
        // header owns only the 80-byte v1 prefix. Step a v1-negotiated
        // engine with a vtable pointer backed by exactly
        // KP_WORLD_VTABLE_V1_BYTES of storage: the negotiated read must
        // stay in bounds and still degrade to GND.
        let v1_storage = ExactV1World::from_current(&full_v2);
        let v1_world = v1_storage.as_const_ptr();
        let degraded = unsafe {
            let engine = kart_physics_create(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32);
            assert!(!engine.is_null());
            let mut parameters = ExactV1Parameters::default();
            assert_eq!(
                kart_physics_parameters_default(parameters.as_mut_ptr()),
                KP_OK
            );
            let mut inputs = KpInputs {
                tick_ms: 1000,
                keys: 0,
            };
            assert_eq!(
                kart_physics_step(engine, &inputs, parameters.as_const_ptr(), v1_world),
                KP_OK
            );
            let mut domains = Vec::new();
            for _ in 0..40 {
                inputs.tick_ms = inputs.tick_ms.wrapping_add(16);
                inputs.keys = KP_KEY_UP;
                assert_eq!(
                    kart_physics_step(engine, &inputs, parameters.as_const_ptr(), v1_world),
                    KP_OK
                );
                let mut state = ExactV1State::default();
                assert_eq!(kart_physics_get_state(engine, state.as_mut_ptr()), KP_OK);
                domains.push(state.to_current().topology.domain);
            }
            kart_physics_destroy(engine);
            domains
        };
        assert_all_gnd(&degraded, "v1 shim engine, 80-byte v1-layout vtable");
    }

    #[test]
    fn ffi_create_v2_version_gate() {
        unsafe {
            // Every version other than KP_ABI_VERSION is refused: NULL
            // handle + KP_ERR_ABI_VERSION detail in the thread-local
            // message.
            for bad_version in [0, 1, 2, 3, 4, u32::MAX] {
                let engine =
                    kart_physics_create_v2(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, bad_version);
                assert!(engine.is_null(), "version {bad_version} must be refused");
                let mut buffer = [0 as c_char; 256];
                let needed = kart_physics_last_error_message(buffer.as_mut_ptr(), buffer.len());
                assert!(needed > 0);
                let message = std::ffi::CStr::from_ptr(buffer.as_ptr()).to_str().unwrap();
                assert!(
                    message.contains("ABI version") && message.contains("KP_ERR_ABI_VERSION"),
                    "unexpected refusal message: {message}"
                );
            }
            // The matching version creates a working engine.
            let engine =
                kart_physics_create_v2(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, KP_ABI_VERSION);
            assert!(!engine.is_null());
            kart_physics_destroy(engine);

            // The compatibility decoder for a pre-v3 negotiated host keeps
            // bit 5 unknown. The public handshake refuses an old header
            // before creation; this direct internal handle exercises the
            // step-side defense as if an ABI-v2 host reached the decoder.
            let v2_engine =
                create_engine_handle(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, 2).unwrap();
            let inputs = KpInputs {
                tick_ms: 0,
                keys: KP_KEY_CTRL,
            };
            let parameters = KpVehicleParameters::default();
            assert_eq!(
                kart_physics_step(v2_engine, &inputs, &parameters, std::ptr::null()),
                KP_ERR_INVALID_INPUT
            );
            kart_physics_destroy(v2_engine);

            // ABI v3 knew CTRL but not the v4 ITEM_USE bit.
            let v3_engine =
                create_engine_handle(std::ptr::null(), KP_NUMERIC_MODE_FLOAT32, 3).unwrap();
            let inputs = KpInputs {
                tick_ms: 0,
                keys: KP_KEY_ITEM_USE,
            };
            assert_eq!(
                kart_physics_step(v3_engine, &inputs, &parameters, std::ptr::null()),
                KP_ERR_INVALID_INPUT
            );
            kart_physics_destroy(v3_engine);
        }
    }
}
