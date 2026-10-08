# kart-wasm

WASM (wasm32-unknown-unknown) boundary for the kart-physics engine plus the
kart-track-world `TrackWorld` — milestone W3 of
`docs/kart-physics-wasm-demo-handoff-2026-07-29.md`.

Red lines (handoff §3): no wasm-bindgen; plain `extern "C"` exports over
linear memory; JS never computes a physical number — the boundary shapes are
exactly `step(keys_bitmask, elapsed_ms)` and `snapshot() -> pointer/length
POD view`, plus bytes-in (assets/spec JSON) and bytes-out (result/error)
helpers. All numerics stay inside the .wasm.

## Build

```bash
cd rust
export CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/kart-rust-target
cargo build -p kart-wasm --release --target wasm32-unknown-unknown
# artifact: /tmp/kart-rust-target/wasm32-unknown-unknown/release/kart_wasm.wasm
```

No wasm-opt pass is required (none is applied; default-safe flags only).
Current size: ~2.3 MB (unstripped, LTO thin). The module is freestanding:
`WebAssembly.Module.imports` is empty — instantiate with `{}`.

## Export ABI (version `kart_wasm_abi_version() == 4`)

Memory protocol:

- `kart_wasm_alloc(len) -> ptr` / `kart_wasm_free(ptr, len)` — host→wasm
  transfer buffers; every export copies what it needs before returning, so
  the host frees right after the call.
- `kart_wasm_result_ptr/len`, `kart_wasm_last_error_ptr/len`,
  `kart_wasm_snapshot_ptr/len` — wasm→host views into session-owned memory;
  valid until the next session-mutating call. Copy, never free, never write
  back.

Session (demo loop, W4):

1. `kart_wasm_load_kap(kap_ptr, kap_len, path_ptr, path_len)` — optional,
   repeatable, MUST precede the world load (checkpoint selectors).
2. `kart_wasm_load_world(track_ptr, track_len, path_ptr, path_len)` —
   `.track.1s` bytes plus the identity path string (for bit-parity with a
   native run this is the exact `scenario.track` string).
3. `kart_wasm_create_engine(state_json_ptr, state_json_len, numeric_mode)` —
   initial `VehicleState` as the spec-shaped serde JSON; `numeric_mode` 0 =
   float32 (production), 1 = float64.
4. `kart_wasm_set_parameters(params_json_ptr, params_json_len)` — optional;
   captured `VehicleParameters` JSON. Default: `VehicleParameters::default()`
   (the captured reference configuration).
5. `kart_wasm_step(keys_bitmask, elapsed_ms) -> status` — one game frame;
   `elapsed_ms` is the u32 game-clock tick (accumulated whole ms, wraps at
   2^32; first call establishes the clock without advancing physics).
   Keys: bit0 UP, bit1 DOWN, bit2 LEFT, bit3 RIGHT, bit4 R, bit5 CTRL
   (semantic drift), bit6 ITEM_USE (current-family ordinary N₂O consume).
6. `kart_wasm_snapshot_ptr/len` — the `KartWasmSnapshot` POD view (312
   bytes, `#[repr(C)]`, offsets pinned by the `snapshot_layout` unit test):
   frame_index(u64), position/velocity/angular_velocity(f64[3]),
   matrix(f64[9]), time_ms(u64), rail_state_elapsed_ms(f64),
   transition_index/domain_step_index/rail_state/boost_state/
   boost_remaining_ms(i64), domain + previous/return domain with has-flags
   (u8; ids match the kart-physics FFI `KP_DOMAIN_*`: 0=GND … 6=RAIL …
   10=NET-CORRECTION), then `selector_b08` at byte 213. The ABI v3 tail is
   item-use latch + explicit zero aligner, fixed two-slot item ids/quantities,
   four raw gauge fields, authoritative display amount, and cap. ABI v4 adds
   the banana-effect remaining milliseconds and scoped event index. There is no
   implicit padding; all bytes `[0,312)` are deterministic and hashed.
7. `kart_wasm_snapshot_field_len()` — deterministic snapshot byte count
   (312, equal to the full POD size in ABI v4).

Status codes: 0 OK, 1 null pointer, 2 invalid state, 3 invalid params,
4 invalid input (key bits / numeric mode / call order), 5 world load,
7 step error, 9 no engine. Message via `kart_wasm_last_error_*`. Rust
panics are NOT caught (wasm traps); exports return errors on all
host-driven inputs.

Parity-gate replay (W3, independent of the session engine):

- `kart_wasm_run_replay(spec_ptr, spec_len, track_ptr, track_len, path_ptr,
  path_len)` — runs a W2-style live-world parity spec end-to-end inside the
  .wasm through the SAME `kart_track_world::replay::run_live_world_replay`
  code the native `live_world_trace` example executes; output JSON via
  `kart_wasm_result_ptr/len`.

## Node runner

```bash
node rust/crates/kart-wasm/runner/run_spec.mjs \
  --wasm /tmp/kart-rust-target/wasm32-unknown-unknown/release/kart_wasm.wasm \
  --spec <spec.json> [--track <track.1s>] [--out <result.json>]
```

Plain node (no npm deps). `--track` defaults to `spec.scenario.track` read
from the host filesystem; in the browser a `fetch()` ArrayBuffer takes its
place.

## Bit-parity gate (W3 acceptance)

```bash
python3 scripts/verify_kart_physics_wasm_parity.py          # full 6-scenario matrix
python3 scripts/verify_kart_physics_wasm_parity.py --reuse-specs   # reuse /tmp specs
```

Regenerates/reuses the six W2 specs, replays each natively and through the
.wasm, diffs the two documents (f64s cross as IEEE bit-pattern hex strings,
so JSON equality is bit equality), re-applies the W2 budgets/verdicts, and
writes `analysis/motion_topology/rust_wasm_parity_20260729.json`.
Requirement: native-vs-wasm divergence = 0 on every scenario.

## Tests

```bash
cd rust && cargo test -p kart-wasm
cargo clippy -p kart-wasm --all-targets -- -D warnings
```
