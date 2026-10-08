# kartrider-rs

A clean-room Rust re-implementation of the KartRider asset stack, built toward a
native (wgpu) viewer / playable demo. Reuses the **format knowledge** we reverse-
engineered for the C# `.1s→glTF` pipeline (see `../docs/`, `../tools/Kart1sToGltf`),
but shares **no code** with the GPL KartLibrary — everything here is reimplemented
from the documented formats and validated against the C# output.

> **Repository scope:** this repository contains the Rust workspace only. The
> local reference implementations it cross-validates against (the C#
> `Kart1sToGltf` pipeline, the closed `oracle-local` karttools, captured
> game-client data) are **not included** — `cargo xval` / `cargo smoke` skip
> gracefully when they are absent. No game assets or client files are bundled;
> you point the tools at your own installed game client (see
> *End-to-end client pipeline* below). See `docs/rho5-format.md` for format
> research notes.

## Why Rust + wgpu

- One core crate → native CLI **and** WASM (browser) from the same code.
- Kills the macOS arm64/x64 (Rosetta) split the C# RhoUnpacker needs.
- No GPL runtime dependency; `wgpu` renders natively and in the browser.

## Workspace layout

```
crates/
  kart-rho       .rho/.rho5 unpack + PIN + aaa.pk runtime mount decode     ✅  (bin: kart-unpack)
  kart-fmt       .1s parse: raw model · KartObject · anim · DDS · param    ✅  (zero deps; the core)
                 + BML (BinaryXml) + track_common config (track registry,
                   challenge, randomTrack, rail) as typed, cross-validated data
  kart-render    wgpu render core: reusable Renderer + headless PNG        ✅
  kart-gltf      .1s → glTF/GLB export (parity with C# Kart1sToGltf)      ✅  (bin: kart-gltf)
  kart-usd       .1s → USD ASCII (.usda) export (oracle-local format)     ✅  (bin: kart-usd)
  kart-pipeline  batch convert-all (glTF/USDA) + theme-tex + coverage     ✅  (bin: kart-pipeline)
  kart-auditor   visual audit reports: GLB stats + source PNG/contact sheet ✅ (bin: kart-audit)
  kart-viewer    interactive winit+wgpu viewer: orbit/fly cam · KAP AI    ✅  (bin: kart-viewer)
                   path overlay · skeletal-animation playback · asset cycle
  kart-physics   vehicle physics engine: bit-exact port of the Python     ✅  (lib + cdylib,
                 reference (scripts/kartrider_vehicle_engine.py) — all      frozen C ABI in
                 domains (GND/AIR/CONTACT/RAIL/MOVING-CONTACT/RESET/        include/kart_physics.h)
                 NET-CORRECTION), rollback, scheduler; f32-quantization
                 discipline mirrors the authority exactly
xtask            cross-validation harness vs C# + oracle (`cargo xval`)   ✅
```

Dependency layering is acyclic: `kart-rho` (unpack) and `kart-fmt` (parse, no
internal deps) at the base; `kart-render` / `kart-gltf` / `kart-usd` each depend
only on `kart-fmt`; `kart-pipeline` and `xtask` sit on top. Entry-point
convention: a library crate's bundled CLI is `src/bin/<name>.rs` (e.g.
`kart-rho` → `kart-unpack`, `kart-gltf`, `kart-usd`); a pure-binary crate uses
`src/main.rs` (e.g. `kart-pipeline`, `kart-auditor`); debug / hashing probes
are `examples/` (e.g. `kart-fmt`'s `dump_*`). `kart-auditor` keeps a small
`[[bin]]` rename only so the installed command is `kart-audit`.

`kart-viewer` (the interactive winit+wgpu runtime) is the first route-A step —
it reuses the `kart-render` pipeline live. `kart-game` (drift physics) is still
deferred (M7+), but the *replicated* vehicle physics engine now lives in
`kart-physics`: a bit-exact Rust port of the Python authority
(`../scripts/kartrider_vehicle_engine.py` + `evaluate_*` modules), validated
per-substep bit-identical on captured free-running, rail-cycle, and
contact-cycle scenarios (see `../scripts/verify_kart_physics_*_parity.py` and
`../analysis/motion_topology/rust_*_parity_20260728.json`). Porting contract:
`../docs/kart-physics-rust-handoff-2026-07-28.md`.

## Milestones

| # | crate | goal | status |
|---|-------|------|--------|
| **M1a** | kart-rho | crypto primitives (keys, XOR + header ciphers) | ✅ done, unit-tested vs C# |
| M1b | kart-rho | `.rho` unpack: header → block-info → folder/file tree → zlib blocks | ✅ byte-identical vs C# (10203 files) |
| M1c | kart-rho | `.rho5` unpack (custom AES-like cipher; SSE2 log sidestepped via digit-count) | ✅ byte-identical vs C# (1185 files) |
| M1d | kart-rho | `KartRider.pin` read-only decode (`KREncodedBlock` + PIN header/auth/BML) | ✅ smoke-tested against `modern/kart/KartRider.pin` |
| M1e | kart-rho | `Data/aaa.pk` read-only decode (`PackFolder` runtime RHO mount list) | ✅ 1611/1611 mounted RHO present in `modern/kart/Data` |
| M2 | kart-fmt | parse raw model (character/kart) → vertices/normals/uvs/faces | ✅ geometry FNV-identical vs C# (23 models) |
| M3 | kart-render | wgpu headless render of a mesh → PNG (window later) | ✅ character + kart rendered |
| M4 | kart-render | textures + materials (PNG + DDS BC1/2/3) | ✅ char/kart PNG-textured; DDS decode verified |
| M5 | kart-fmt/render | skin + skeletal animation playback | ✅ anim parse FNV-identical vs C# (16 clips); CPU-skinned poses render |
| M6 | kart-fmt/render | track (KartObject protocol) geometry + camera | ✅ parser byte-identical vs C# (669/670 track files); tracks render textured |
| M7+ | kart-game | drift physics — "a kart drifting on a track" | (deferred) |

### Replicated vehicle physics (kart-physics)

| # | scope | status |
|---|-------|--------|
| P1 | core step: scheduler + integrators + GND/GND-SLIP | ✅ bit-exact vs Python (5150-substep free-running scenario) |
| P2 | all domains: CONTACT / AIR / LAND / RAIL(1-6) / MOVING-CONTACT / RESET | ✅ bit-exact; 1877-step rail cycle + 5150-step contact cycle vs real track world |
| P3 | rollback + network correction | ✅ bit-exact replay trajectories |
| P4 | frozen C ABI (`cdylib`, `include/kart_physics.h`, C smoke) | ✅ dependency-free; bit-exact through the FFI |

Validation methodology: per-domain trace parity — Python authority dumps
per-substep state as f64 bit patterns, Rust replays with a scripted world
replaying recorded world-query responses, diff is bit-exact (not epsilon).
Suites: `../scripts/verify_kart_physics_*_parity.py`. Build/test on this
volume requires `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/kart-rust-target`
(the project volume does not support cargo file locks).

### Export-tool parity (E-series)

Goal: make the Rust stack an **export tool** equivalent to the C# `Kart1sToGltf`
(.1s → glTF) and catch up to the Python `oracle-local` karttools. Validation is
semantic: convert with both, load each `.glb` and compare node/mesh/vertex/
triangle/material/animation/skin data.

| # | crate | goal | status |
|---|-------|------|--------|
| E1 | kart-gltf | minimal GLB writer (nodes, meshes, accessors) + raw-model export | ✅ geometry parity vs C# (world bbox + tris + mats) |
| E2 | kart-gltf | materials (alpha/cull/double-side) + DDS→PNG textures | ✅ folded into E1/E5 |
| E3 | kart-gltf | skins + character skeletal animation export | ✅ parity vs C# (anims/channels/joints + bind bbox) |
| E4 | kart-fmt/gltf | kart rigid-body anim parser + export (+ single-mesh dummy collapse) | ✅ parser 7/7 FNV vs C#; kart geometry+anim glTF parity |
| E5 | kart-gltf | track export (KartObject → glTF) + track PRS anim + UV-anim extras | ✅ geometry/material/anim-channel parity vs C# (track PRS incl. ThreeAxis rotation + UV-curve extras) |
| E6 | kart-pipeline | batch convert-all + theme-tex association + coverage | ✅ **100.00%** of 16167 convertible (16167/16167); Rust handles the legacy `smoke/effect.1s` layout that C# still misses |
| E7 | kart-fmt | oracle-local: vehicle `param.xml` physics parser (typed coercion) | ✅ **71/71 exact vs oracle**; all 1412 parse (40761 typed values) |
| E8 | kart-usd | oracle-local: USD ASCII (`.usda`) exporter (static geometry + skel customData) | ✅ models + tracks; tri/mesh/material counts match the glTF path |

**Oracle-local cross-validation** (ran the closed karttools 1.2.0 in its bundled
`.venv` Python 3.13): track parser **8/8 tracks** match oracle's node/mesh counts;
kart anim **exact** (55 channels / frames / duration); `param.xml` **71/71 exact**.
The two oracle features that looked "missing" turn out to be **non-functional in
oracle itself**: `parse_track` only stores `event_count` (To* zones are *not*
decoded — we expose `event_count` too), and localization returns `None`
everywhere (its `asset_names.csv` is absent and the `.bml` source is never
decoded). So nothing oracle actually produces is left unmatched.

End goal is route **A** (assets → render → self-authored physics), a *neishi*
playable tribute. The current goal — **export parity with the C# converter and
oracle-local** — is met; native rendering (M3–M6) already exceeds what the C#
converter does.

## Validation strategy

Every layer is checked against the existing, validated C# output **and** the
oracle-local karttools (run via its `.venv`):
- crypto / unpack → byte-compare extracted files vs the C# RhoUnpacker
- model/anim/track parse → FNV / count compare vs both the C# glb and oracle
- param.xml → exact dict compare vs oracle `parse_kart_params`

## Build

```bash
cd rust
cargo test          # crypto vs reference vectors
cargo smoke         # local modern-client smoke: PIN + aaa.pk + tiny pipeline run
cargo build --release
```

## End-to-end client pipeline

`kart-pipeline` can run either from an already-unpacked archive tree or directly
from a modern client root containing `LK.txf` and `Data/aaa.pk`:

```bash
# existing mode: convert an unpacked tree
cargo run -p kart-pipeline -- pipeline/unpacked pipeline/glb-rs --anims

# end-to-end: LK.txf -> .rho/.rho5 unpack -> GLB/USDA conversion
cargo run -p kart-pipeline -- --client ../modern/kart pipeline-rs --anims
cargo run -p kart-pipeline -- --client ../modern/kart pipeline-rs --usd
```

Client mode uses `aaa.pk` as the authoritative Launcher-style `.rho` mount list
when present. Ordinary `.rho` archives are materialized under their virtual
`PackFolder/RhoFolder` mount paths (for example `track_/common`, `theme_/village`),
while `.rho5` datapacks stay under their `DataPack*_*.rho5` extraction namespace
for collision-safe incremental unpacking. It writes `pipeline-rs/unpacked/`,
`pipeline-rs/glb-rs/` or `pipeline-rs/usda-rs/`, plus `pipeline-rs/state.tsv`
for manifest-MD5 based incremental unpack skips. Use `--rebuild` to force
re-unpack, `--limit N` for a smoke run, and `--rho5-salt cn|kr|tw|<salt>` when
targeting a non-CN client.

## Visual audit

`kart-auditor` turns a GLB pipeline output into a reviewable static report:

```bash
cargo run -p kart-auditor -- pipeline-rs audit-rs --per-family 8 --rebuild
open audit-rs/index.html
```

It reads `pipeline-rs/glb-rs/` for structural GLB stats, directly renders each
sampled GLB to a PNG, maps it back to `pipeline-rs/unpacked/`, renders the source
`.1s` as a reference PNG, then writes `index.html`, `report.md`, `report.json`,
`contact-sheet.png`, and per-item JSON/PNG files. This is an auditor, not the
deferred interactive `kart-viewer`: the purpose is to catch blank renders, broken
bbox/geometry, bad texture lookup, and family-level visual regressions quickly.

## License

MIT — see [LICENSE](LICENSE). All format knowledge here was obtained by
clean-room reverse engineering of file formats for interoperability purposes;
this repository contains no game assets, client binaries, or original game code.
