# kart-autoprobe

`kart-autoprobe` is the runtime control side of the physics probe loop. It is
not a racing AI; it is a low-speed experiment driver that connects authored KAP
geometry to runtime probe feedback.

## Data Flow

```text
track.1s KAP geometry
  -> targets: road.surface / road.rail triangle centers
probe_pose.tsv
  -> current kart position, yaw, and speed
probe_collision_face.tsv
  -> current runtime surface_id / face identity
kart-autoprobe
  -> SendInput control steps
  -> probe_control.tsv-compatible action log
```

On Windows, control timestamps use `GetTickCount64()` so they line up with
probe `tick_ms` logs. On non-Windows hosts the binary runs in dry-run mode and
prints the intended key events.

## Commands

List authored KAP targets for a surface:

```bash
cargo run -p kart-autoprobe -- \
  targets ../pipeline/unpacked/DataPack3_00027.rho5/track_/abyss_I03/track.1s \
  --surface slip --limit 8
```

Generate the low-speed vehicle response calibration script:

```bash
cargo run -p kart-autoprobe -- calibration-script --out calibration.tsv
cargo run -p kart-autoprobe -- script calibration.tsv --log probe_control.tsv
cargo run -p kart-autoprobe -- fit-model probe_pose.tsv probe_control.tsv --out vehicle_model.tsv
```

Drive toward an explicit world-space point using probe pose feedback:

```bash
kart-autoprobe drive-to D:\kr\probe\probe_pose.tsv 220.56 411.54 22.17 \
  --model D:\kr\probe\vehicle_model.tsv \
  --log D:\kr\probe\probe_control.tsv \
  --collision D:\kr\probe\probe_collision_face.tsv \
  --stop-on-surface-id 144966072
```

Drive toward the nearest authored KAP surface target and stop when runtime
collision confirms the matching surface:

```bash
kart-autoprobe drive-surface D:\kr\track\abyss_I03\track.1s D:\kr\probe\probe_pose.tsv \
  --surface slip \
  --model D:\kr\probe\vehicle_model.tsv \
  --collision D:\kr\probe\probe_collision_face.tsv \
  --stop-on-surface \
  --log D:\kr\probe\probe_control.tsv
```

By default, `drive-surface` builds a lightweight drivable-space mesh from the
same `track.1s`: upward-facing triangles are connected by shared vertices, while
reset/pit/transparent faces are excluded. The largest drivable connected
component is treated as the main course loop. Target selection first filters to
the current drivable component, then scores main-loop targets by forward ring
progress from the current pose, using the kart yaw to choose clockwise vs
counter-clockwise progress. It falls back to nearest-by-distance only if the
loop/component test cannot produce a target. Use `--ignore-reachability` to
disable this filter for debugging.

For full KAP centerline following, call `drive-surface` without a surface token
and pass the AI path directory. The controller selects the KAP lane nearest to
the current pose, projects the kart onto that path every step, and drives toward
a lookahead point on the ring:

```bash
kart-autoprobe drive-surface D:\kr\track\china_I03\track.1s D:\kr\probe\probe_pose.tsv \
  --kap-dir D:\kr\track_common\aiPath \
  --kap-track china_I03 \
  --lookahead 12 \
  --max-ms 20000 \
  --timeout-ok \
  --log D:\kr\probe\probe_control.tsv
```

The HTTP daemon exposes the same mode through `/control/drive-surface` with
`kap_dir`, `kap_track`, optional `kap_slot`, `lookahead`, and `max_cross_track`
query parameters.

## Current Model

The first controller model is deliberately simple:

- target direction from `atan2(target.y - y, target.x - x)`
- yaw from `probe_pose.post_ang_z`
- speed from `sqrt(vel_x^2 + vel_y^2)`
- hard/soft turn thresholds loaded from `vehicle_model.tsv`
- estimated brake distance from `speed^2 / (2 * brake_decel)`
- hard-turn braking when speed is too high for the fitted turn rate
- target braking when the fitted brake distance reaches the remaining target distance

The calibration script and `fit-model` command estimate:

```text
max_control_speed
hard_turn_deg
soft_turn_deg
forward_accel
coast_decel
brake_decel
turn_rate_low
turn_rate_mid
```

These values are still a low-speed heuristic controller, not a full physics
model. They are meant to make repeated experiments reach target faces more
reliably while the real physics formulas are being recovered.
