// perf_step.mjs — WASM-side per-frame kart_wasm_step timing for the P4
// performance gate (docs/kart-physics-performance-handoff-2026-07-30.md §2-P4).
// Mirrors the native perf_step example: same sequence, same session call
// order, one warmup rep excluded, first-step catch-up burst reported
// separately. Zero npm deps.
//
// usage: node rust/crates/kart-wasm/runner/perf_step.mjs \
//   --wasm <kart_wasm.wasm> --sequence <seq.json> --track <track.1s> [--reps 3]

import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { KartWasmSession } from '../../../../kart-three/engine.js';

function parseArgs(argv) {
  const args = { reps: 3 };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--reps') {
      args.reps = Number(argv[++i]);
    } else if (argv[i].startsWith('--')) {
      args[argv[i].slice(2)] = argv[++i];
    }
  }
  for (const key of ['wasm', 'sequence', 'track']) {
    if (!args[key]) throw new Error(`missing --${key}`);
  }
  return args;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const rank = Math.round((sorted.length - 1) * p);
  return sorted[Math.min(rank, sorted.length - 1)];
}

const args = parseArgs(process.argv.slice(2));
const [wasmBytes, trackBytes, sequence] = await Promise.all([
  readFile(args.wasm),
  readFile(args.track),
  readFile(args.sequence, 'utf8').then(JSON.parse),
]);
const frameMs = sequence.frame_ms;
const frames = [];
for (const [keys, count] of sequence.segments) {
  for (let i = 0; i < count; i += 1) frames.push(keys);
}
const numericMode = sequence.numeric_mode === 'float32' ? 0 : 1;
const module_ = await KartWasmSession.compile(wasmBytes);

let warmupMs = 0;
const timedMs = [];
const firstStepMs = [];
for (let rep = 0; rep <= args.reps; rep += 1) {
  const session = await KartWasmSession.instantiate(module_);
  session.loadWorld(trackBytes, sequence.track_identity);
  session.createEngine(JSON.stringify(sequence.initial_state), numericMode);
  session.setParameters(JSON.stringify(sequence.parameters));
  for (let index = 0; index < frames.length; index += 1) {
    const started = performance.now();
    session.step(frames[index], (index * frameMs) >>> 0);
    const took = performance.now() - started;
    if (rep === 0) warmupMs += took;
    else if (index === 0) firstStepMs.push(took);
    else timedMs.push(took);
  }
}

timedMs.sort((a, b) => a - b);
firstStepMs.sort((a, b) => a - b);
const mean = timedMs.reduce((a, b) => a + b, 0) / timedMs.length;
console.log(JSON.stringify({
  schema: 'kart_physics_perf_step_v1',
  runner: 'node rust/crates/kart-wasm/runner/perf_step.mjs',
  track_identity: sequence.track_identity,
  numeric_mode: sequence.numeric_mode,
  frames_per_rep: frames.length,
  frame_ms: frameMs,
  warmup_reps: 1,
  timed_reps: args.reps,
  warmup_total_ms: warmupMs,
  steady_per_frame_ms: {
    mean,
    p50: percentile(timedMs, 0.5),
    p95: percentile(timedMs, 0.95),
    p99: percentile(timedMs, 0.99),
    max: percentile(timedMs, 1.0),
  },
  first_step_catchup_ms: {
    note: 'frame-scheduler catch-up burst from the captured initial state\'s scheduler tick; one per engine instance, excluded from steady_per_frame_ms',
    p50: percentile(firstStepMs, 0.5),
    max: percentile(firstStepMs, 1.0),
  },
}, null, 2));
