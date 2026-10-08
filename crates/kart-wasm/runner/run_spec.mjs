#!/usr/bin/env node
// W3 parity-gate runner: executes one W2-style live-world parity spec JSON
// end-to-end INSIDE the kart-wasm .wasm and prints the emitted result rows.
//
// Plain node, zero npm dependencies. The wasm module is freestanding (no
// imports); all numerics stay inside wasm — this script only copies bytes
// in (spec JSON, track.1s) and copies the result JSON bytes out.
//
// Usage:
//   node run_spec.mjs --wasm <kart_wasm.wasm> --spec <spec.json> \
//       [--track <track.1s>] [--out <result.json>]
//
// --track defaults to the spec's scenario.track path (read from the host
// filesystem; in the browser this is where a fetch() would hand over the
// ArrayBuffer instead). The result document ({"rows": [...],
// "world_response_diff": {...}}) goes to --out or stdout.

import { readFileSync, writeFileSync } from "node:fs";
import { argv, exit, stdout } from "node:process";

function parseArgs(argv) {
  const args = { track: null, out: null };
  for (let i = 2; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === "--wasm") args.wasm = value;
    else if (flag === "--spec") args.spec = value;
    else if (flag === "--track") args.track = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--help" || flag === "-h") {
      stdout.write("usage: node run_spec.mjs --wasm <wasm> --spec <spec.json> [--track <track.1s>] [--out <result.json>]\n");
      exit(0);
    } else {
      throw new Error(`unknown argument ${flag}`);
    }
    i += 1;
  }
  if (!args.wasm || !args.spec) throw new Error("--wasm and --spec are required");
  return args;
}

function main() {
  const args = parseArgs(argv);
  const specBytes = readFileSync(args.spec);
  const spec = JSON.parse(specBytes.toString("utf8"));
  const trackPath = args.track ?? spec?.scenario?.track;
  if (!trackPath) throw new Error("no track path: pass --track or set spec.scenario.track");
  const trackBytes = readFileSync(trackPath);
  const wasmBytes = readFileSync(args.wasm);

  const module = new WebAssembly.Module(wasmBytes);
  const imports = WebAssembly.Module.imports(module);
  if (imports.length > 0) {
    throw new Error(`unexpected wasm imports (module must be freestanding): ${JSON.stringify(imports)}`);
  }
  const instance = new WebAssembly.Instance(module, {});
  const wasm = instance.exports;
  if (wasm.kart_wasm_abi_version?.() !== 4) {
    throw new Error(`kart-wasm ABI version mismatch: ${wasm.kart_wasm_abi_version?.()}`);
  }

  const memory = wasm.memory;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  // Copies a byte buffer into wasm linear memory; returns {ptr, len}.
  // Freed by the caller after the export returns (exports copy what they
  // need before returning — see the kart-wasm memory protocol).
  function pushBytes(bytes) {
    const ptr = wasm.kart_wasm_alloc(bytes.length);
    if (!ptr) throw new Error(`kart_wasm_alloc(${bytes.length}) failed`);
    new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  }

  function lastError() {
    const ptr = wasm.kart_wasm_last_error_ptr();
    const len = wasm.kart_wasm_last_error_len();
    return len > 0 ? decoder.decode(new Uint8Array(memory.buffer, ptr, len)) : "";
  }

  const specBuf = pushBytes(specBytes);
  const trackBuf = pushBytes(trackBytes);
  const pathBuf = pushBytes(encoder.encode(trackPath));
  let rc;
  try {
    rc = wasm.kart_wasm_run_replay(
      specBuf.ptr, specBuf.len,
      trackBuf.ptr, trackBuf.len,
      pathBuf.ptr, pathBuf.len,
    );
  } finally {
    wasm.kart_wasm_free(specBuf.ptr, specBuf.len);
    wasm.kart_wasm_free(trackBuf.ptr, trackBuf.len);
    wasm.kart_wasm_free(pathBuf.ptr, pathBuf.len);
  }
  if (rc !== 0) {
    throw new Error(`kart_wasm_run_replay failed with status ${rc}: ${lastError()}`);
  }

  const resultPtr = wasm.kart_wasm_result_ptr();
  const resultLen = wasm.kart_wasm_result_len();
  // Copy out of linear memory before any further wasm call could invalidate it.
  const resultBytes = new Uint8Array(resultLen);
  resultBytes.set(new Uint8Array(memory.buffer, resultPtr, resultLen));
  if (args.out) {
    writeFileSync(args.out, resultBytes);
  } else {
    stdout.write(resultBytes);
    stdout.write("\n");
  }
}

try {
  main();
} catch (error) {
  console.error(`run_spec: ${error.message}`);
  exit(1);
}
