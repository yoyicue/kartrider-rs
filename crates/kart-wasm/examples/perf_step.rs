//! perf_step — per-frame `kart_wasm_step` CPU timing for the P4 performance
//! gate (`docs/kart-physics-performance-handoff-2026-07-30.md` §2-P4).
//!
//! Runs the shared self-check sequence through the same kart-wasm session
//! exports the browser drives (identical call order to `self_check_hash`),
//! timing every `kart_wasm_step` call. One warmup rep (which includes the
//! one-time track query-index build) is excluded from the statistics.
//!
//! Usage:
//!
//! ```bash
//! cargo run -p kart-wasm --release --example perf_step -- \
//!   --sequence kart-three/self_check_sequence.json \
//!   --track analysis/rail/factory_I02.track.1s [--reps 3]
//! ```
//!
//! Prints one JSON document on stdout: frame/rep counts, warmup total, and
//! per-frame milliseconds pooled across the timed reps
//! (mean/p50/p95/p99/max).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

struct Args {
    sequence: PathBuf,
    track: PathBuf,
    reps: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut sequence = None;
    let mut track = None;
    let mut reps = 3u32;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--sequence" => sequence = Some(PathBuf::from(it.next().ok_or("--sequence needs a value")?)),
            "--track" => track = Some(PathBuf::from(it.next().ok_or("--track needs a value")?)),
            "--reps" => {
                reps = it
                    .next()
                    .ok_or("--reps needs a value")?
                    .parse()
                    .map_err(|_| "--reps must be an integer".to_string())?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        sequence: sequence.ok_or("missing --sequence")?,
        track: track.ok_or("missing --track")?,
        reps,
    })
}

fn last_error() -> String {
    unsafe {
        let ptr = kart_wasm::kart_wasm_last_error_ptr();
        let len = kart_wasm::kart_wasm_last_error_len();
        if ptr.is_null() || len == 0 {
            return String::new();
        }
        let bytes = std::slice::from_raw_parts(ptr, len);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

fn check(status: i32, what: &str) -> Result<(), String> {
    if status == kart_wasm::KART_WASM_OK {
        Ok(())
    } else {
        Err(format!("{what} failed with status {status}: {}", last_error()))
    }
}

fn with_buffer(bytes: &[u8], f: impl FnOnce(*const u8, usize) -> i32) -> i32 {
    unsafe {
        let ptr = kart_wasm::kart_wasm_alloc(bytes.len());
        if ptr.is_null() {
            return kart_wasm::KART_WASM_ERR_NULL;
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        let status = f(ptr, bytes.len());
        kart_wasm::kart_wasm_free(ptr, bytes.len());
        status
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn run(args: &Args) -> Result<String, String> {
    let sequence_text = std::fs::read_to_string(&args.sequence)
        .map_err(|e| format!("read {}: {e}", args.sequence.display()))?;
    let sequence: serde_json::Value =
        serde_json::from_str(&sequence_text).map_err(|e| format!("parse sequence: {e}"))?;
    let track_identity = sequence["track_identity"]
        .as_str()
        .ok_or("sequence.track_identity must be a string")?;
    let numeric_mode = match sequence["numeric_mode"].as_str() {
        Some("float32") => kart_wasm::KART_WASM_NUMERIC_MODE_FLOAT32,
        Some("float64") => kart_wasm::KART_WASM_NUMERIC_MODE_FLOAT64,
        other => return Err(format!("sequence.numeric_mode must be float32|float64, got {other:?}")),
    };
    let frame_ms = sequence["frame_ms"]
        .as_u64()
        .ok_or("sequence.frame_ms must be an integer")? as u32;
    let segments = sequence["segments"]
        .as_array()
        .ok_or("sequence.segments must be an array")?;
    let mut frames: Vec<u32> = Vec::new();
    for segment in segments {
        let keys = segment[0].as_u64().ok_or("segment keys must be an integer")? as u32;
        let count = segment[1].as_u64().ok_or("segment count must be an integer")?;
        for _ in 0..count {
            frames.push(keys);
        }
    }
    let state_json = serde_json::to_vec(&sequence["initial_state"])
        .map_err(|e| format!("encode initial_state: {e}"))?;
    let params_json = serde_json::to_vec(&sequence["parameters"])
        .map_err(|e| format!("encode parameters: {e}"))?;
    let track = std::fs::read(&args.track)
        .map_err(|e| format!("read {}: {e}", args.track.display()))?;

    let mut warmup_ms = 0.0;
    let mut timed_ms: Vec<f64> = Vec::with_capacity(frames.len() * args.reps as usize);
    let mut first_step_ms: Vec<f64> = Vec::new();
    let mut outliers: Vec<serde_json::Value> = Vec::new();
    for rep in 0..=args.reps {
        // Fresh session per rep, same call order as the browser (README §Session).
        let phase_started = Instant::now();
        let identity = track_identity.as_bytes();
        let status = with_buffer(&track, |track_ptr, track_len| unsafe {
            kart_wasm::kart_wasm_load_world(track_ptr, track_len, identity.as_ptr(), identity.len())
        });
        check(status, "kart_wasm_load_world")?;
        eprintln!("rep {rep} load_world: {:.3} ms", phase_started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = Instant::now();
        let status = with_buffer(&state_json, |ptr, len| unsafe {
            kart_wasm::kart_wasm_create_engine(ptr, len, numeric_mode)
        });
        check(status, "kart_wasm_create_engine")?;
        eprintln!("rep {rep} create_engine: {:.3} ms", phase_started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = Instant::now();
        let status = with_buffer(&params_json, |ptr, len| unsafe {
            kart_wasm::kart_wasm_set_parameters(ptr, len)
        });
        check(status, "kart_wasm_set_parameters")?;
        eprintln!("rep {rep} set_parameters: {:.3} ms", phase_started.elapsed().as_secs_f64() * 1000.0);

        for (index, keys) in frames.iter().enumerate() {
            let elapsed_ms = (index as u32).wrapping_mul(frame_ms);
            let started = Instant::now();
            let status = kart_wasm::kart_wasm_step(*keys, elapsed_ms);
            let took = started.elapsed().as_secs_f64() * 1000.0;
            check(status, &format!("kart_wasm_step frame {index}"))?;
            if rep == 0 {
                warmup_ms += took;
            } else if index == 0 {
                // First step of a fresh engine: the frame scheduler's catch-up
                // burst from the captured state's scheduler tick (one per
                // engine instance, ~250 substeps for the run129 capture) —
                // reported separately, excluded from steady-state percentiles.
                first_step_ms.push(took);
            } else {
                if took > 20.0 {
                    outliers.push(serde_json::json!({
                        "rep": rep,
                        "frame": index,
                        "ms": took,
                    }));
                }
                timed_ms.push(took);
            }
        }
    }

    timed_ms.sort_by(f64::total_cmp);
    first_step_ms.sort_by(f64::total_cmp);
    let mean = timed_ms.iter().sum::<f64>() / timed_ms.len() as f64;
    let document = serde_json::json!({
        "schema": "kart_physics_perf_step_v1",
        "runner": "native rust/crates/kart-wasm/examples/perf_step.rs",
        "track_identity": track_identity,
        "numeric_mode": sequence["numeric_mode"],
        "frames_per_rep": frames.len(),
        "frame_ms": frame_ms,
        "warmup_reps": 1,
        "timed_reps": args.reps,
        "warmup_total_ms": warmup_ms,
        "steady_per_frame_ms": {
            "mean": mean,
            "p50": percentile(&timed_ms, 0.50),
            "p95": percentile(&timed_ms, 0.95),
            "p99": percentile(&timed_ms, 0.99),
            "max": percentile(&timed_ms, 1.0),
        },
        "first_step_catchup_ms": {
            "note": "frame-scheduler catch-up burst from the captured initial state's scheduler tick; one per engine instance, excluded from steady_per_frame_ms",
            "p50": percentile(&first_step_ms, 0.50),
            "max": percentile(&first_step_ms, 1.0),
        },
        "outliers_over_20ms": outliers,
    });
    serde_json::to_string_pretty(&document).map_err(|e| format!("encode document: {e}"))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("usage: perf_step --sequence <seq.json> --track <track.1s> [--reps N]");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(document) => {
            println!("{document}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
