//! Native generator for the W4 demo self-check expected hash
//! (`docs/kart-physics-wasm-demo-handoff-2026-07-29.md` §2-W4).
//!
//! Runs the shared built-in key sequence (`kart-three/self_check_sequence.json`)
//! through the SAME `kart-wasm` session exports the browser drives
//! (`kart_wasm_load_world` → `kart_wasm_create_engine` →
//! `kart_wasm_set_parameters` → `kart_wasm_step` × N), hashes the
//! all 312 deterministic `KartWasmSnapshot` POD field bytes after every step
//! with SHA-256,
//! and writes the expected-hash document (`kart-three/self_check_expected.json`).
//! The page and
//! `kart-three/check_self_check.mjs` reproduce the identical procedure inside the
//! .wasm and compare against this value, so native-vs-wasm bit drift on the
//! `step()` path fails loudly.
//!
//! Usage:
//!
//! ```bash
//! cargo run -p kart-wasm --example self_check_hash -- \
//!   --sequence kart-three/self_check_sequence.json \
//!   --track analysis/rail/factory_I02.track.1s \
//!   [--out kart-three/self_check_expected.json]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use sha2::{Digest, Sha256};

struct Args {
    sequence: PathBuf,
    track: PathBuf,
    out: Option<PathBuf>,
    continuous_up: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut sequence = None;
    let mut track = None;
    let mut out = None;
    let mut continuous_up = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--sequence" => {
                sequence = Some(PathBuf::from(it.next().ok_or("--sequence needs a value")?))
            }
            "--track" => track = Some(PathBuf::from(it.next().ok_or("--track needs a value")?)),
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a value")?)),
            "--continuous-up" => continuous_up = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        sequence: sequence.ok_or("missing --sequence")?,
        track: track.ok_or("missing --track")?,
        out,
        continuous_up,
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
        Err(format!(
            "{what} failed with status {status}: {}",
            last_error()
        ))
    }
}

/// Passes `bytes` to `f` through the host→wasm transfer protocol.
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

fn snapshot_bytes() -> Vec<u8> {
    let ptr = kart_wasm::kart_wasm_snapshot_ptr();
    let len = kart_wasm::kart_wasm_snapshot_len();
    assert!(
        !ptr.is_null(),
        "snapshot pointer is null after a successful step"
    );
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
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
        other => {
            return Err(format!(
                "sequence.numeric_mode must be float32|float64, got {other:?}"
            ))
        }
    };
    let frame_ms = sequence["frame_ms"]
        .as_u64()
        .ok_or("sequence.frame_ms must be an integer")? as u32;
    let segments = sequence["segments"]
        .as_array()
        .ok_or("sequence.segments must be an array")?;
    let mut frames: Vec<u32> = Vec::new();
    for segment in segments {
        let keys = segment[0]
            .as_u64()
            .ok_or("segment keys must be an integer")? as u32;
        let count = segment[1]
            .as_u64()
            .ok_or("segment count must be an integer")?;
        for _ in 0..count {
            frames.push(keys);
        }
    }
    if args.continuous_up {
        frames.fill(kart_wasm::KART_WASM_KEY_UP);
    }
    let state_json = serde_json::to_vec(&sequence["initial_state"])
        .map_err(|e| format!("encode initial_state: {e}"))?;
    let params_json = serde_json::to_vec(&sequence["parameters"])
        .map_err(|e| format!("encode parameters: {e}"))?;
    let track =
        std::fs::read(&args.track).map_err(|e| format!("read {}: {e}", args.track.display()))?;

    // The exact session call order the browser performs (README §Session).
    let identity = track_identity.as_bytes();
    let status = with_buffer(&track, |track_ptr, track_len| unsafe {
        kart_wasm::kart_wasm_load_world(track_ptr, track_len, identity.as_ptr(), identity.len())
    });
    check(status, "kart_wasm_load_world")?;
    let status = with_buffer(&state_json, |ptr, len| unsafe {
        kart_wasm::kart_wasm_create_engine(ptr, len, numeric_mode)
    });
    check(status, "kart_wasm_create_engine")?;
    let status = with_buffer(&params_json, |ptr, len| unsafe {
        kart_wasm::kart_wasm_set_parameters(ptr, len)
    });
    check(status, "kart_wasm_set_parameters")?;

    let mut hasher = Sha256::new();
    let mut checkpoints = Vec::new();
    let mut previous_continuous_up_checkpoint: Option<serde_json::Value> = None;
    for (index, keys) in frames.iter().enumerate() {
        let elapsed_ms = (index as u32).wrapping_mul(frame_ms);
        let status = kart_wasm::kart_wasm_step(*keys, elapsed_ms);
        check(status, &format!("kart_wasm_step frame {index}"))?;
        // ABI v4 has no implicit padding: the alignment byte is an explicit
        // zero-valued field, so every snapshot byte is deterministic.
        let snapshot = snapshot_bytes();
        hasher.update(&snapshot[..kart_wasm::KART_WASM_SNAPSHOT_FIELD_LEN]);
        if args.continuous_up {
            let read_f64 = |offset: usize| -> f64 {
                f64::from_le_bytes(snapshot[offset..offset + 8].try_into().expect("8 bytes"))
            };
            let read_i64 = |offset: usize| -> i64 {
                i64::from_le_bytes(snapshot[offset..offset + 8].try_into().expect("8 bytes"))
            };
            let current = serde_json::json!({
                "frame": index,
                "position": [read_f64(8), read_f64(16), read_f64(24)],
                "velocity": [read_f64(32), read_f64(40), read_f64(48)],
                "rail_state": read_i64(184),
                "domain": snapshot[208],
            });
            if checkpoints.is_empty()
                && previous_continuous_up_checkpoint
                    .as_ref()
                    .is_some_and(|previous| {
                        previous["domain"] == 6
                            && previous["rail_state"] == 6
                            && current["domain"] == 0
                            && current["rail_state"] == 0
                    })
            {
                checkpoints.push(
                    previous_continuous_up_checkpoint
                        .as_ref()
                        .expect("checked above")
                        .clone(),
                );
                checkpoints.push(current.clone());
            }
            previous_continuous_up_checkpoint = Some(current);
        }
    }
    let sha256 = format!("{:x}", hasher.finalize());

    // Reporting fields (not part of the hash): where the run ended up.
    let snapshot = snapshot_bytes();
    let read_f64 = |offset: usize| -> f64 {
        f64::from_le_bytes(snapshot[offset..offset + 8].try_into().expect("8 bytes"))
    };
    let final_position = [read_f64(8), read_f64(16), read_f64(24)];
    let final_velocity = [read_f64(32), read_f64(40), read_f64(48)];
    let final_domain = snapshot[208];
    eprintln!(
        "frames={} frame_ms={frame_ms} final_position={final_position:?} final_domain={final_domain}",
        frames.len()
    );

    let mut document = serde_json::json!({
        "version": 1,
        "sequence_version": sequence["version"],
        "hash_algorithm": "sha256",
        "hash_input": "concatenation of all 312 deterministic KartWasmSnapshot POD bytes read after every kart_wasm_step call (including the first clock-establishing call); ABI v4 has no implicit padding",
        "frames": frames.len(),
        "frame_ms": frame_ms,
        "numeric_mode": sequence["numeric_mode"],
        "track_identity": track_identity,
        "input_mode": if args.continuous_up {
            "continuous_up_override"
        } else {
            "sequence_segments"
        },
        "sha256": sha256,
        "generator": "rust/crates/kart-wasm/examples/self_check_hash.rs (native)",
        "final_position": final_position,
        "final_velocity": final_velocity,
        "final_domain": final_domain,
    });
    if args.continuous_up {
        if checkpoints.len() != 2 {
            return Err(
                "continuous-UP rail-exit gate requires a state6-to-ground boundary".to_string(),
            );
        }
        let before_velocity_x = checkpoints[0]["velocity"][0]
            .as_f64()
            .expect("native checkpoint velocity is f64");
        let after_velocity_x = checkpoints[1]["velocity"][0]
            .as_f64()
            .expect("native checkpoint velocity is f64");
        document["checkpoints"] = serde_json::Value::Array(checkpoints);
        document["semantic_checks"] = serde_json::json!({
            "rail_exit_before_is_state6": (
                document["checkpoints"][0]["domain"] == 6
                    && document["checkpoints"][0]["rail_state"] == 6
            ),
            "rail_exit_after_is_ground": (
                document["checkpoints"][1]["domain"] == 0
                    && document["checkpoints"][1]["rail_state"] == 0
            ),
            "rail_exit_retains_at_least_90_percent_abs_vx": (
                after_velocity_x.abs() >= before_velocity_x.abs() * 0.9
            ),
        });
    }
    serde_json::to_string_pretty(&document).map_err(|e| format!("encode document: {e}"))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("usage: self_check_hash --sequence <seq.json> --track <track.1s> [--out <expected.json>] [--continuous-up]");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(document) => {
            if let Some(out) = &args.out {
                if let Err(e) = std::fs::write(out, format!("{document}\n")) {
                    eprintln!("error: write {}: {e}", out.display());
                    return ExitCode::FAILURE;
                }
                eprintln!("wrote {}", out.display());
            } else {
                println!("{document}");
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
