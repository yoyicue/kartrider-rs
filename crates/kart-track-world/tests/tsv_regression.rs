//! TSV protocol byte regression: replays the committed baseline
//! (`analysis/track_world/tsv_baseline_20260729`, frozen argv/stdin manifest
//! plus expected stdout and expected exit code) through
//! `cli::render_command` and the persistent-session function, byte-for-byte.
//!
//! The baseline was recorded from the pre-extraction binary (W1) and
//! re-verified after every later refactor; wiring it into `cargo test` keeps
//! future render-layer drift from slipping through silently.
//!
//! Skips cleanly when the baseline or the track/KAP assets are absent (they
//! are local, gitignored assets).

use std::fs;
use std::path::{Path, PathBuf};

use kart_track_world::cli::{render_command, run_world_query_server_session};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn asset_args_present(argv: &[String], root: &Path) -> bool {
    // Query coordinates/flags never carry a dot-suffixed asset extension, so
    // checking extension-bearing args is enough to gate on missing assets.
    argv.iter()
        .filter(|arg| arg.ends_with(".1s") || arg.ends_with(".kap") || arg.ends_with(".rho"))
        .all(|arg| root.join(arg).is_file())
}

#[test]
fn tsv_baseline_byte_regression() {
    let root = repo_root();
    let baseline = root.join("analysis/track_world/tsv_baseline_20260729");
    if !baseline.is_dir() {
        eprintln!("skipping: TSV baseline not present at {baseline:?}");
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(baseline.join("cases.json")).expect("read cases.json"),
    )
    .expect("parse cases.json");
    let cases = manifest["cases"].as_array().expect("cases array");

    // argv asset paths are repo-relative; resolve them exactly as recorded.
    std::env::set_current_dir(&root).expect("chdir repo root");

    let mut skipped_missing_assets = 0usize;
    let mut compared = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        // KAP-only cases are filed under _global (map is null in the manifest).
        let map = case["map"].as_str().unwrap_or("_global");
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        let command = argv[0].clone();
        assert!(
            command == "world-query-server" || kart_track_world::cli::is_query_command(&command),
            "baseline case {map}/{name} is not owned by kart-track-world: {command}"
        );
        let stem = baseline.join(map).join(format!("{command}_{name}"));
        let expected_out = fs::read(stem.with_extension("out"))
            .unwrap_or_else(|err| panic!("read {}: {err}", stem.with_extension("out").display()));
        let expected_code: i32 = fs::read_to_string(stem.with_extension("code"))
            .unwrap_or_else(|err| panic!("read {}: {err}", stem.with_extension("code").display()))
            .trim()
            .parse()
            .expect("exit code integer");
        // Gate on missing assets only for cases expected to succeed: the
        // missing-track cases are error-path fixtures whose asset is
        // intentionally absent.
        if expected_code == 0 && !asset_args_present(&argv, &root) {
            skipped_missing_assets += 1;
            continue;
        }

        if command == "world-query-server" {
            let stdin = case["stdin"].as_str().expect("server case stdin");
            let mut out: Vec<u8> = Vec::new();
            let result = run_world_query_server_session(
                Path::new(&argv[1]),
                std::io::Cursor::new(stdin.as_bytes()),
                &mut out,
            );
            if expected_code == 0 {
                if let Err(err) = &result {
                    failures.push(format!("{map}/{name}: session failed: {err}"));
                }
            } else if result.is_ok() {
                failures.push(format!("{map}/{name}: session unexpectedly ok"));
            }
            if out != expected_out {
                failures.push(format!(
                    "{map}/{name}: session bytes differ (got {} want {})",
                    out.len(),
                    expected_out.len()
                ));
            }
            compared += 1;
            continue;
        }

        match render_command(&command, &argv[1..]) {
            Ok(rendered) => {
                if expected_code != 0 {
                    failures.push(format!(
                        "{map}/{name}: expected exit {expected_code} but render ok"
                    ));
                }
                if rendered.as_bytes() != expected_out.as_slice() {
                    failures.push(format!(
                        "{map}/{name}: stdout bytes differ (got {} want {})",
                        rendered.len(),
                        expected_out.len()
                    ));
                }
            }
            Err(err) => {
                if expected_code == 0 {
                    failures.push(format!(
                        "{map}/{name}: expected exit 0 but render failed: {}",
                        err.message
                    ));
                }
                if err.partial.as_bytes() != expected_out.as_slice() {
                    failures.push(format!(
                        "{map}/{name}: partial stdout bytes differ (got {} want {})",
                        err.partial.len(),
                        expected_out.len()
                    ));
                }
            }
        }
        compared += 1;
    }

    eprintln!(
        "tsv regression: {compared} cases compared, \
         {skipped_missing_assets} skipped (missing assets)"
    );
    assert_eq!(
        compared + skipped_missing_assets,
        cases.len(),
        "every baseline case must be owned and accounted for"
    );
    assert!(compared > 0, "manifest had no runnable cases");
    assert!(
        failures.is_empty(),
        "TSV baseline regressions ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
