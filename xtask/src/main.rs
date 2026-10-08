//! Workspace task harness.
//!
//! `cargo xval` diffs this workspace's clean-room parsers/exporter against the
//! two reference implementations when they are present locally:
//!   * the closed-freeware oracle `karttools` (Python) — exact parser parity, and
//!   * the GPL-derived C# `Kart1sToGltf` — structural glTF-export parity.
//!
//! Both references are gitignored, so each section SKIPs cleanly (exit 0) when its
//! reference is absent — a fresh clone or CI run reports SKIP, not failure. With a
//! reference present, any mismatch fails the run (exit 1).
//!
//! `cargo smoke` is the local modern-client smoke: `KartRider.pin`,
//! `Data/aaa.pk`, and a tiny `kart-pipeline --client` run.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

mod glb;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/rust/xtask
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn driver_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/oracle_driver.py")
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("xval") | Some("cross-validate") => run_xval(),
        Some("smoke") => run_smoke_cmd(args.collect()),
        Some("audit-corpus") => run_audit_corpus_cmd(args.collect()),
        Some("kap") => run_kap_cmd(args.collect()),
        Some("kap-gltf") => run_kap_gltf(args.collect()),
        Some("-h") | Some("--help") | Some("help") => print_usage(),
        Some(cmd) => {
            eprintln!("unknown xtask command: {cmd}");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo xval                 # cross-validate vs local references\n  cargo xtask smoke          # local modern-client smoke\n  cargo smoke                # same as cargo xtask smoke\n  cargo xtask audit-corpus [input-root] [run-root] [options]\n  cargo xtask kap [dir]      # decode+parse every .kap (default: pipeline/unpacked)\n  cargo xtask kap-gltf <track.1s> <out.glb> <kap>...   # track + AI paths as one glTF"
    );
}

// ---- audit corpus orchestration -------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisualMode {
    None,
    Risk,
    Sample,
    All,
}

impl VisualMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" | "off" => Ok(Self::None),
            "risk" => Ok(Self::Risk),
            "sample" => Ok(Self::Sample),
            "all" => Ok(Self::All),
            _ => Err(format!(
                "bad --visual value: {value} (expected none|risk|sample|all)"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Risk => "risk",
            Self::Sample => "sample",
            Self::All => "all",
        }
    }
}

struct AuditCorpusArgs {
    input_root: PathBuf,
    run_root: PathBuf,
    rebuild: bool,
    skip_convert: bool,
    pipeline_limit: Option<usize>,
    audit_limit: Option<usize>,
    visual: VisualMode,
    per_family: usize,
    width: u32,
    height: u32,
    contact_sheet_limit: usize,
    families: Option<String>,
    anims: bool,
    clamp_outliers: bool,
    rho5_salt: Option<String>,
    fail_on_issue: bool,
}

fn run_audit_corpus_cmd(args: Vec<String>) {
    match AuditCorpusArgs::parse(args).and_then(|args| run_audit_corpus(&args)) {
        Ok(()) => println!("audit-corpus OK"),
        Err(err) => {
            eprintln!("audit-corpus FAILED: {err}");
            std::process::exit(1);
        }
    }
}

impl AuditCorpusArgs {
    fn parse(raw: Vec<String>) -> Result<Self, String> {
        let root = repo_root();
        let mut pos = Vec::new();
        let mut rebuild = false;
        let mut skip_convert = false;
        let mut pipeline_limit = None;
        let mut audit_limit = None;
        let mut visual = VisualMode::Sample;
        let mut per_family = 8usize;
        let mut width = 640u32;
        let mut height = 640u32;
        let mut contact_sheet_limit = 500usize;
        let mut families = None;
        let mut anims = false;
        let mut clamp_outliers = false;
        let mut rho5_salt = None;
        let mut fail_on_issue = false;

        let mut i = 0;
        while i < raw.len() {
            match raw[i].as_str() {
                "-h" | "--help" => {
                    Self::usage();
                    std::process::exit(0);
                }
                "--rebuild" => rebuild = true,
                "--skip-convert" => skip_convert = true,
                "--pipeline-limit" => {
                    i += 1;
                    pipeline_limit = Some(parse_usize_arg("--pipeline-limit", raw.get(i))?);
                }
                "--audit-limit" => {
                    i += 1;
                    audit_limit = Some(parse_usize_arg("--audit-limit", raw.get(i))?);
                }
                "--visual" => {
                    i += 1;
                    visual = VisualMode::parse(parse_value_arg("--visual", raw.get(i))?)?;
                }
                "--per-family" => {
                    i += 1;
                    per_family = parse_usize_arg("--per-family", raw.get(i))?;
                }
                "--width" => {
                    i += 1;
                    width = parse_u32_arg("--width", raw.get(i))?;
                }
                "--height" => {
                    i += 1;
                    height = parse_u32_arg("--height", raw.get(i))?;
                }
                "--contact-sheet-limit" => {
                    i += 1;
                    contact_sheet_limit = parse_usize_arg("--contact-sheet-limit", raw.get(i))?;
                }
                "--no-contact-sheet" => contact_sheet_limit = 0,
                "--families" => {
                    i += 1;
                    families = Some(parse_value_arg("--families", raw.get(i))?.to_string());
                }
                "--anims" => anims = true,
                "--clamp-outliers" => clamp_outliers = true,
                "--rho5-salt" => {
                    i += 1;
                    rho5_salt = Some(parse_value_arg("--rho5-salt", raw.get(i))?.to_string());
                }
                "--fail-on-issue" => fail_on_issue = true,
                x if x.starts_with('-') => return Err(format!("unknown audit-corpus option: {x}")),
                x => pos.push(PathBuf::from(x)),
            }
            i += 1;
        }

        if pos.len() > 2 {
            Self::usage();
            return Err("audit-corpus accepts at most <input-root> <run-root>".into());
        }

        let input_root = pos
            .first()
            .cloned()
            .unwrap_or_else(|| root.join("modern/kart"));
        let run_root = pos
            .get(1)
            .cloned()
            .unwrap_or_else(|| root.join("pipeline/audit-corpus"));

        Ok(Self {
            input_root,
            run_root,
            rebuild,
            skip_convert,
            pipeline_limit,
            audit_limit,
            visual,
            per_family,
            width,
            height,
            contact_sheet_limit,
            families,
            anims,
            clamp_outliers,
            rho5_salt,
            fail_on_issue,
        })
    }

    fn usage() {
        eprintln!(
            "usage:\n  cargo xtask audit-corpus [input-root] [run-root] [options]\n\n<input-root> may be a modern client root with LK.txf, or an existing pipeline root containing glb-rs/.\nDefault input-root is modern/kart; default run-root is pipeline/audit-corpus.\n\nOptions:\n  --skip-convert              treat input-root as an existing pipeline root\n  --rebuild                   rebuild pipeline/audit outputs instead of reusing their state\n  --pipeline-limit N          forward --limit N to kart-pipeline client conversion\n  --audit-limit N             cap audited items in each auditor pass\n  --visual none|risk|sample|all\n                              visual render mode: skip, render static WARN/FAIL, sample, or all\n  --per-family N              visual sample count per family (default: 8)\n  --families a,b              restrict auditor passes to families\n  --width W --height H        visual render size (default: 640x640)\n  --contact-sheet-limit N     visual contact sheet cap (default: 500)\n  --no-contact-sheet          skip visual contact sheet\n  --anims                     forward --anims to kart-pipeline\n  --clamp-outliers            forward --clamp-outliers to kart-pipeline\n  --rho5-salt S               forward --rho5-salt to kart-pipeline\n  --fail-on-issue             exit non-zero after reports if any stage has FAIL items"
        );
    }
}

fn parse_value_arg<'a>(label: &str, value: Option<&'a String>) -> Result<&'a str, String> {
    value
        .map(String::as_str)
        .ok_or_else(|| format!("{label} requires a value"))
}

fn parse_usize_arg(label: &str, value: Option<&String>) -> Result<usize, String> {
    parse_value_arg(label, value)?
        .parse()
        .map_err(|_| format!("bad {label} value"))
}

fn parse_u32_arg(label: &str, value: Option<&String>) -> Result<u32, String> {
    parse_value_arg(label, value)?
        .parse()
        .map_err(|_| format!("bad {label} value"))
}

fn run_audit_corpus(args: &AuditCorpusArgs) -> Result<(), String> {
    std::fs::create_dir_all(&args.run_root)
        .map_err(|e| format!("mkdir {}: {e}", args.run_root.display()))?;
    std::fs::create_dir_all(args.run_root.join("logs"))
        .map_err(|e| format!("mkdir {}: {e}", args.run_root.join("logs").display()))?;

    let root = repo_root();
    let pipeline_root = prepare_audit_pipeline(&root, args)?;

    let static_root = args.run_root.join("audit-static");
    run_auditor_static(&root, args, &pipeline_root, &static_root)?;
    let static_report = static_root.join("report.json");
    let static_digest = load_report_digest("static", &static_report)?;

    let mut risk_include_list = None;
    let visual_digest = match args.visual {
        VisualMode::None => None,
        VisualMode::Risk => {
            risk_include_list = write_risk_include_list(args, &static_digest)?;
            if let Some(include_list) = risk_include_list.as_deref() {
                let visual_root = args.run_root.join("audit-visual");
                run_auditor_visual(
                    &root,
                    args,
                    &pipeline_root,
                    &visual_root,
                    Some(include_list),
                )?;
                Some(load_report_digest(
                    "visual",
                    &visual_root.join("report.json"),
                )?)
            } else {
                println!("audit-corpus: visual risk audit skipped (static pass found no WARN/FAIL assets)");
                None
            }
        }
        VisualMode::Sample | VisualMode::All => {
            let visual_root = args.run_root.join("audit-visual");
            run_auditor_visual(&root, args, &pipeline_root, &visual_root, None)?;
            Some(load_report_digest(
                "visual",
                &visual_root.join("report.json"),
            )?)
        }
    };

    write_audit_corpus_reports(
        args,
        &pipeline_root,
        &static_digest,
        visual_digest.as_ref(),
        risk_include_list.as_deref(),
    )?;

    let visual_fail = visual_digest.as_ref().map(|d| d.fail).unwrap_or(0);
    if args.fail_on_issue && (static_digest.fail > 0 || visual_fail > 0) {
        return Err(format!(
            "issues detected: static_fail={} visual_fail={}",
            static_digest.fail, visual_fail
        ));
    }
    Ok(())
}

fn prepare_audit_pipeline(root: &Path, args: &AuditCorpusArgs) -> Result<PathBuf, String> {
    if args.skip_convert || args.input_root.join("glb-rs").is_dir() {
        if !args.input_root.join("glb-rs").is_dir() {
            return Err(format!(
                "{} is not a pipeline root (missing glb-rs/)",
                args.input_root.display()
            ));
        }
        println!(
            "audit-corpus: using existing pipeline root {}",
            args.input_root.display()
        );
        return Ok(args.input_root.clone());
    }

    if !args.input_root.join("LK.txf").is_file() {
        return Err(format!(
            "{} is neither a modern client root (missing LK.txf) nor a pipeline root (missing glb-rs/)",
            args.input_root.display()
        ));
    }

    let pipeline_root = args.run_root.join("pipeline");
    println!(
        "audit-corpus: converting client {} -> {}",
        args.input_root.display(),
        pipeline_root.display()
    );
    let mut cmd = cargo_run(root, "kart-pipeline");
    cmd.arg("--client")
        .arg(&args.input_root)
        .arg(&pipeline_root);
    if args.anims {
        cmd.arg("--anims");
    }
    if args.rebuild {
        cmd.arg("--rebuild");
    }
    if args.clamp_outliers {
        cmd.arg("--clamp-outliers");
    }
    if let Some(n) = args.pipeline_limit {
        cmd.args(["--limit", &n.to_string()]);
    }
    if let Some(salt) = &args.rho5_salt {
        cmd.args(["--rho5-salt", salt]);
    }
    run_logged(cmd, &args.run_root.join("logs/pipeline.log"))?;
    Ok(pipeline_root)
}

fn run_auditor_static(
    root: &Path,
    args: &AuditCorpusArgs,
    pipeline_root: &Path,
    audit_root: &Path,
) -> Result<(), String> {
    println!(
        "audit-corpus: static audit {} -> {}",
        pipeline_root.display(),
        audit_root.display()
    );
    let mut cmd = cargo_run(root, "kart-auditor");
    cmd.arg(pipeline_root)
        .arg(audit_root)
        .args(["--all", "--no-render", "--no-contact-sheet"]);
    push_common_auditor_args(&mut cmd, args);
    run_logged(cmd, &args.run_root.join("logs/audit-static.log"))
}

fn run_auditor_visual(
    root: &Path,
    args: &AuditCorpusArgs,
    pipeline_root: &Path,
    audit_root: &Path,
    include_list: Option<&Path>,
) -> Result<(), String> {
    println!(
        "audit-corpus: visual audit ({}) {} -> {}",
        args.visual.label(),
        pipeline_root.display(),
        audit_root.display()
    );
    let mut cmd = cargo_run(root, "kart-auditor");
    cmd.arg(pipeline_root).arg(audit_root);
    match args.visual {
        VisualMode::None => {}
        VisualMode::Risk => {
            cmd.arg("--all");
        }
        VisualMode::Sample => {
            cmd.args(["--per-family", &args.per_family.to_string()]);
        }
        VisualMode::All => {
            cmd.arg("--all");
        }
    }
    cmd.args(["--width", &args.width.to_string()])
        .args(["--height", &args.height.to_string()])
        .args([
            "--contact-sheet-limit",
            &args.contact_sheet_limit.to_string(),
        ]);
    if let Some(path) = include_list {
        cmd.arg("--include-list").arg(path);
    }
    push_common_auditor_args(&mut cmd, args);
    run_logged(cmd, &args.run_root.join("logs/audit-visual.log"))
}

fn push_common_auditor_args(cmd: &mut Command, args: &AuditCorpusArgs) {
    if args.rebuild {
        cmd.arg("--rebuild");
    }
    if let Some(n) = args.audit_limit {
        cmd.args(["--limit", &n.to_string()]);
    }
    if let Some(families) = &args.families {
        cmd.args(["--families", families]);
    }
}

fn cargo_run(root: &Path, package: &str) -> Command {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(root.join("rust"))
        .args(["run", "--release", "-q", "-p", package, "--"]);
    cmd
}

fn run_logged(mut cmd: Command, log_path: &Path) -> Result<(), String> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let command_line = format!("{cmd:?}");
    let output = cmd
        .output()
        .map_err(|e| format!("spawn {command_line}: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let log = format!(
        "$ {command_line}\nstatus: {}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n",
        output.status
    );
    std::fs::write(log_path, log).map_err(|e| format!("write {}: {e}", log_path.display()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{command_line} exited {}; see {}{}",
            output.status,
            log_path.display(),
            tail_hint(&stderr)
        ))
    }
}

fn tail_hint(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().rev().take(6).collect();
    if lines.is_empty() {
        String::new()
    } else {
        format!(
            "\nstderr tail:\n{}",
            lines.into_iter().rev().collect::<Vec<_>>().join("\n")
        )
    }
}

#[derive(Default)]
struct ReportDigest {
    stage: String,
    path: PathBuf,
    candidates: usize,
    sampled: usize,
    rendered: usize,
    ok: usize,
    warn: usize,
    fail: usize,
    issue_counts: BTreeMap<String, usize>,
    family_rows: Vec<FamilyDigest>,
    issue_rows: Vec<IssueDigest>,
}

struct FamilyDigest {
    family: String,
    sampled: usize,
    warn: usize,
    fail: usize,
}

struct IssueDigest {
    stage: String,
    status: String,
    family: String,
    rel_path: String,
    code: String,
    severity: String,
    detail: String,
    glb_png_path: String,
    source_png_path: String,
}

fn load_report_digest(stage: &str, path: &Path) -> Result<ReportDigest, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let summary = value
        .get("summary")
        .ok_or_else(|| format!("{} missing summary", path.display()))?;

    let mut digest = ReportDigest {
        stage: stage.to_string(),
        path: path.to_path_buf(),
        candidates: json_usize(summary, "candidates"),
        sampled: json_usize(summary, "sampled"),
        rendered: json_usize(summary, "rendered"),
        ok: json_usize(summary, "ok"),
        warn: json_usize(summary, "warn"),
        fail: json_usize(summary, "fail"),
        ..Default::default()
    };

    if let Some(counts) = summary.get("issue_counts").and_then(Value::as_object) {
        for (code, count) in counts {
            digest
                .issue_counts
                .insert(code.clone(), count.as_u64().unwrap_or(0) as usize);
        }
    }

    if let Some(families) = value.get("families").and_then(Value::as_array) {
        for family in families {
            let warn = json_usize(family, "warn");
            let fail = json_usize(family, "fail");
            if warn > 0 || fail > 0 {
                digest.family_rows.push(FamilyDigest {
                    family: json_str(family, "family"),
                    sampled: json_usize(family, "sampled"),
                    warn,
                    fail,
                });
            }
        }
    }
    digest
        .family_rows
        .sort_by_key(|f| std::cmp::Reverse((f.fail, f.warn, f.sampled)));

    if let Some(items) = value.get("items").and_then(Value::as_array) {
        for item in items {
            let Some(issues) = item.get("issues").and_then(Value::as_array) else {
                continue;
            };
            for issue in issues {
                digest.issue_rows.push(IssueDigest {
                    stage: stage.to_string(),
                    status: json_str(item, "status"),
                    family: json_str(item, "family"),
                    rel_path: json_str(item, "rel_path"),
                    code: json_str(issue, "code"),
                    severity: json_str(issue, "severity"),
                    detail: json_str(issue, "detail"),
                    glb_png_path: json_str(item, "glb_png_path"),
                    source_png_path: json_str(item, "source_png_path"),
                });
            }
        }
    }

    Ok(digest)
}

fn json_usize(value: &Value, key: &str) -> usize {
    value.get(key).and_then(Value::as_u64).unwrap_or(0) as usize
}

fn json_str(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn write_risk_include_list(
    args: &AuditCorpusArgs,
    static_digest: &ReportDigest,
) -> Result<Option<PathBuf>, String> {
    let mut paths = BTreeSet::new();
    for issue in &static_digest.issue_rows {
        if matches!(issue.severity.as_str(), "warn" | "fail") || issue.status != "OK" {
            paths.insert(issue.rel_path.clone());
        }
    }
    if paths.is_empty() {
        return Ok(None);
    }

    let path = args.run_root.join("audit-risk-include.txt");
    let mut text = String::new();
    text.push_str("# GLB rel_path entries from the static audit WARN/FAIL set.\n");
    text.push_str("# Passed to kart-auditor --include-list for --visual risk.\n");
    for rel_path in paths {
        text.push_str(&rel_path);
        text.push('\n');
    }
    std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(Some(path))
}

fn write_audit_corpus_reports(
    args: &AuditCorpusArgs,
    pipeline_root: &Path,
    static_digest: &ReportDigest,
    visual_digest: Option<&ReportDigest>,
    risk_include_list: Option<&Path>,
) -> Result<(), String> {
    let generated_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut md = String::new();
    md.push_str("# audit-corpus summary\n\n");
    md.push_str(&format!(
        "- generated_unix: `{generated_unix}`\n- input_root: `{}`\n- pipeline_root: `{}`\n- run_root: `{}`\n- visual: `{}`\n",
        args.input_root.display(),
        pipeline_root.display(),
        args.run_root.display(),
        args.visual.label()
    ));
    md.push_str("\n## Stages\n\n");
    md.push_str("| stage | report | candidates | sampled | rendered | ok | warn | fail |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");
    push_stage_row(&mut md, static_digest);
    if let Some(visual) = visual_digest {
        push_stage_row(&mut md, visual);
    }

    md.push_str("\n## Issue Clusters\n\n");
    md.push_str("| stage | issue | count |\n");
    md.push_str("|---|---|---:|\n");
    push_issue_cluster_rows(&mut md, static_digest);
    if let Some(visual) = visual_digest {
        push_issue_cluster_rows(&mut md, visual);
    }

    md.push_str("\n## Families With Issues\n\n");
    md.push_str("| stage | family | sampled | warn | fail |\n");
    md.push_str("|---|---|---:|---:|---:|\n");
    push_family_rows(&mut md, static_digest);
    if let Some(visual) = visual_digest {
        push_family_rows(&mut md, visual);
    }

    md.push_str("\n## Outputs\n\n");
    md.push_str(&format!(
        "- static report: `{}`\n",
        static_digest.path.display()
    ));
    if let Some(visual) = visual_digest {
        md.push_str(&format!("- visual report: `{}`\n", visual.path.display()));
    }
    if let Some(path) = risk_include_list {
        md.push_str(&format!("- risk include list: `{}`\n", path.display()));
    }
    md.push_str(&format!(
        "- issue table: `{}`\n",
        args.run_root.join("audit-corpus-issues.tsv").display()
    ));

    std::fs::write(args.run_root.join("audit-corpus-summary.md"), md)
        .map_err(|e| format!("write audit-corpus-summary.md: {e}"))?;
    write_issue_tsv(args, static_digest, visual_digest)
}

fn push_stage_row(out: &mut String, digest: &ReportDigest) {
    out.push_str(&format!(
        "| {} | `{}` | {} | {} | {} | {} | {} | {} |\n",
        digest.stage,
        digest.path.display(),
        digest.candidates,
        digest.sampled,
        digest.rendered,
        digest.ok,
        digest.warn,
        digest.fail
    ));
}

fn push_issue_cluster_rows(out: &mut String, digest: &ReportDigest) {
    let mut rows: Vec<_> = digest.issue_counts.iter().collect();
    rows.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    if rows.is_empty() {
        out.push_str(&format!("| {} | none | 0 |\n", digest.stage));
    }
    for (code, count) in rows {
        out.push_str(&format!(
            "| {} | `{}` | {} |\n",
            digest.stage,
            clean_md_cell(code),
            count
        ));
    }
}

fn push_family_rows(out: &mut String, digest: &ReportDigest) {
    if digest.family_rows.is_empty() {
        out.push_str(&format!("| {} | none | 0 | 0 | 0 |\n", digest.stage));
    }
    for family in digest.family_rows.iter().take(40) {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            digest.stage,
            clean_md_cell(&family.family),
            family.sampled,
            family.warn,
            family.fail
        ));
    }
}

fn write_issue_tsv(
    args: &AuditCorpusArgs,
    static_digest: &ReportDigest,
    visual_digest: Option<&ReportDigest>,
) -> Result<(), String> {
    let mut out = String::from(
        "stage\tstatus\tfamily\tseverity\tcode\trel_path\tdetail\tglb_png_path\tsource_png_path\n",
    );
    push_issue_tsv_rows(&mut out, static_digest);
    if let Some(visual) = visual_digest {
        push_issue_tsv_rows(&mut out, visual);
    }
    std::fs::write(args.run_root.join("audit-corpus-issues.tsv"), out)
        .map_err(|e| format!("write audit-corpus-issues.tsv: {e}"))
}

fn push_issue_tsv_rows(out: &mut String, digest: &ReportDigest) {
    for row in &digest.issue_rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            clean_tsv(&row.stage),
            clean_tsv(&row.status),
            clean_tsv(&row.family),
            clean_tsv(&row.severity),
            clean_tsv(&row.code),
            clean_tsv(&row.rel_path),
            clean_tsv(&row.detail),
            clean_tsv(&row.glb_png_path),
            clean_tsv(&row.source_png_path)
        ));
    }
}

fn clean_md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn clean_tsv(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

// ---- KAP corpus validation -------------------------------------------------

fn run_kap_cmd(args: Vec<String>) {
    let root = repo_root();
    let dir = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("pipeline/unpacked"));
    match run_kap(&dir) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("kap FAILED: {err}");
            std::process::exit(1);
        }
    }
}

/// Decode (KREncodedBlock) + parse (`kart_fmt::kap`) every `.kap` under `dir`,
/// checking byte round-trip, in-range markers, and finite/in-range waypoints.
fn run_kap(dir: &Path) -> Result<(), String> {
    println!("== kap corpus ==");
    let mut kaps = Vec::new();
    collect_ext(dir, "kap", &mut kaps);
    if kaps.is_empty() {
        println!("  SKIP ({} has no .kap files)", dir.display());
        return Ok(());
    }
    kaps.sort();

    let (mut decoded, mut parsed, mut roundtrip, mut markers_ok, mut sane, mut total_pts) =
        (0, 0, 0, 0, 0, 0usize);
    let mut fails: Vec<String> = Vec::new();
    for path in &kaps {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let clear = match kart_rho::encoded_block::decode_len_prefixed(&bytes) {
            Ok(c) => c,
            Err(e) => {
                fails.push(format!("{}: decode {e}", name(path)));
                continue;
            }
        };
        decoded += 1;
        let Some(kap) = kart_fmt::kap::parse(&clear) else {
            fails.push(format!("{}: parse rejected", name(path)));
            continue;
        };
        parsed += 1;
        if kap.to_bytes() == clear {
            roundtrip += 1;
        } else {
            fails.push(format!("{}: round-trip differs", name(path)));
        }
        if kap.markers_in_range() {
            markers_ok += 1;
        } else {
            fails.push(format!("{}: marker out of range", name(path)));
        }
        if kap
            .points
            .iter()
            .flatten()
            .all(|c| c.is_finite() && c.abs() < 100_000.0)
        {
            sane += 1;
        } else {
            fails.push(format!("{}: non-finite/huge coord", name(path)));
        }
        total_pts += kap.points.len();
    }

    let n = kaps.len();
    println!("  files={n} decoded={decoded} parsed={parsed} roundtrip={roundtrip} markers_in_range={markers_ok} coords_sane={sane}");
    println!("  total waypoints={total_pts}");
    if decoded == n && parsed == n && roundtrip == n && markers_ok == n && sane == n {
        println!("kap OK ({n}/{n})");
        Ok(())
    } else {
        for f in fails.iter().take(10) {
            eprintln!("  FAIL {f}");
        }
        Err(format!(
            "{}/{n} fully valid",
            roundtrip.min(parsed).min(markers_ok).min(sane)
        ))
    }
}

fn collect_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_ext(&p, ext, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}

fn name(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string()
}

// ---- KAP path → glTF (track geometry + AI paths as colored tubes) ----------

fn run_kap_gltf(args: Vec<String>) {
    if args.len() < 3 {
        eprintln!("usage: cargo xtask kap-gltf <track.1s> <out.glb> <kap>...");
        std::process::exit(2);
    }
    match build_kap_gltf(Path::new(&args[0]), Path::new(&args[1]), &args[2..]) {
        Ok(n) => println!("kap-gltf: track + {n} AI path(s) -> {}", args[1]),
        Err(e) => {
            eprintln!("kap-gltf FAILED: {e}");
            std::process::exit(1);
        }
    }
}

fn build_kap_gltf(track_path: &Path, out: &Path, kaps: &[String]) -> Result<usize, String> {
    use kart_gltf::{Alpha, Gltf, MaterialDef, NodeDef, Primitive};
    let data =
        std::fs::read(track_path).map_err(|e| format!("read {}: {e}", track_path.display()))?;
    let track = kart_fmt::kart_object::parse(&data).map_err(|e| format!("track parse: {e}"))?;

    let mut g = Gltf::new();
    let mut roots = Vec::new();
    let track_mat = g.add_material(MaterialDef {
        name: "track".into(),
        base_color_image: None,
        base_color: [0.55, 0.55, 0.60, 1.0],
        double_sided: true,
        alpha: Alpha::Opaque,
        unlit: false,
        extras: None,
    });
    let track_prim = track_world_prim(&track, track_mat);
    if track_prim.positions.is_empty() {
        return Err("track has no geometry".into());
    }
    let tmesh = g.add_mesh("track", vec![track_prim]);
    roots.push(g.add_node(NodeDef {
        name: Some("track".into()),
        mesh: Some(tmesh),
        ..Default::default()
    }));

    // bright, distinct colors per path
    const PALETTE: &[[f32; 4]] = &[
        [1.0, 0.1, 0.1, 1.0],
        [0.1, 1.0, 0.25, 1.0],
        [0.25, 0.4, 1.0, 1.0],
        [1.0, 0.85, 0.1, 1.0],
        [0.1, 1.0, 1.0, 1.0],
        [1.0, 0.2, 1.0, 1.0],
    ];
    let mut npaths = 0;
    for (i, kp) in kaps.iter().enumerate() {
        let bytes = std::fs::read(kp).map_err(|e| format!("read {kp}: {e}"))?;
        let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
            .map_err(|e| format!("{kp}: decode {e}"))?;
        let kap = kart_fmt::kap::parse(&clear).ok_or_else(|| format!("{kp}: not a KAP"))?;
        if kap.points.len() < 2 {
            continue;
        }
        let (lo, hi) = kap.bbox().unwrap();
        let diag =
            ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let width = (diag * 0.005).max(1.0);
        let color = PALETTE[i % PALETTE.len()];
        // colour via a 1x1 texture so the line shows in the simple renderer too
        // (kart-render only honours textures, not baseColorFactor).
        let tex = g.add_png(&solid_png(color));
        let mat = g.add_material(MaterialDef {
            name: format!("path{i}"),
            base_color_image: Some(tex),
            base_color: color,
            double_sided: true,
            alpha: Alpha::Opaque,
            unlit: true, // self-lit so the line reads regardless of normals
            extras: None,
        });
        // lift the path above the track surface (track is z-up) so it isn't
        // buried in the geometry.
        let lift = width * 4.0;
        let lifted: Vec<[f32; 3]> = kap
            .points
            .iter()
            .map(|p| [p[0], p[1], p[2] + lift])
            .collect();
        let (positions, normals, indices) = path_tube(&lifted, width);
        let prim = Primitive {
            positions,
            normals,
            uvs: vec![],
            indices,
            joints: None,
            weights: None,
            material: Some(mat),
        };
        let pmesh = g.add_mesh(&format!("aiPath{i}"), vec![prim]);
        roots.push(g.add_node(NodeDef {
            name: Some(format!("aiPath{i}")),
            mesh: Some(pmesh),
            ..Default::default()
        }));
        npaths += 1;
    }
    g.set_scene_roots(&roots);
    if let Some(p) = out.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    std::fs::write(out, g.into_glb()).map_err(|e| format!("write {}: {e}", out.display()))?;
    Ok(npaths)
}

/// Flatten a parsed track to one world-space, non-indexed triangle primitive
/// (raw track space — same frame the KAP waypoints live in, so the path overlays
/// correctly). Skips out-of-range indices.
fn track_world_prim(track: &kart_fmt::kart_object::Track, material: usize) -> kart_gltf::Primitive {
    let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for (i, m) in track.flatten() {
        let Some(geom) = &track.nodes[i].geom else {
            continue;
        };
        let nv = geom.vertices.len();
        for tri in geom_triangles(geom) {
            if tri.iter().any(|&v| v as usize >= nv) {
                continue;
            }
            let p: Vec<[f32; 3]> = tri
                .iter()
                .map(|&v| xform(&m, geom.vertices[v as usize]))
                .collect();
            let n = face_normal(p[0], p[1], p[2]);
            let base = positions.len() as u32;
            for pt in &p {
                positions.push(*pt);
                normals.push(n);
            }
            indices.extend([base, base + 1, base + 2]);
        }
    }
    kart_gltf::Primitive {
        positions,
        normals,
        uvs: vec![],
        indices,
        joints: None,
        weights: None,
        material: Some(material),
    }
}

fn geom_triangles(g: &kart_fmt::kart_object::Geom) -> Vec<[u32; 3]> {
    if g.strip {
        (0..g.indices.len().saturating_sub(2))
            .filter_map(|k| {
                let (a, b, c) = (g.indices[k], g.indices[k + 1], g.indices[k + 2]);
                if a == b || b == c || a == c {
                    None
                } else if k % 2 == 0 {
                    Some([a, b, c])
                } else {
                    Some([b, a, c])
                }
            })
            .collect()
    } else {
        g.indices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }
}

/// A square-cross-section tube along the polyline, so the path is visible from
/// any angle.
fn path_tube(pts: &[[f32; 3]], w: f32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let n = pts.len();
    let mut rings: Vec<[[f32; 3]; 4]> = Vec::with_capacity(n);
    for i in 0..n {
        let dir = sub(pts[(i + 1).min(n - 1)], pts[i.saturating_sub(1)]);
        let (r, u) = frame(normalize(dir));
        let p = pts[i];
        rings.push([
            add(p, add(scale(r, w), scale(u, w))),
            add(p, add(scale(r, -w), scale(u, w))),
            add(p, add(scale(r, -w), scale(u, -w))),
            add(p, add(scale(r, w), scale(u, -w))),
        ]);
    }
    let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n.saturating_sub(1) {
        for k in 0..4 {
            let k2 = (k + 1) % 4;
            let (a, b, c, d) = (rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]);
            let nrm = face_normal(a, b, c);
            let base = positions.len() as u32;
            for v in [a, b, c, a, c, d] {
                positions.push(v);
                normals.push(nrm);
            }
            indices.extend([base, base + 1, base + 2, base + 3, base + 4, base + 5]);
        }
    }
    (positions, normals, indices)
}

/// Encode a 1×1 RGBA PNG of one colour (linear [0,1] → 8-bit).
fn solid_png(rgba: [f32; 4]) -> Vec<u8> {
    let px = [
        (rgba[0] * 255.0) as u8,
        (rgba[1] * 255.0) as u8,
        (rgba[2] * 255.0) as u8,
        (rgba[3] * 255.0) as u8,
    ];
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&px).unwrap();
    }
    out
}

// ---- tiny vec3 helpers -----------------------------------------------------
fn xform(m: &[f32; 16], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
    ]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 1e-9 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}
fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize(cross(sub(b, a), sub(c, a)))
}
/// Two unit vectors perpendicular to `dir` (a stable frame for the tube cross-section).
fn frame(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let r#ref = if dir[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize(cross(dir, r#ref));
    let up = normalize(cross(right, dir));
    (right, up)
}

fn run_xval() {
    let root = repo_root();
    let mut fails = 0;
    fails += run_oracle(&root);
    println!();
    fails += run_csharp(&root);

    println!();
    if fails > 0 {
        eprintln!("cross-validation FAILED: {fails} mismatch(es)");
        std::process::exit(1);
    }
    println!("cross-validation OK");
}

// ---- local smoke -----------------------------------------------------------

fn run_smoke_cmd(args: Vec<String>) {
    if !args.is_empty() {
        eprintln!("smoke takes no arguments");
        print_usage();
        std::process::exit(2);
    }
    let root = repo_root();
    match run_smoke(&root) {
        Ok(()) => println!("smoke OK"),
        Err(err) => {
            eprintln!("smoke FAILED: {err}");
            std::process::exit(1);
        }
    }
}

fn run_smoke(root: &Path) -> Result<(), String> {
    println!("== local smoke ==");
    let client = root.join("modern/kart");
    if !client.is_dir() {
        println!("  SKIP (modern/kart corpus not present)");
        return Ok(());
    }

    smoke_pin(&client)?;
    smoke_pack(&client)?;
    smoke_pipeline(root)?;
    Ok(())
}

fn smoke_pin(client: &Path) -> Result<(), String> {
    let path = client.join("KartRider.pin");
    let pin = kart_rho::pin::read_pin_file(&path)?;
    expect_eq("pin.locale_id", pin.header.locale_id, 3002u16)?;
    expect_eq("pin.client_loc", pin.header.unk2, 47u16)?;
    expect_eq("pin.minor_version", pin.header.minor_version, 3502u16)?;
    expect_eq("pin.auth_methods", pin.auth_methods.len(), 4usize)?;
    let bml_names: Vec<&str> = pin.bml_objects.iter().map(|b| b.name.as_str()).collect();
    if bml_names != ["storage", "extra"] {
        return Err(format!("pin BML roots mismatch: {bml_names:?}"));
    }
    println!(
        "  PASS pin        locale={} client_loc={} version={} auth_methods={}",
        pin.header.locale_id,
        pin.header.unk2,
        pin.header.minor_version,
        pin.auth_methods.len()
    );
    Ok(())
}

fn smoke_pack(client: &Path) -> Result<(), String> {
    let path = client.join("Data/aaa.pk");
    let index = kart_rho::pack::read_pack_index(&path)?;
    let unique: std::collections::BTreeSet<&str> =
        index.mounts.iter().map(|m| m.file_name.as_str()).collect();
    let missing = index
        .mounts
        .iter()
        .filter(|m| !client.join("Data").join(&m.file_name).is_file())
        .count();
    expect_eq("aaa.mounts", index.mounts.len(), 1611usize)?;
    expect_eq("aaa.unique_rho", unique.len(), 1611usize)?;
    expect_eq("aaa.missing", missing, 0usize)?;
    println!(
        "  PASS aaa.pk     mounts={} unique_rho={} missing={}",
        index.mounts.len(),
        unique.len(),
        missing
    );
    Ok(())
}

fn smoke_pipeline(root: &Path) -> Result<(), String> {
    let out_dir = std::env::temp_dir().join(format!("kart-pipeline-smoke-{}", std::process::id()));
    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir).map_err(|e| format!("rm {}: {e}", out_dir.display()))?;
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = Command::new(cargo)
        .current_dir(root.join("rust"))
        .args([
            "run",
            "-q",
            "-p",
            "kart-pipeline",
            "--",
            "--client",
            "../modern/kart",
        ])
        .arg(&out_dir)
        .args(["--limit", "3", "--unpack-only", "--rebuild"])
        .output()
        .map_err(|e| format!("spawn kart-pipeline smoke: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&out_dir);
        return Err(format!(
            "kart-pipeline smoke exited {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        ));
    }
    let expected = "aaa.pk mounted_rho=1611 rho5=62 skipped_rho=0 missing_rho=0";
    if !stdout.contains(expected) {
        let _ = std::fs::remove_dir_all(&out_dir);
        return Err(format!("kart-pipeline smoke missing expected aaa.pk summary\nstdout:\n{stdout}\nstderr:\n{stderr}"));
    }
    for rel in [
        "unpacked/stuff2_",
        "unpacked/stuff2_/ethisItem",
        "unpacked/stuff2_/slotBG",
    ] {
        let path = out_dir.join(rel);
        if !path.is_dir() {
            let _ = std::fs::remove_dir_all(&out_dir);
            return Err(format!(
                "kart-pipeline smoke missing Launcher mount path {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                path.display()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&out_dir);
    println!("  PASS pipeline   --client --limit 3 --unpack-only");
    Ok(())
}

fn expect_eq<T>(label: &str, got: T, expected: T) -> Result<(), String>
where
    T: std::fmt::Display + PartialEq,
{
    if got == expected {
        Ok(())
    } else {
        Err(format!("{label}: got {got}, expected {expected}"))
    }
}

/// Compare two `k=v` stat lines on the keys the *reference* emitted; print a
/// per-case PASS/FAIL line. Returns 1 on mismatch.
fn diff(label: &str, ours: &str, reference: &str) -> usize {
    if reference.starts_with("ERROR") || reference.is_empty() {
        println!("  FAIL {label:<14} reference error: {reference}");
        return 1;
    }
    let a = glb::parse_kv(ours);
    let b = glb::parse_kv(reference);
    let mismatched: Vec<String> = b
        .iter()
        .filter(|(k, v)| a.get(*k) != Some(*v))
        .map(|(k, v)| {
            format!(
                "{k}: ours={} ref={v}",
                a.get(k).map(|s| s.as_str()).unwrap_or("∅")
            )
        })
        .collect();
    if mismatched.is_empty() {
        println!("  PASS {label:<14} {reference}");
        0
    } else {
        println!("  FAIL {label:<14} {}", mismatched.join("  "));
        println!("       ours: {ours}");
        1
    }
}

// ---- oracle (karttools) parser parity --------------------------------------

fn run_oracle(root: &Path) -> usize {
    println!("== oracle (karttools) parser parity ==");
    let py = root.join("oracle-local/.venv/bin/python");
    let oracle_dir = root.join("oracle-local");
    let samples = oracle_dir.join("samples");
    if !py.exists() || !oracle_dir.join("karttools").exists() || !samples.exists() {
        println!("  SKIP (oracle-local reference not present)");
        return 0;
    }

    let kart = samples.join("kart_blackLeopardXUN");
    let chr = samples.join("char_mrkart/character_mrkart.rho");
    let track = samples.join("track_beach_R03/track_beach_R03.rho");

    let mut fails = 0;
    // (label, oracle-kind, oracle-arg path, our stat closure)
    type Case<'a> = (&'a str, &'a str, PathBuf, Box<dyn Fn() -> String>);
    let cases: Vec<Case> = vec![
        (
            "model:kart",
            "model",
            kart.clone(),
            Box::new({
                let p = kart.join("model.1s");
                move || rust_model(&p)
            }),
        ),
        (
            "model:char",
            "model",
            chr.clone(),
            Box::new({
                let p = chr.join("model.1s");
                move || rust_model(&p)
            }),
        ),
        (
            "param",
            "param",
            kart.clone(),
            Box::new({
                let p = kart.join("param.xml");
                move || rust_param(&p)
            }),
        ),
        (
            "track",
            "track",
            track.clone(),
            Box::new({
                let t = track.join("track.1s");
                let s = track.join("skydome.1s");
                move || rust_track(&[&t, &s])
            }),
        ),
        (
            "char_anim",
            "char_anim",
            chr.join("f02.1s"),
            Box::new({
                let p = chr.join("f02.1s");
                move || rust_char_anim(&p)
            }),
        ),
        (
            "kart_anim",
            "kart_anim",
            kart.join("f01.1s"),
            Box::new({
                let p = kart.join("f01.1s");
                move || rust_kart_anim(&p)
            }),
        ),
    ];

    let driver = driver_path();
    for (label, kind, arg, ours_fn) in &cases {
        let reference = run_oracle_driver(&py, &oracle_dir, &driver, kind, arg);
        fails += diff(label, &ours_fn(), &reference);
    }
    fails
}

fn run_oracle_driver(
    py: &Path,
    oracle_dir: &Path,
    driver: &Path,
    kind: &str,
    arg: &Path,
) -> String {
    let out = Command::new(py)
        .arg(driver)
        .arg(kind)
        .arg(arg)
        .env("PYTHONPATH", oracle_dir)
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(e) => format!("ERROR spawn: {e}"),
    }
}

fn rust_model(file: &Path) -> String {
    let d = std::fs::read(file).expect("read model");
    let m = kart_fmt::raw_model::parse(&d);
    let (mut mn, mut v, mut u, mut f) = (0usize, 0usize, 0usize, 0usize);
    for n in m.all_nodes() {
        if n.is_mesh {
            mn += 1;
            v += n.vertices.len();
            u += n.uvs.len();
            f += n.vertex_indices.len() / 3;
        }
    }
    format!(
        "asset={} meshnodes={mn} verts={v} uvs={u} faces={f}",
        m.asset_type
    )
}

fn rust_param(file: &Path) -> String {
    use kart_fmt::param::Value;
    let d = std::fs::read(file).expect("read param");
    let ps = kart_fmt::param::parse(&d);
    // Match oracle_driver.py norm_value + sort-by-key + FNV-1a 64-bit.
    let mut pairs: Vec<(String, String)> = ps
        .iter()
        .map(|(k, val)| {
            let nv = match val {
                Value::Bool(b) => {
                    if *b {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                }
                Value::Int(i) => i.to_string(),
                Value::Float(f) => format!("{f:.6}"),
                Value::Str(s) => s.clone(),
            };
            (k.clone(), nv)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut h: u64 = 0xcbf29ce484222325;
    for (k, v) in &pairs {
        for b in format!("{k}={v}").bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    format!("count={} checksum={h:016x}", ps.len())
}

fn rust_track(files: &[&Path]) -> String {
    let (mut nodes, mut meshes, mut events) = (0usize, 0usize, 0i32);
    for f in files {
        let Ok(d) = std::fs::read(f) else { continue };
        let Ok(t) = kart_fmt::kart_object::parse(&d) else {
            continue;
        };
        events += t.event_count;
        if let Some(r) = t.root {
            dfs_track(&t, r, &mut nodes, &mut meshes);
        }
    }
    format!("nodes={nodes} meshes={meshes} events={events}")
}

fn dfs_track(t: &kart_fmt::kart_object::Track, i: usize, nodes: &mut usize, meshes: &mut usize) {
    *nodes += 1;
    if t.nodes[i].geom.is_some() {
        *meshes += 1;
    }
    for &c in &t.nodes[i].children {
        dfs_track(t, c, nodes, meshes);
    }
}

fn rust_char_anim(file: &Path) -> String {
    let d = std::fs::read(file).expect("read char anim");
    let c = kart_fmt::anim_clip::parse(&d, "x").expect("parse char anim");
    let kf: usize = c
        .tracks
        .iter()
        .map(|t| t.pos.len() + t.rot.len() + t.scale.len())
        .sum();
    format!("bones={} kf={kf} dur={}", c.tracks.len(), c.duration_ms)
}

fn rust_kart_anim(file: &Path) -> String {
    let d = std::fs::read(file).expect("read kart anim");
    let k = kart_fmt::kart_anim::parse(&d, "x").expect("parse kart anim");
    let kf: usize = k.channels.iter().map(|c| c.keyframes.len()).sum();
    let animated = k.channels.iter().filter(|c| c.keyframes.len() > 1).count();
    format!(
        "channels={} animated={animated} frames={} kf={kf} dur={}",
        k.channels.len(),
        k.num_frames,
        k.duration_ms
    )
}

// ---- C# (Kart1sToGltf) glTF-export parity ----------------------------------

fn run_csharp(root: &Path) -> usize {
    println!("== C# (Kart1sToGltf) glTF-export parity ==");
    let csproj = root.join("tools/Kart1sToGltf/Kart1sToGltf.csproj");
    let samples = root.join("oracle-local/samples");
    if which("dotnet").is_none() {
        println!("  SKIP (dotnet not on PATH)");
        return 0;
    }
    if !csproj.exists() {
        println!("  SKIP (tools/Kart1sToGltf not present)");
        return 0;
    }
    if !samples.exists() {
        println!("  SKIP (sample assets not present)");
        return 0;
    }

    // Build once (quiet); locate the produced dll.
    let build = Command::new("dotnet")
        .args(["build", "-c", "Release"])
        .arg(&csproj)
        .output();
    match build {
        Ok(o) if !o.status.success() => {
            println!(
                "  FAIL (dotnet build failed)\n{}",
                String::from_utf8_lossy(&o.stderr)
            );
            return 1;
        }
        Err(e) => {
            println!("  SKIP (dotnet build could not run: {e})");
            return 0;
        }
        _ => {}
    }
    let Some(dll) = find_dll(&root.join("tools/Kart1sToGltf/bin/Release")) else {
        println!("  FAIL (kart1s2gltf.dll not found after build)");
        return 1;
    };

    let out_dir = std::env::temp_dir().join("kart-xval");
    let _ = std::fs::create_dir_all(&out_dir);

    let mut fails = 0;
    // Compare GLB geometry for a kart model and a character model. Both serializers
    // apply the same dedup + degenerate-cull rules (the Rust writer replicates
    // SharpGLTF), so mesh/material/primitive/vertex/triangle counts match exactly.
    let cases: [(&str, PathBuf, &str); 2] = [
        (
            "kart",
            samples.join("kart_blackLeopardXUN/model.1s"),
            "kart",
        ),
        (
            "char",
            samples.join("char_mrkart/character_mrkart.rho/model.1s"),
            "char",
        ),
    ];
    for (label, model, name) in &cases {
        let cs_glb = out_dir.join(format!("cs_{label}.glb"));
        let conv = Command::new("dotnet")
            .arg(&dll)
            .arg(model)
            .arg("-o")
            .arg(&cs_glb)
            .output();
        match conv {
            Ok(o) if !o.status.success() => {
                println!(
                    "  FAIL {label:<14} C# convert failed: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                fails += 1;
                continue;
            }
            Err(e) => {
                println!("  FAIL {label:<14} C# convert could not run: {e}");
                fails += 1;
                continue;
            }
            _ => {}
        }
        let cs_bytes = std::fs::read(&cs_glb).unwrap_or_default();
        let Some(cs) = glb::stats(&cs_bytes) else {
            println!("  FAIL {label:<14} C# GLB unreadable");
            fails += 1;
            continue;
        };
        let (ours, ours_verts) = rust_glb(model, name);
        fails += diff(label, &ours, &cs.line());
        println!(
            "       (verts not compared — C#/SharpGLTF dedups: ours {ours_verts}, ref {})",
            cs.verts_note()
        );
    }
    fails
}

/// Convert a raw model to a Rust GLB and summarize its geometry; returns the
/// compared stat line plus the informational vertex note.
fn rust_glb(model: &Path, name: &str) -> (String, String) {
    let d = std::fs::read(model).expect("read model");
    let m = kart_fmt::raw_model::parse(&d);
    let tex_dir = model.parent().unwrap_or_else(|| Path::new("."));
    let gltf = kart_gltf::scene::convert_raw(&m, name, tex_dir, &[], &[]);
    let bytes = gltf.into_glb();
    match glb::stats(&bytes) {
        Some(s) => (s.line(), s.verts_note()),
        None => ("ERROR rust glb".to_string(), "verts=?".to_string()),
    }
}

// ---- small helpers ---------------------------------------------------------

fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(bin))
        .find(|p| p.is_file())
}

fn find_dll(dir: &Path) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "kart1s2gltf.dll") {
                return Some(p);
            }
        }
    }
    None
}
