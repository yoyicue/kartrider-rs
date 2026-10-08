//! Visual audit harness for `kart-pipeline` outputs.
//!
//! `kart-audit <pipeline-root> <audit-root>` consumes a pipeline output tree
//! (`glb-rs/` + `unpacked/`), samples GLB exports by asset family, extracts
//! structural GLB stats, renders the corresponding source `.1s` to PNG through
//! `kart-render`, and writes a static audit report.

use glam::{Mat4, Quat, Vec3};
use kart_render::texture::Image as RenderImage;
use kart_render::{RenderMesh, RenderVertex};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_PER_FAMILY: usize = 8;
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 640;

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    if args.rebuild && args.audit_root.exists() {
        std::fs::remove_dir_all(&args.audit_root)
            .map_err(|e| format!("rm {}: {e}", args.audit_root.display()))?;
    }
    std::fs::create_dir_all(args.audit_root.join("items"))
        .map_err(|e| format!("mkdir {}: {e}", args.audit_root.join("items").display()))?;

    let glb_root = infer_glb_root(&args.input_root)?;
    let source_root = args.input_root.join("unpacked");
    let source_root = source_root.is_dir().then_some(source_root);
    let registry = source_root.as_deref().and_then(load_track_registry);

    let mut candidates = walk_ext(&glb_root, "glb")?;
    candidates.sort();
    if candidates.is_empty() {
        return Err(format!("no .glb files under {}", glb_root.display()));
    }
    let include_list = args
        .include_list
        .as_ref()
        .map(|path| load_include_list(path))
        .transpose()?;

    let mut by_family: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for path in candidates {
        let rel = path.strip_prefix(&glb_root).unwrap_or(&path);
        let rel_string = rel_to_string(rel);
        if include_list
            .as_ref()
            .is_some_and(|wanted| !wanted.contains(&rel_string))
        {
            continue;
        }
        let family = family_of_rel(rel);
        if args
            .families
            .as_ref()
            .is_some_and(|wanted| !wanted.contains(&family))
        {
            continue;
        }
        by_family.entry(family).or_default().push(path);
    }

    let total_candidates: usize = by_family.values().map(Vec::len).sum();
    if total_candidates == 0 {
        return Err("family filter matched no GLB files".into());
    }

    let mut selected = Vec::new();
    for paths in by_family.values() {
        selected.extend(select_spread(paths, args.per_family));
    }
    selected.sort();
    if let Some(limit) = args.limit {
        selected.truncate(limit);
    }

    let mut family_summaries: BTreeMap<String, FamilySummary> = by_family
        .iter()
        .map(|(family, paths)| {
            (
                family.clone(),
                FamilySummary {
                    family: family.clone(),
                    candidates: paths.len(),
                    sampled: 0,
                    ok: 0,
                    warn: 0,
                    fail: 0,
                },
            )
        })
        .collect();

    let mut items = Vec::new();
    for (idx, glb_path) in selected.iter().enumerate() {
        let rel = glb_path.strip_prefix(&glb_root).unwrap_or(glb_path);
        let family = family_of_rel(rel);
        let id = format!("{idx:05}_{}", safe_id(rel));
        let glb_png_rel = format!("items/{id}.glb.png");
        let source_png_rel = format!("items/{id}.source.png");
        let json_rel = format!("items/{id}.json");
        let glb_png_path = args.audit_root.join(&glb_png_rel);
        let source_png_path = args.audit_root.join(&source_png_rel);
        let source_path = source_root
            .as_ref()
            .map(|root| root.join(rel).with_extension("1s"));

        let mut issues = Vec::new();
        let glb_doc = match read_glb(glb_path) {
            Ok(doc) => Some(doc),
            Err(e) => {
                issues.push(Issue::fail("glb_parse", e));
                None
            }
        };
        let glb = match glb_doc.as_ref() {
            Some(doc) => {
                let stats = glb_stats(doc);
                add_glb_issues(&stats, &mut issues);
                Some(stats)
            }
            None => None,
        };

        let mut glb_image = None;
        let mut glb_png_path_for_report = None;
        if args.glb_render {
            match glb_doc.as_ref() {
                Some(doc) => match render_glb_doc(doc, args.width, args.height, &glb_png_path) {
                    Ok(stats) => {
                        add_image_issues("glb", &stats, &mut issues);
                        glb_image = Some(stats);
                        glb_png_path_for_report = Some(glb_png_rel.clone());
                    }
                    Err(e) => issues.push(Issue::fail("glb_render", e)),
                },
                None => issues.push(Issue::fail(
                    "glb_render_missing_doc",
                    "GLB render skipped because GLB parse failed",
                )),
            }
        }

        let mut source_image = None;
        let mut source_png_path_for_report = None;
        if args.source_render {
            match source_path.as_ref() {
                Some(path) if path.is_file() => {
                    match render_source(
                        path,
                        source_root.as_ref().unwrap(),
                        registry.as_ref(),
                        &args,
                        &source_png_path,
                    ) {
                        Ok(stats) => {
                            add_image_issues("source", &stats, &mut issues);
                            source_image = Some(stats);
                            source_png_path_for_report = Some(source_png_rel.clone());
                        }
                        Err(e) => issues.push(Issue::fail("render_source", e)),
                    }
                }
                Some(path) => issues.push(Issue::fail(
                    "missing_source",
                    format!("missing {}", path.display()),
                )),
                None => issues.push(Issue::warn(
                    "missing_source_root",
                    "input has no unpacked/ source tree; skipped PNG render",
                )),
            }
        }
        add_render_comparison_issues(glb_image.as_ref(), source_image.as_ref(), &mut issues);

        // An empty GLB whose source `.1s` genuinely has no geometry (a dummy /
        // placeholder node tree) is a faithful conversion, not a defect — C#
        // emits the same empty GLB. Downgrade the empty-geometry issues to info
        // so these don't dominate the FAIL list.
        if glb.as_ref().is_some_and(|s| s.meshes == 0) {
            if let Some(sp) = source_path.as_ref().filter(|p| p.is_file()) {
                if source_is_geometryless(sp) {
                    const EMPTY: &[&str] = &[
                        "no_meshes",
                        "no_primitives",
                        "no_triangles",
                        "missing_bbox",
                        "glb_render",
                        "glb_render_missing_doc",
                    ];
                    for i in &mut issues {
                        if EMPTY.contains(&i.code.as_str()) {
                            i.severity = "info".into();
                        }
                    }
                    issues.push(Issue::info(
                        "empty_source",
                        "source .1s has no geometry (dummy/placeholder)",
                    ));
                }
            }
        }

        let status = status_for(&issues);
        if let Some(summary) = family_summaries.get_mut(&family) {
            summary.sampled += 1;
            match status {
                "OK" => summary.ok += 1,
                "WARN" => summary.warn += 1,
                "FAIL" => summary.fail += 1,
                _ => {}
            }
        }

        let item = AuditItem {
            id,
            family,
            rel_path: rel_to_string(rel),
            glb_path: path_to_string(glb_path),
            source_path: source_path
                .as_ref()
                .filter(|p| p.is_file())
                .map(|p| path_to_string(p)),
            png_path: glb_png_path_for_report
                .clone()
                .or_else(|| source_png_path_for_report.clone()),
            glb_png_path: glb_png_path_for_report,
            source_png_path: source_png_path_for_report,
            item_json_path: json_rel.clone(),
            status: status.to_string(),
            issues,
            glb,
            glb_image,
            source_image,
        };
        write_json(&args.audit_root.join(json_rel), &item)?;
        items.push(item);
    }

    let mut issue_counts = BTreeMap::new();
    let mut status_counts = StatusCounts::default();
    for item in &items {
        match item.status.as_str() {
            "OK" => status_counts.ok += 1,
            "WARN" => status_counts.warn += 1,
            "FAIL" => status_counts.fail += 1,
            _ => {}
        }
        for issue in &item.issues {
            *issue_counts.entry(issue.code.clone()).or_insert(0) += 1;
        }
    }

    let rendered = items.iter().filter(|i| i.png_path.is_some()).count();
    let glb_rendered = items.iter().filter(|i| i.glb_png_path.is_some()).count();
    let source_rendered = items.iter().filter(|i| i.source_png_path.is_some()).count();
    let (contact_sheet, contact_sheet_items) = if rendered > 0 && args.contact_sheet_limit > 0 {
        let count = rendered.min(args.contact_sheet_limit);
        (
            Some(write_contact_sheet(
                &args.audit_root,
                &items,
                args.contact_sheet_limit,
            )?),
            count,
        )
    } else {
        (None, 0)
    };
    let generated_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let report = AuditReport {
        tool: "kart-audit".into(),
        generated_unix,
        input_root: path_to_string(&args.input_root),
        glb_root: path_to_string(&glb_root),
        source_root: source_root.as_ref().map(|p| path_to_string(p)),
        audit_root: path_to_string(&args.audit_root),
        config: AuditConfig {
            per_family: args.per_family_label(),
            limit: args.limit,
            width: args.width,
            height: args.height,
            glb_render: args.glb_render,
            source_render: args.source_render,
            contact_sheet_limit: args.contact_sheet_limit,
            include_list: args.include_list.as_ref().map(|path| path_to_string(path)),
            families: args
                .families
                .as_ref()
                .map(|f| f.iter().cloned().collect::<Vec<_>>()),
        },
        summary: AuditSummary {
            candidates: total_candidates,
            sampled: items.len(),
            rendered,
            glb_rendered,
            source_rendered,
            ok: status_counts.ok,
            warn: status_counts.warn,
            fail: status_counts.fail,
            issue_counts,
            contact_sheet,
            contact_sheet_items,
        },
        families: family_summaries.into_values().collect(),
        items,
    };

    write_json(&args.audit_root.join("report.json"), &report)?;
    write_markdown(&args.audit_root.join("report.md"), &report)?;
    write_html(&args.audit_root.join("index.html"), &report)?;

    println!(
        "audit: candidates={} sampled={} rendered={} ok={} warn={} fail={} -> {}",
        report.summary.candidates,
        report.summary.sampled,
        report.summary.rendered,
        report.summary.ok,
        report.summary.warn,
        report.summary.fail,
        args.audit_root.display()
    );

    if args.fail_on_issue && report.summary.fail > 0 {
        return Err(format!("{} audited item(s) failed", report.summary.fail));
    }
    Ok(())
}

struct Args {
    input_root: PathBuf,
    audit_root: PathBuf,
    per_family: usize,
    limit: Option<usize>,
    width: u32,
    height: u32,
    glb_render: bool,
    source_render: bool,
    contact_sheet_limit: usize,
    rebuild: bool,
    fail_on_issue: bool,
    families: Option<BTreeSet<String>>,
    include_list: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut raw = std::env::args().skip(1).peekable();
        if raw.peek().is_none() {
            Self::usage();
            return Err("missing arguments".into());
        }

        let mut pos = Vec::new();
        let mut per_family = DEFAULT_PER_FAMILY;
        let mut limit = None;
        let mut width = DEFAULT_WIDTH;
        let mut height = DEFAULT_HEIGHT;
        let mut glb_render = true;
        let mut source_render = true;
        let mut contact_sheet_limit = 500usize;
        let mut rebuild = false;
        let mut fail_on_issue = false;
        let mut families = None;
        let mut include_list = None;

        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    Self::usage();
                    std::process::exit(0);
                }
                "--per-family" => per_family = parse_usize("--per-family", raw.next())?,
                "--all" => per_family = usize::MAX,
                "--limit" => limit = Some(parse_usize("--limit", raw.next())?),
                "--width" => width = parse_u32("--width", raw.next())?,
                "--height" => height = parse_u32("--height", raw.next())?,
                "--no-render" => {
                    glb_render = false;
                    source_render = false;
                }
                "--render" => {
                    glb_render = true;
                    source_render = true;
                }
                "--no-glb-render" => glb_render = false,
                "--glb-render" => glb_render = true,
                "--no-source-render" => source_render = false,
                "--source-render" => source_render = true,
                "--contact-sheet-limit" => {
                    contact_sheet_limit = parse_usize("--contact-sheet-limit", raw.next())?
                }
                "--no-contact-sheet" => contact_sheet_limit = 0,
                "--rebuild" => rebuild = true,
                "--fail-on-issue" => fail_on_issue = true,
                "--include-list" => {
                    include_list =
                        Some(PathBuf::from(raw.next().ok_or_else(|| {
                            "--include-list requires a path".to_string()
                        })?));
                }
                "--families" => {
                    let value = raw
                        .next()
                        .ok_or_else(|| "--families requires comma-separated names".to_string())?;
                    let set: BTreeSet<String> = value
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect();
                    if set.is_empty() {
                        return Err("--families matched no family names".into());
                    }
                    families = Some(set);
                }
                x if x.starts_with('-') => return Err(format!("unknown option: {x}")),
                x => pos.push(PathBuf::from(x)),
            }
        }

        if pos.len() != 2 {
            Self::usage();
            return Err("usage requires <pipeline-root> <audit-root>".into());
        }
        Ok(Self {
            input_root: pos.remove(0),
            audit_root: pos.remove(0),
            per_family,
            limit,
            width,
            height,
            glb_render,
            source_render,
            contact_sheet_limit,
            rebuild,
            fail_on_issue,
            families,
            include_list,
        })
    }

    fn usage() {
        eprintln!(
            "usage:\n  kart-audit <pipeline-root> <audit-root> [--per-family N|--all] [--limit N] [--families a,b] [--include-list paths.txt] [--width W] [--height H] [--no-render] [--no-glb-render] [--no-source-render] [--contact-sheet-limit N] [--no-contact-sheet] [--rebuild] [--fail-on-issue]\n\n<pipeline-root> is expected to contain glb-rs/ and, for source PNG render, unpacked/."
        );
    }

    fn per_family_label(&self) -> String {
        if self.per_family == usize::MAX {
            "all".into()
        } else {
            self.per_family.to_string()
        }
    }
}

fn parse_usize(label: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{label} requires a number"))?
        .parse()
        .map_err(|_| format!("bad {label} value"))
}

fn parse_u32(label: &str, value: Option<String>) -> Result<u32, String> {
    value
        .ok_or_else(|| format!("{label} requires a number"))?
        .parse()
        .map_err(|_| format!("bad {label} value"))
}

#[derive(Serialize)]
struct AuditReport {
    tool: String,
    generated_unix: u64,
    input_root: String,
    glb_root: String,
    source_root: Option<String>,
    audit_root: String,
    config: AuditConfig,
    summary: AuditSummary,
    families: Vec<FamilySummary>,
    items: Vec<AuditItem>,
}

#[derive(Serialize)]
struct AuditConfig {
    per_family: String,
    limit: Option<usize>,
    width: u32,
    height: u32,
    glb_render: bool,
    source_render: bool,
    contact_sheet_limit: usize,
    include_list: Option<String>,
    families: Option<Vec<String>>,
}

#[derive(Serialize)]
struct AuditSummary {
    candidates: usize,
    sampled: usize,
    rendered: usize,
    glb_rendered: usize,
    source_rendered: usize,
    ok: usize,
    warn: usize,
    fail: usize,
    issue_counts: BTreeMap<String, usize>,
    contact_sheet: Option<String>,
    contact_sheet_items: usize,
}

#[derive(Default)]
struct StatusCounts {
    ok: usize,
    warn: usize,
    fail: usize,
}

#[derive(Serialize)]
struct FamilySummary {
    family: String,
    candidates: usize,
    sampled: usize,
    ok: usize,
    warn: usize,
    fail: usize,
}

#[derive(Serialize)]
struct AuditItem {
    id: String,
    family: String,
    rel_path: String,
    glb_path: String,
    source_path: Option<String>,
    png_path: Option<String>,
    glb_png_path: Option<String>,
    source_png_path: Option<String>,
    item_json_path: String,
    status: String,
    issues: Vec<Issue>,
    glb: Option<GltfStats>,
    glb_image: Option<ImageStats>,
    source_image: Option<ImageStats>,
}

#[derive(Serialize, Clone)]
struct Issue {
    severity: String,
    code: String,
    detail: String,
}

impl Issue {
    fn fail(code: &str, detail: impl Into<String>) -> Self {
        Self {
            severity: "fail".into(),
            code: code.into(),
            detail: detail.into(),
        }
    }

    fn warn(code: &str, detail: impl Into<String>) -> Self {
        Self {
            severity: "warn".into(),
            code: code.into(),
            detail: detail.into(),
        }
    }

    fn info(code: &str, detail: impl Into<String>) -> Self {
        Self {
            severity: "info".into(),
            code: code.into(),
            detail: detail.into(),
        }
    }
}

#[derive(Serialize)]
struct GltfStats {
    scenes: usize,
    nodes: usize,
    meshes: usize,
    primitives: usize,
    triangles: usize,
    materials: usize,
    images: usize,
    textures: usize,
    skins: usize,
    animations: usize,
    bbox: Option<BBox>,
}

#[derive(Serialize)]
struct BBox {
    min: [f64; 3],
    max: [f64; 3],
    extent: [f64; 3],
    diagonal: f64,
}

#[derive(Serialize, Clone)]
struct ImageStats {
    width: u32,
    height: u32,
    foreground_ratio: f64,
    transparent_ratio: f64,
    luma_mean: f64,
    luma_stddev: f64,
}

fn infer_glb_root(input: &Path) -> Result<PathBuf, String> {
    let nested = input.join("glb-rs");
    if nested.is_dir() {
        Ok(nested)
    } else if input.is_dir() {
        Ok(input.to_path_buf())
    } else {
        Err(format!("input root does not exist: {}", input.display()))
    }
}

fn walk_ext(root: &Path, ext: &str) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = std::fs::read_dir(&dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
                out.push(path);
            }
        }
    }
    Ok(out)
}

fn load_include_list(path: &Path) -> Result<BTreeSet<String>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.replace('\\', "/"))
        .collect())
}

fn select_spread(paths: &[PathBuf], per_family: usize) -> Vec<PathBuf> {
    if per_family == usize::MAX || paths.len() <= per_family {
        return paths.to_vec();
    }
    if per_family == 0 {
        return Vec::new();
    }
    if per_family == 1 {
        return vec![paths[paths.len() / 2].clone()];
    }
    (0..per_family)
        .map(|i| {
            let idx = i * (paths.len() - 1) / (per_family - 1);
            paths[idx].clone()
        })
        .collect()
}

fn family_of_rel(rel: &Path) -> String {
    let first = rel
        .components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("other");
    let stem = first
        .strip_suffix(".rho5")
        .or_else(|| first.strip_suffix(".rho"))
        .unwrap_or(first);
    stem.trim_end_matches('_')
        .split('_')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("other")
        .to_string()
}

fn safe_id(rel: &Path) -> String {
    rel_to_string(rel)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

struct GlbDoc {
    json: Value,
    bin: Vec<u8>,
}

fn read_glb(path: &Path) -> Result<GlbDoc, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 20 {
        return Err("truncated GLB".into());
    }
    if &bytes[0..4] != b"glTF" {
        return Err("bad GLB magic".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 2 {
        return Err(format!("unsupported GLB version {version}"));
    }
    let declared_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if declared_len != bytes.len() {
        return Err(format!(
            "GLB length mismatch: header={declared_len} actual={}",
            bytes.len()
        ));
    }

    let mut off = 12usize;
    let mut json = None;
    let mut bin = None;
    while off + 8 <= bytes.len() {
        let chunk_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let chunk_type = &bytes[off + 4..off + 8];
        off += 8;
        if off + chunk_len > bytes.len() {
            return Err("chunk overruns GLB length".into());
        }
        if chunk_type == b"JSON" {
            let chunk = &bytes[off..off + chunk_len];
            let text = std::str::from_utf8(chunk)
                .map_err(|e| format!("JSON chunk is not utf8: {e}"))?
                .trim_end_matches([' ', '\0']);
            json = Some(
                serde_json::from_str::<Value>(text).map_err(|e| format!("parse glTF JSON: {e}"))?,
            );
        } else if chunk_type == b"BIN\0" {
            bin = Some(bytes[off..off + chunk_len].to_vec());
        }
        off += chunk_len;
    }
    let json = json.ok_or("missing JSON chunk")?;
    Ok(GlbDoc {
        json,
        bin: bin.unwrap_or_default(),
    })
}

fn glb_stats(doc: &GlbDoc) -> GltfStats {
    let json = &doc.json;
    let scenes = array_len(json, "scenes");
    let nodes = array_len(json, "nodes");
    let meshes_count = array_len(json, "meshes");
    let materials = array_len(json, "materials");
    let images = array_len(json, "images");
    let textures = array_len(json, "textures");
    let skins = array_len(json, "skins");
    let animations = array_len(json, "animations");

    let accessors = json.get("accessors").and_then(Value::as_array);
    let mut primitives = 0usize;
    let mut triangles = 0usize;
    let mut bbox_min = [f64::INFINITY; 3];
    let mut bbox_max = [f64::NEG_INFINITY; 3];
    let mut have_bbox = false;

    if let Some(meshes) = json.get("meshes").and_then(Value::as_array) {
        for mesh in meshes {
            if let Some(prims) = mesh.get("primitives").and_then(Value::as_array) {
                for prim in prims {
                    primitives += 1;
                    if let Some(idx) = prim.get("indices").and_then(Value::as_u64) {
                        triangles += accessor_count(accessors, idx as usize).unwrap_or(0) / 3;
                    } else if let Some(pos_idx) = position_accessor(prim) {
                        triangles += accessor_count(accessors, pos_idx).unwrap_or(0) / 3;
                    }
                    if let Some(pos_idx) = position_accessor(prim) {
                        if let Some((mn, mx)) = accessor_bbox(accessors, pos_idx) {
                            merge_bbox(&mut bbox_min, &mut bbox_max, mn, mx);
                            have_bbox = true;
                        }
                    }
                }
            }
        }
    }

    let bbox = have_bbox.then(|| {
        let extent = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        ];
        let diagonal =
            (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();
        BBox {
            min: bbox_min,
            max: bbox_max,
            extent,
            diagonal,
        }
    });

    GltfStats {
        scenes,
        nodes,
        meshes: meshes_count,
        primitives,
        triangles,
        materials,
        images,
        textures,
        skins,
        animations,
        bbox,
    }
}

fn array_len(json: &Value, key: &str) -> usize {
    json.get(key).and_then(Value::as_array).map_or(0, Vec::len)
}

fn position_accessor(prim: &Value) -> Option<usize> {
    prim.get("attributes")?
        .get("POSITION")?
        .as_u64()
        .map(|v| v as usize)
}

fn accessor_count(accessors: Option<&Vec<Value>>, idx: usize) -> Option<usize> {
    accessors?
        .get(idx)?
        .get("count")?
        .as_u64()
        .map(|v| v as usize)
}

fn accessor_bbox(accessors: Option<&Vec<Value>>, idx: usize) -> Option<([f64; 3], [f64; 3])> {
    let acc = accessors?.get(idx)?;
    let min = acc.get("min")?.as_array()?;
    let max = acc.get("max")?.as_array()?;
    if min.len() != 3 || max.len() != 3 {
        return None;
    }
    Some((
        [min[0].as_f64()?, min[1].as_f64()?, min[2].as_f64()?],
        [max[0].as_f64()?, max[1].as_f64()?, max[2].as_f64()?],
    ))
}

fn merge_bbox(global_min: &mut [f64; 3], global_max: &mut [f64; 3], min: [f64; 3], max: [f64; 3]) {
    for i in 0..3 {
        global_min[i] = global_min[i].min(min[i]);
        global_max[i] = global_max[i].max(max[i]);
    }
}

fn add_glb_issues(stats: &GltfStats, issues: &mut Vec<Issue>) {
    if stats.meshes == 0 {
        issues.push(Issue::fail("no_meshes", "GLB contains no meshes"));
    }
    if stats.primitives == 0 {
        issues.push(Issue::fail(
            "no_primitives",
            "GLB contains no mesh primitives",
        ));
    }
    if stats.triangles == 0 {
        issues.push(Issue::fail("no_triangles", "GLB contains no triangles"));
    }
    match &stats.bbox {
        Some(b) if !b.diagonal.is_finite() => {
            issues.push(Issue::fail("bbox_non_finite", "GLB bbox is not finite"));
        }
        Some(b) if b.diagonal > 1_000_000.0 => {
            issues.push(Issue::warn(
                "bbox_huge",
                format!("bbox diagonal is {:.3}", b.diagonal),
            ));
        }
        Some(_) => {}
        None => issues.push(Issue::warn(
            "missing_bbox",
            "POSITION accessors have no min/max",
        )),
    }
}

fn render_glb_doc(
    doc: &GlbDoc,
    width: u32,
    height: u32,
    png_path: &Path,
) -> Result<ImageStats, String> {
    let meshes = glb_render_meshes(doc)?;
    let frame = robust_frame(&meshes);
    let png = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match frame {
        Some((mn, mx)) => kart_render::render_meshes_y_up_png_framed(meshes, mn, mx, width, height),
        None => kart_render::render_meshes_y_up_png(meshes, width, height),
    }))
    .unwrap_or_else(|_| Err("panic while rendering GLB".into()))?;
    if let Some(parent) = png_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(png_path, &png).map_err(|e| format!("write {}: {e}", png_path.display()))?;
    image_stats(&png)
}

/// A robust per-axis [1%, 99%] framing AABB, returned **only** when the full
/// extent is dominated by outliers (full diagonal > 4× the trimmed diagonal) —
/// e.g. a track with one stray vertex at 1e8. Otherwise `None`, so normal assets
/// keep framing on their true bbox (no render change for the ~15k clean assets).
fn robust_frame(meshes: &[RenderMesh]) -> Option<([f32; 3], [f32; 3])> {
    let mut axes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for m in meshes {
        for v in &m.vertices {
            for k in 0..3 {
                if v.pos[k].is_finite() {
                    axes[k].push(v.pos[k]);
                }
            }
        }
    }
    if axes[0].len() < 16 {
        return None;
    }
    let (mut full_min, mut full_max) = ([0f32; 3], [0f32; 3]);
    let (mut rmin, mut rmax) = ([0f32; 3], [0f32; 3]);
    for k in 0..3 {
        let a = &mut axes[k];
        a.sort_by(f32::total_cmp);
        full_min[k] = a[0];
        full_max[k] = a[a.len() - 1];
        rmin[k] = a[((a.len() as f32 * 0.01) as usize).min(a.len() - 1)];
        rmax[k] = a[((a.len() as f32 * 0.99) as usize).min(a.len() - 1)];
    }
    let diag = |mn: [f32; 3], mx: [f32; 3]| {
        ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt()
    };
    let (fd, rd) = (diag(full_min, full_max), diag(rmin, rmax).max(1e-3));
    (fd > rd * 4.0).then_some((rmin, rmax))
}

fn glb_render_meshes(doc: &GlbDoc) -> Result<Vec<RenderMesh>, String> {
    let nodes = doc
        .json
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or("GLB has no nodes")?;
    let meshes_json = doc
        .json
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or("GLB has no meshes")?;
    let roots = scene_roots(&doc.json, nodes);
    if roots.is_empty() {
        return Err("GLB scene has no root nodes".into());
    }

    let mut out = Vec::new();
    let mut image_cache: HashMap<usize, RenderImage> = HashMap::new();
    let mut stack: Vec<(usize, Mat4)> =
        roots.into_iter().map(|idx| (idx, Mat4::IDENTITY)).collect();
    while let Some((node_idx, parent)) = stack.pop() {
        let node = nodes
            .get(node_idx)
            .ok_or_else(|| format!("node index out of range: {node_idx}"))?;
        let world = parent * node_local_matrix(node);
        if let Some(mesh_idx) = node.get("mesh").and_then(Value::as_u64).map(|v| v as usize) {
            let mesh = meshes_json
                .get(mesh_idx)
                .ok_or_else(|| format!("mesh index out of range: {mesh_idx}"))?;
            emit_glb_mesh(doc, mesh, world, &mut image_cache, &mut out)?;
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for child in children.iter().filter_map(Value::as_u64) {
                stack.push((child as usize, world));
            }
        }
    }
    if out.is_empty() {
        return Err("GLB has no renderable triangles".into());
    }
    Ok(out)
}

fn scene_roots(json: &Value, nodes: &[Value]) -> Vec<usize> {
    let scene_idx = json.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize;
    if let Some(roots) = json
        .get("scenes")
        .and_then(Value::as_array)
        .and_then(|scenes| scenes.get(scene_idx))
        .and_then(|scene| scene.get("nodes"))
        .and_then(Value::as_array)
    {
        return roots
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
    }

    let mut is_child = vec![false; nodes.len()];
    for node in nodes {
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for child in children.iter().filter_map(Value::as_u64) {
                if let Some(slot) = is_child.get_mut(child as usize) {
                    *slot = true;
                }
            }
        }
    }
    is_child
        .iter()
        .enumerate()
        .filter_map(|(idx, child)| (!child).then_some(idx))
        .collect()
}

fn node_local_matrix(node: &Value) -> Mat4 {
    if let Some(matrix) = node.get("matrix").and_then(Value::as_array) {
        if matrix.len() == 16 {
            let mut m = [0.0f32; 16];
            for (i, v) in matrix.iter().enumerate() {
                m[i] = v.as_f64().unwrap_or(if i % 5 == 0 { 1.0 } else { 0.0 }) as f32;
            }
            return Mat4::from_cols_array(&m);
        }
    }
    let t = read_vec3_json(node.get("translation"), [0.0, 0.0, 0.0]);
    let s = read_vec3_json(node.get("scale"), [1.0, 1.0, 1.0]);
    let r = read_vec4_json(node.get("rotation"), [0.0, 0.0, 0.0, 1.0]);
    Mat4::from_scale_rotation_translation(
        Vec3::from_array(s),
        Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
        Vec3::from_array(t),
    )
}

fn read_vec3_json(value: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    let Some(values) = value.and_then(Value::as_array) else {
        return default;
    };
    if values.len() != 3 {
        return default;
    }
    [
        values[0].as_f64().unwrap_or(default[0] as f64) as f32,
        values[1].as_f64().unwrap_or(default[1] as f64) as f32,
        values[2].as_f64().unwrap_or(default[2] as f64) as f32,
    ]
}

fn read_vec4_json(value: Option<&Value>, default: [f32; 4]) -> [f32; 4] {
    let Some(values) = value.and_then(Value::as_array) else {
        return default;
    };
    if values.len() != 4 {
        return default;
    }
    [
        values[0].as_f64().unwrap_or(default[0] as f64) as f32,
        values[1].as_f64().unwrap_or(default[1] as f64) as f32,
        values[2].as_f64().unwrap_or(default[2] as f64) as f32,
        values[3].as_f64().unwrap_or(default[3] as f64) as f32,
    ]
}

fn emit_glb_mesh(
    doc: &GlbDoc,
    mesh: &Value,
    world: Mat4,
    image_cache: &mut HashMap<usize, RenderImage>,
    out: &mut Vec<RenderMesh>,
) -> Result<(), String> {
    let Some(primitives) = mesh.get("primitives").and_then(Value::as_array) else {
        return Ok(());
    };
    for prim in primitives {
        let attrs = prim
            .get("attributes")
            .ok_or("primitive has no attributes")?;
        let pos_idx = attrs
            .get("POSITION")
            .and_then(Value::as_u64)
            .ok_or("primitive has no POSITION accessor")? as usize;
        let positions = read_vec3_accessor(doc, pos_idx)?;
        let normals = attrs
            .get("NORMAL")
            .and_then(Value::as_u64)
            .map(|idx| read_vec3_accessor(doc, idx as usize))
            .transpose()?
            .unwrap_or_default();
        let uvs = attrs
            .get("TEXCOORD_0")
            .and_then(Value::as_u64)
            .map(|idx| read_vec2_accessor(doc, idx as usize))
            .transpose()?
            .unwrap_or_default();
        let indices = match prim.get("indices").and_then(Value::as_u64) {
            Some(idx) => read_indices_accessor(doc, idx as usize)?,
            None => (0..positions.len() as u32).collect(),
        };
        let material = material_render_info(doc, prim)?;
        if material.skip {
            continue;
        }
        let image = match material.image_idx {
            Some(image_idx) => {
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    image_cache.entry(image_idx)
                {
                    entry.insert(decode_glb_image(doc, image_idx)?);
                }
                image_cache.get(&image_idx).cloned()
            }
            None => None,
        };

        let normal_matrix = world;
        let mut verts = Vec::new();
        for tri in indices.chunks_exact(3) {
            let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            if ids.iter().any(|&idx| idx >= positions.len()) {
                continue;
            }
            let wp = [
                world.transform_point3(Vec3::from(positions[ids[0]])),
                world.transform_point3(Vec3::from(positions[ids[1]])),
                world.transform_point3(Vec3::from(positions[ids[2]])),
            ];
            let face_n = (wp[1] - wp[0]).cross(wp[2] - wp[0]).normalize_or_zero();
            for k in 0..3 {
                let id = ids[k];
                let normal = normals
                    .get(id)
                    .map(|n| {
                        normal_matrix
                            .transform_vector3(Vec3::from(*n))
                            .normalize_or_zero()
                    })
                    .unwrap_or(face_n);
                let uv = uvs.get(id).copied().unwrap_or([0.0, 0.0]);
                verts.push(RenderVertex {
                    pos: wp[k].into(),
                    normal: normal.into(),
                    uv,
                });
            }
        }
        if !verts.is_empty() {
            out.push(RenderMesh {
                vertices: verts,
                image,
                alpha_cutoff: material.alpha_cutoff,
            });
        }
    }
    Ok(())
}

struct MaterialRenderInfo {
    image_idx: Option<usize>,
    alpha_cutoff: Option<f32>,
    skip: bool,
}

fn material_render_info(doc: &GlbDoc, prim: &Value) -> Result<MaterialRenderInfo, String> {
    let Some(material_idx) = prim
        .get("material")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    else {
        return Ok(MaterialRenderInfo {
            image_idx: None,
            alpha_cutoff: None,
            skip: false,
        });
    };
    let materials = doc
        .json
        .get("materials")
        .and_then(Value::as_array)
        .ok_or("material reference but GLB has no materials")?;
    let material = materials
        .get(material_idx)
        .ok_or_else(|| format!("material index out of range: {material_idx}"))?;
    let alpha_mode = material
        .get("alphaMode")
        .and_then(Value::as_str)
        .unwrap_or("OPAQUE");
    let alpha_cutoff = match alpha_mode {
        "MASK" => Some(
            material
                .get("alphaCutoff")
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32,
        ),
        "BLEND" => Some(0.05),
        _ => None,
    };
    let texture_idx = material
        .get("pbrMetallicRoughness")
        .and_then(|pbr| pbr.get("baseColorTexture"))
        .and_then(|tex| tex.get("index"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let image_idx = match texture_idx {
        Some(texture_idx) => {
            let textures = doc
                .json
                .get("textures")
                .and_then(Value::as_array)
                .ok_or("texture reference but GLB has no textures")?;
            let texture = textures
                .get(texture_idx)
                .ok_or_else(|| format!("texture index out of range: {texture_idx}"))?;
            texture
                .get("source")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        }
        None => None,
    };
    let skip = alpha_cutoff.is_some() && image_idx.is_none();
    Ok(MaterialRenderInfo {
        image_idx,
        alpha_cutoff,
        skip,
    })
}

fn decode_glb_image(doc: &GlbDoc, image_idx: usize) -> Result<RenderImage, String> {
    let images = doc
        .json
        .get("images")
        .and_then(Value::as_array)
        .ok_or("image reference but GLB has no images")?;
    let image = images
        .get(image_idx)
        .ok_or_else(|| format!("image index out of range: {image_idx}"))?;
    let mime = image
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    if mime != "image/png" {
        return Err(format!("unsupported embedded image mime type: {mime}"));
    }
    let view_idx = image
        .get("bufferView")
        .and_then(Value::as_u64)
        .ok_or("embedded image has no bufferView")? as usize;
    decode_render_image(buffer_view_slice(doc, view_idx)?)
}

fn decode_render_image(bytes: &[u8]) -> Result<RenderImage, String> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("embedded png header: {e}"))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("embedded png frame: {e}"))?;
    let rgba = rgba_pixels(&buf[..info.buffer_size()], info.color_type)?;
    Ok(RenderImage {
        width: info.width,
        height: info.height,
        rgba,
    })
}

fn read_vec3_accessor(doc: &GlbDoc, idx: usize) -> Result<Vec<[f32; 3]>, String> {
    read_f32_vec_accessor(doc, idx, "VEC3", 3).map(|rows| {
        rows.into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    })
}

fn read_vec2_accessor(doc: &GlbDoc, idx: usize) -> Result<Vec<[f32; 2]>, String> {
    read_f32_vec_accessor(doc, idx, "VEC2", 2)
        .map(|rows| rows.into_iter().map(|row| [row[0], row[1]]).collect())
}

fn read_f32_vec_accessor(
    doc: &GlbDoc,
    idx: usize,
    expected_type: &str,
    comps: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let acc = accessor_value(doc, idx)?;
    let component_type = acc
        .get("componentType")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("accessor {idx} has no componentType"))?;
    if component_type != 5126 {
        return Err(format!(
            "accessor {idx} expected f32 componentType 5126, got {component_type}"
        ));
    }
    let type_name = acc.get("type").and_then(Value::as_str).unwrap_or("");
    if type_name != expected_type {
        return Err(format!(
            "accessor {idx} expected type {expected_type}, got {type_name}"
        ));
    }
    let count = acc
        .get("count")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("accessor {idx} has no count"))? as usize;
    let (view, data) = accessor_data(doc, idx)?;
    let stride = view
        .get("byteStride")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(comps * 4);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * stride;
        if off + comps * 4 > data.len() {
            return Err(format!("accessor {idx} overruns bufferView"));
        }
        let mut row = Vec::with_capacity(comps);
        for c in 0..comps {
            let p = off + c * 4;
            row.push(f32::from_le_bytes(data[p..p + 4].try_into().unwrap()));
        }
        out.push(row);
    }
    Ok(out)
}

fn read_indices_accessor(doc: &GlbDoc, idx: usize) -> Result<Vec<u32>, String> {
    let acc = accessor_value(doc, idx)?;
    let component_type = acc
        .get("componentType")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("accessor {idx} has no componentType"))?;
    let type_name = acc.get("type").and_then(Value::as_str).unwrap_or("");
    if type_name != "SCALAR" {
        return Err(format!(
            "index accessor {idx} expected SCALAR, got {type_name}"
        ));
    }
    let count = acc
        .get("count")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("accessor {idx} has no count"))? as usize;
    let component_size = match component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        other => return Err(format!("unsupported index componentType {other}")),
    };
    let (view, data) = accessor_data(doc, idx)?;
    let stride = view
        .get("byteStride")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(component_size);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * stride;
        if off + component_size > data.len() {
            return Err(format!("index accessor {idx} overruns bufferView"));
        }
        let value = match component_type {
            5121 => data[off] as u32,
            5123 => u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as u32,
            5125 => u32::from_le_bytes(data[off..off + 4].try_into().unwrap()),
            _ => unreachable!(),
        };
        out.push(value);
    }
    Ok(out)
}

fn accessor_value(doc: &GlbDoc, idx: usize) -> Result<&Value, String> {
    doc.json
        .get("accessors")
        .and_then(Value::as_array)
        .and_then(|accessors| accessors.get(idx))
        .ok_or_else(|| format!("accessor index out of range: {idx}"))
}

fn accessor_data(doc: &GlbDoc, idx: usize) -> Result<(&Value, &[u8]), String> {
    let acc = accessor_value(doc, idx)?;
    let view_idx = acc
        .get("bufferView")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("accessor {idx} has no bufferView"))? as usize;
    let view = buffer_view_value(doc, view_idx)?;
    let view_data = buffer_view_slice(doc, view_idx)?;
    let acc_offset = acc.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
    if acc_offset > view_data.len() {
        return Err(format!("accessor {idx} byteOffset overruns bufferView"));
    }
    Ok((view, &view_data[acc_offset..]))
}

fn buffer_view_value(doc: &GlbDoc, idx: usize) -> Result<&Value, String> {
    doc.json
        .get("bufferViews")
        .and_then(Value::as_array)
        .and_then(|views| views.get(idx))
        .ok_or_else(|| format!("bufferView index out of range: {idx}"))
}

fn buffer_view_slice(doc: &GlbDoc, idx: usize) -> Result<&[u8], String> {
    let view = buffer_view_value(doc, idx)?;
    let offset = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let length = view
        .get("byteLength")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("bufferView {idx} has no byteLength"))? as usize;
    if offset + length > doc.bin.len() {
        return Err(format!(
            "bufferView {idx} overruns BIN chunk: offset={offset} length={length} bin={}",
            doc.bin.len()
        ));
    }
    Ok(&doc.bin[offset..offset + length])
}

fn render_source(
    source_path: &Path,
    source_root: &Path,
    registry: Option<&HashMap<String, Vec<String>>>,
    args: &Args,
    png_path: &Path,
) -> Result<ImageStats, String> {
    let data =
        std::fs::read(source_path).map_err(|e| format!("read {}: {e}", source_path.display()))?;
    let png = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if kart_fmt::raw_model::is_raw_model(&data) {
            kart_render::render_model_png(source_path, args.width, args.height)
        } else if data.len() >= 2 && data[0] == 0xAA && data[1] == 0x47 {
            let tex_dirs = texture_dirs_for(source_root, source_path, registry);
            kart_render::render_track_png_with_tex_dirs(
                source_path,
                &tex_dirs,
                args.width,
                args.height,
            )
        } else {
            Err("unsupported source .1s layout".into())
        }
    }))
    .unwrap_or_else(|_| Err("panic while rendering source .1s".into()))?;

    if let Some(parent) = png_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(png_path, &png).map_err(|e| format!("write {}: {e}", png_path.display()))?;
    image_stats(&png)
}

fn image_stats(png_bytes: &[u8]) -> Result<ImageStats, String> {
    let decoder = png::Decoder::new(Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png frame: {e}"))?;
    let data = &buf[..info.buffer_size()];
    let rgba = rgba_pixels(data, info.color_type)?;
    let total = (info.width as usize) * (info.height as usize);
    if total == 0 {
        return Err("empty PNG".into());
    }
    let bg = [rgba[0], rgba[1], rgba[2], rgba[3]];
    let mut foreground = 0usize;
    let mut transparent = 0usize;
    let mut lumas = Vec::with_capacity(total);
    for px in rgba.chunks_exact(4) {
        let diff = (px[0] as i16 - bg[0] as i16).abs()
            + (px[1] as i16 - bg[1] as i16).abs()
            + (px[2] as i16 - bg[2] as i16).abs()
            + (px[3] as i16 - bg[3] as i16).abs();
        if diff > 24 {
            foreground += 1;
        }
        if px[3] < 16 {
            transparent += 1;
        }
        lumas.push(0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64);
    }
    let luma_mean = lumas.iter().sum::<f64>() / total as f64;
    let variance = lumas
        .iter()
        .map(|v| {
            let d = v - luma_mean;
            d * d
        })
        .sum::<f64>()
        / total as f64;
    Ok(ImageStats {
        width: info.width,
        height: info.height,
        foreground_ratio: foreground as f64 / total as f64,
        transparent_ratio: transparent as f64 / total as f64,
        luma_mean,
        luma_stddev: variance.sqrt(),
    })
}

fn rgba_pixels(data: &[u8], color: png::ColorType) -> Result<Vec<u8>, String> {
    match color {
        png::ColorType::Rgba => Ok(data.to_vec()),
        png::ColorType::Rgb => {
            let mut out = Vec::with_capacity(data.len() / 3 * 4);
            for px in data.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            Ok(out)
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(data.len() * 4);
            for &g in data {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            Ok(out)
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(data.len() / 2 * 4);
            for px in data.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
            Ok(out)
        }
        other => Err(format!("unsupported PNG color type {other:?}")),
    }
}

fn add_image_issues(prefix: &str, stats: &ImageStats, issues: &mut Vec<Issue>) {
    if stats.foreground_ratio < 0.002 {
        issues.push(Issue::fail(
            &format!("{prefix}_blank_render"),
            format!("foreground ratio {:.4}", stats.foreground_ratio),
        ));
    } else if stats.foreground_ratio < 0.01 {
        issues.push(Issue::warn(
            &format!("{prefix}_tiny_render"),
            format!("foreground ratio {:.4}", stats.foreground_ratio),
        ));
    }
    if stats.luma_stddev < 1.0 {
        issues.push(Issue::warn(
            &format!("{prefix}_flat_render"),
            format!("luma stddev {:.3}", stats.luma_stddev),
        ));
    }
}

fn add_render_comparison_issues(
    glb: Option<&ImageStats>,
    source: Option<&ImageStats>,
    issues: &mut Vec<Issue>,
) {
    let (Some(glb), Some(source)) = (glb, source) else {
        return;
    };
    if source.foreground_ratio >= 0.03 && glb.foreground_ratio < source.foreground_ratio * 0.25 {
        issues.push(Issue::warn(
            "glb_source_foreground_mismatch",
            format!(
                "glb foreground {:.4} vs source {:.4}",
                glb.foreground_ratio, source.foreground_ratio
            ),
        ));
    }
    if source.luma_stddev >= 8.0 && glb.luma_stddev < source.luma_stddev * 0.25 {
        issues.push(Issue::warn(
            "glb_source_luma_mismatch",
            format!(
                "glb luma stddev {:.3} vs source {:.3}",
                glb.luma_stddev, source.luma_stddev
            ),
        ));
    }
}

fn status_for(issues: &[Issue]) -> &'static str {
    if issues.iter().any(|i| i.severity == "fail") {
        "FAIL"
    } else if issues.iter().any(|i| i.severity == "warn") {
        "WARN"
    } else {
        "OK"
    }
}

/// Whether a source `.1s` genuinely carries no renderable geometry (a dummy /
/// placeholder node tree), so an empty GLB from it is faithful, not a defect.
fn source_is_geometryless(path: &Path) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    if kart_fmt::raw_model::is_raw_model(&data) {
        return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let m = kart_fmt::raw_model::parse(&data);
            m.all_nodes()
                .iter()
                .all(|n| !n.is_mesh || n.vertices.is_empty())
        }))
        .unwrap_or(false);
    }
    // KartObject: parse (guarded — the parser can panic on a desynced stream) and
    // check whether any node carries geometry.
    match std::panic::catch_unwind(|| kart_fmt::kart_object::parse(&data)) {
        Ok(Ok(track)) => track.nodes.iter().all(|n| n.geom.is_none()),
        _ => false, // unparseable → not a clean empty; leave the issues as-is
    }
}

fn load_track_registry(root: &Path) -> Option<HashMap<String, Vec<String>>> {
    [
        root.join("track_common.rho/track@zz.bml"),
        root.join("track_/common/track@zz.bml"),
    ]
    .into_iter()
    .find_map(|path| {
        std::fs::read(path)
            .ok()
            .and_then(|b| kart_fmt::track_list::parse(&b))
            .map(|tl| {
                tl.tracks
                    .iter()
                    .map(|t| {
                        (
                            t.id.clone(),
                            t.texture_themes().iter().map(|s| s.to_string()).collect(),
                        )
                    })
                    .collect()
            })
    })
}

fn texture_dirs_for(
    root: &Path,
    one_s: &Path,
    registry: Option<&HashMap<String, Vec<String>>>,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let rel = one_s.strip_prefix(root).unwrap_or(one_s);
    let arch_name = rel
        .components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("");
    let track_id = track_id_of(root, one_s);
    if let Some(themes) = track_id
        .as_ref()
        .and_then(|id| registry.and_then(|reg| reg.get(id)))
    {
        for theme in themes {
            push_theme_dirs(&mut dirs, root, theme);
        }
    }
    if let Some(theme) = track_id.as_deref().and_then(|id| id.split('_').next()) {
        push_theme_dirs(&mut dirs, root, theme);
    }
    if let Some(theme) = theme_from_archive_name(arch_name) {
        push_theme_dirs(&mut dirs, root, &theme);
    }
    push_common_theme_dirs(&mut dirs, root);
    if let Some(parent) = one_s.parent() {
        push_unique(&mut dirs, parent.to_path_buf());
    }
    dirs
}

fn track_id_of(root: &Path, one_s: &Path) -> Option<String> {
    if let Some(parent) = one_s.parent() {
        if let Ok(rel) = parent.strip_prefix(root) {
            let comps: Vec<&str> = rel.iter().filter_map(|c| c.to_str()).collect();
            if let Some(i) = comps.iter().position(|c| *c == "track_") {
                if comps.len() > i + 2 {
                    return Some(format!("{}_{}", comps[i + 1], comps[i + 2]));
                }
            }
        }
    }
    let name = one_s.parent()?.file_name()?.to_str()?;
    let s = name.strip_prefix("track_").unwrap_or(name);
    let s = s.strip_suffix(".rho").unwrap_or(s);
    Some(s.to_string())
}

fn theme_from_archive_name(arch_name: &str) -> Option<String> {
    arch_name
        .strip_suffix(".rho")
        .or_else(|| arch_name.strip_suffix(".rho5"))
        .unwrap_or(arch_name)
        .strip_prefix("track_")
        .or_else(|| {
            arch_name
                .strip_suffix(".rho")
                .unwrap_or(arch_name)
                .strip_prefix("stage_")
        })
        .and_then(|s| s.split('_').next())
        .map(str::to_string)
}

fn push_theme_dirs(dirs: &mut Vec<PathBuf>, root: &Path, theme: &str) {
    push_unique(
        dirs,
        root.join(format!("theme_{theme}.rho")).join("texture"),
    );
    push_unique(dirs, root.join(format!("theme_{theme}.rho")));
    push_unique(dirs, root.join("theme_").join(theme).join("texture"));
    push_unique(dirs, root.join("theme_").join(theme));
}

fn push_common_theme_dirs(dirs: &mut Vec<PathBuf>, root: &Path) {
    for common in ["theme_common.rho", "track_common.rho"] {
        push_unique(dirs, root.join(common).join("texture"));
        push_unique(dirs, root.join(common));
    }
    for common in [("theme_", "common"), ("track_", "common")] {
        push_unique(dirs, root.join(common.0).join(common.1).join("texture"));
        push_unique(dirs, root.join(common.0).join(common.1));
    }
}

fn push_unique(dirs: &mut Vec<PathBuf>, path: PathBuf) {
    if !dirs.iter().any(|p| p == &path) {
        dirs.push(path);
    }
}

fn write_contact_sheet(
    audit_root: &Path,
    items: &[AuditItem],
    limit: usize,
) -> Result<String, String> {
    let pngs: Vec<_> = items
        .iter()
        .filter_map(|item| item.png_path.as_ref().map(|p| (item, audit_root.join(p))))
        .take(limit)
        .collect();
    if pngs.is_empty() {
        return Err("no PNGs for contact sheet".into());
    }
    let columns = pngs.len().min(5);
    let thumb_w = 220usize;
    let thumb_h = 180usize;
    let gutter = 16usize;
    let rows = pngs.len().div_ceil(columns);
    let width = (columns * thumb_w + (columns + 1) * gutter) as u32;
    let height = (rows * thumb_h + (rows + 1) * gutter) as u32;
    let mut sheet = vec![0u8; width as usize * height as usize * 4];
    for px in sheet.chunks_exact_mut(4) {
        px.copy_from_slice(&[31, 33, 37, 255]);
    }

    for (i, (_, path)) in pngs.iter().enumerate() {
        let (src_w, src_h, src) = read_png_rgba(path)?;
        let col = i % columns;
        let row = i / columns;
        let x0 = gutter + col * (thumb_w + gutter);
        let y0 = gutter + row * (thumb_h + gutter);
        blit_fit(
            &mut sheet,
            width as usize,
            height as usize,
            x0,
            y0,
            thumb_w,
            thumb_h,
            src_w as usize,
            src_h as usize,
            &src,
        );
    }

    let rel = "contact-sheet.png".to_string();
    let out = audit_root.join(&rel);
    write_rgba_png(&out, width, height, &sheet)?;
    Ok(rel)
}

fn read_png_rgba(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png frame: {e}"))?;
    let rgba = rgba_pixels(&buf[..info.buffer_size()], info.color_type)?;
    Ok((info.width, info.height, rgba))
}

fn blit_fit(
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    x0: usize,
    y0: usize,
    box_w: usize,
    box_h: usize,
    src_w: usize,
    src_h: usize,
    src: &[u8],
) {
    let scale = (box_w as f64 / src_w as f64).min(box_h as f64 / src_h as f64);
    let draw_w = (src_w as f64 * scale).max(1.0) as usize;
    let draw_h = (src_h as f64 * scale).max(1.0) as usize;
    let x_start = x0 + (box_w - draw_w) / 2;
    let y_start = y0 + (box_h - draw_h) / 2;
    for y in 0..draw_h {
        let sy = (y as f64 / scale) as usize;
        for x in 0..draw_w {
            let sx = (x as f64 / scale) as usize;
            if x_start + x >= dst_w || y_start + y >= dst_h {
                continue;
            }
            let src_i = ((sy.min(src_h - 1) * src_w + sx.min(src_w - 1)) * 4).min(src.len() - 4);
            let dst_i = ((y_start + y) * dst_w + (x_start + x)) * 4;
            dst[dst_i..dst_i + 4].copy_from_slice(&src[src_i..src_i + 4]);
        }
    }
}

fn write_rgba_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| format!("png data: {e}"))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value).map_err(|e| format!("json: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_markdown(path: &Path, report: &AuditReport) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("# kart-audit report\n\n");
    out.push_str(&format!(
        "- input: `{}`\n- glb: `{}`\n- source: `{}`\n- sampled: `{}` of `{}` candidates\n- rendered: `{}` (glb `{}`, source `{}`)\n- contact sheet: `{}` item(s)\n- status: OK `{}` / WARN `{}` / FAIL `{}`\n",
        report.input_root,
        report.glb_root,
        report.source_root.as_deref().unwrap_or("none"),
        report.summary.sampled,
        report.summary.candidates,
        report.summary.rendered,
        report.summary.glb_rendered,
        report.summary.source_rendered,
        report.summary.contact_sheet_items,
        report.summary.ok,
        report.summary.warn,
        report.summary.fail
    ));
    if let Some(sheet) = &report.summary.contact_sheet {
        out.push_str(&format!("\n![contact sheet]({sheet})\n"));
    }
    out.push_str("\n## Issues\n\n");
    if report.summary.issue_counts.is_empty() {
        out.push_str("No issues detected.\n");
    } else {
        for (code, count) in &report.summary.issue_counts {
            out.push_str(&format!("- `{code}`: {count}\n"));
        }
    }
    out.push_str("\n## Families\n\n");
    out.push_str("| family | candidates | sampled | ok | warn | fail |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for family in &report.families {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            family.family, family.candidates, family.sampled, family.ok, family.warn, family.fail
        ));
    }
    out.push_str("\n## Items\n\n");
    out.push_str("| status | family | path | tris | meshes | materials | images | issues |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|---|\n");
    for item in &report.items {
        let (tris, meshes, mats, imgs) = item
            .glb
            .as_ref()
            .map(|g| (g.triangles, g.meshes, g.materials, g.images))
            .unwrap_or((0, 0, 0, 0));
        let issues = item
            .issues
            .iter()
            .map(|i| i.code.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "| {} | {} | `{}` | {} | {} | {} | {} | {} |\n",
            item.status, item.family, item.rel_path, tris, meshes, mats, imgs, issues
        ));
    }
    std::fs::write(path, out).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_html(path: &Path, report: &AuditReport) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("<!doctype html><meta charset=\"utf-8\"><title>kart-audit</title>");
    out.push_str(
        "<style>
body{margin:0;background:#15171a;color:#e8eaed;font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,sans-serif}
header{position:sticky;top:0;background:#202329;border-bottom:1px solid #343941;padding:14px 20px;z-index:1}
main{padding:18px 20px 32px}
.meta{display:flex;gap:18px;flex-wrap:wrap;color:#b7bec9}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:18px}
.item{background:#202329;border:1px solid #343941;border-radius:6px;overflow:hidden}
.item img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#1f2125;display:block}
.body{padding:10px 12px}
.path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#d7dde6;word-break:break-all}
.stats{color:#aeb6c2;font-size:12px;margin-top:8px}
.issues{margin-top:8px;color:#ffcf8a;font-size:12px}
.OK{color:#78d78a}.WARN{color:#ffd166}.FAIL{color:#ff6b6b}
a{color:#8ab4ff}
</style>",
    );
    out.push_str("<header><h1>kart-audit</h1><div class=\"meta\">");
    out.push_str(&format!(
        "<span>sampled {}/{}</span><span>rendered {} · glb {} · source {}</span><span>sheet {}</span><span class=\"OK\">OK {}</span><span class=\"WARN\">WARN {}</span><span class=\"FAIL\">FAIL {}</span>",
        report.summary.sampled,
        report.summary.candidates,
        report.summary.rendered,
        report.summary.glb_rendered,
        report.summary.source_rendered,
        report.summary.contact_sheet_items,
        report.summary.ok,
        report.summary.warn,
        report.summary.fail
    ));
    out.push_str("</div></header><main>");
    if let Some(sheet) = &report.summary.contact_sheet {
        out.push_str(&format!(
            "<p><a href=\"{}\">contact sheet</a> · <a href=\"report.json\">report.json</a> · <a href=\"report.md\">report.md</a></p>",
            esc(sheet)
        ));
    }
    out.push_str("<section class=\"grid\">");
    for item in &report.items {
        out.push_str("<article class=\"item\">");
        if let Some(png) = &item.png_path {
            out.push_str(&format!(
                "<a href=\"{}\"><img src=\"{}\" loading=\"lazy\"></a>",
                esc(png),
                esc(png)
            ));
        }
        out.push_str("<div class=\"body\">");
        out.push_str(&format!(
            "<strong class=\"{}\">{}</strong> <span>{}</span><div class=\"path\">{}</div>",
            esc(&item.status),
            esc(&item.status),
            esc(&item.family),
            esc(&item.rel_path)
        ));
        if let Some(g) = &item.glb {
            out.push_str(&format!(
                "<div class=\"stats\">tris {} · meshes {} · prims {} · mats {} · images {}",
                g.triangles, g.meshes, g.primitives, g.materials, g.images
            ));
            if let Some(b) = &g.bbox {
                out.push_str(&format!(" · bbox diag {:.2}", b.diagonal));
            }
            out.push_str("</div>");
        }
        if !item.issues.is_empty() {
            let issues = item
                .issues
                .iter()
                .map(|i| format!("{}: {}", esc(&i.code), esc(&i.detail)))
                .collect::<Vec<_>>()
                .join("<br>");
            out.push_str(&format!("<div class=\"issues\">{issues}</div>"));
        }
        let mut links = Vec::new();
        if let Some(path) = &item.glb_png_path {
            links.push(format!("<a href=\"{}\">glb png</a>", esc(path)));
        }
        if let Some(path) = &item.source_png_path {
            links.push(format!("<a href=\"{}\">source png</a>", esc(path)));
        }
        links.push(format!(
            "<a href=\"{}\">item json</a>",
            esc(&item.item_json_path)
        ));
        out.push_str(&format!("<div class=\"stats\">{}</div>", links.join(" · ")));
        out.push_str("</div></article>");
    }
    out.push_str("</section></main>");
    std::fs::write(path, out).map_err(|e| format!("write {}: {e}", path.display()))
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn rel_to_string(path: &Path) -> String {
    path.components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_uses_virtual_and_datapack_roots() {
        assert_eq!(
            family_of_rel(Path::new("track_/village/R01/track.glb")),
            "track"
        );
        assert_eq!(
            family_of_rel(Path::new("DataPack2_00000.rho5/kart_/foo/model.glb")),
            "DataPack2"
        );
        assert_eq!(
            family_of_rel(Path::new("effect_/smoke/effect.glb")),
            "effect"
        );
    }

    #[test]
    fn spread_sample_is_deterministic() {
        let paths: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("{i}.glb"))).collect();
        let got = select_spread(&paths, 4);
        assert_eq!(
            got,
            vec![
                PathBuf::from("0.glb"),
                PathBuf::from("3.glb"),
                PathBuf::from("6.glb"),
                PathBuf::from("9.glb")
            ]
        );
    }
}
