//! End-to-end KartRider asset pipeline.
//!
//! Existing mode converts an already-unpacked archive tree:
//!   kart-pipeline <unpacked-root> <out-root> [--anims] [--usd]
//!
//! Client mode starts from a modern client root with `LK.txf`, unpacks every
//! `.rho`/`.rho5` listed there using the clean-room Rust unpackers, then runs the
//! same convert/index pass:
//!   kart-pipeline --client <client-root> <out-root> [--anims] [--usd]

use kart_fmt::loader::{load_char_anims, load_kart_anims};
use kart_fmt::{kart_object, raw_model, track_list};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Output backend selected on the command line.
#[derive(Clone, Copy, PartialEq)]
enum Format {
    Glb,
    Usda,
}

impl Format {
    fn ext(self) -> &'static str {
        match self {
            Format::Glb => "glb",
            Format::Usda => "usda",
        }
    }
}

#[derive(Default)]
struct FamStat {
    converted: usize,
    convertible: usize,
    anim_other: usize,
}

#[derive(Clone)]
struct Archive {
    phys: PathBuf,
    name: String,
    md5: String,
    is_rho5: bool,
    mount_path: Option<String>,
}

impl Archive {
    fn unpack_rel(&self) -> &str {
        if !self.is_rho5 {
            if let Some(mount_path) = self.mount_path.as_deref().filter(|s| !s.is_empty()) {
                return mount_path;
            }
        }
        &self.name
    }

    fn state_key(&self) -> String {
        format!("{}|{}", self.name, self.unpack_rel())
    }
}

struct Args {
    client_mode: bool,
    input_root: PathBuf,
    out_root: PathBuf,
    anims: bool,
    fmt: Format,
    limit: Option<usize>,
    rebuild: bool,
    unpack_only: bool,
    rho5_salt: String,
    /// Optional: drop stray outlier triangles from track geometry (off by default
    /// — diverges from C#, which faithfully keeps the source's stray vertices).
    clamp_outliers: bool,
}

fn main() {
    let args = Args::parse().unwrap_or_else(|e| {
        eprintln!("{e}");
        Args::usage();
        std::process::exit(2);
    });

    // Per-file panics (a parser hitting an unsupported class layout) are caught
    // and counted as failures — like the C# pipeline's per-file try/catch.
    std::panic::set_hook(Box::new(|_| {}));

    if args.client_mode {
        if let Err(e) = run_client(&args) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    } else {
        let convert_root = args.input_root.clone();
        let convert_out = args.out_root.clone();
        run_convert(&convert_root, &convert_out, &args);
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut client_mode = false;
        let mut anims = false;
        let mut fmt = Format::Glb;
        let mut limit = None;
        let mut rebuild = false;
        let mut unpack_only = false;
        let mut rho5_salt = kart_rho::rho5_archive::SALT_CN.to_string();
        let mut clamp_outliers = false;
        let mut pos = Vec::new();

        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < raw.len() {
            match raw[i].as_str() {
                "--client" => client_mode = true,
                "--anims" => anims = true,
                "--usd" => fmt = Format::Usda,
                "--rebuild" => rebuild = true,
                "--unpack-only" => unpack_only = true,
                "--clamp-outliers" => clamp_outliers = true,
                "--limit" => {
                    i += 1;
                    let Some(v) = raw.get(i) else {
                        return Err("--limit requires a number".into());
                    };
                    limit = Some(
                        v.parse::<usize>()
                            .map_err(|_| format!("bad --limit: {v}"))?,
                    );
                }
                "--rho5-salt" => {
                    i += 1;
                    let Some(v) = raw.get(i) else {
                        return Err("--rho5-salt requires cn|kr|tw|<salt>".into());
                    };
                    rho5_salt = match v.as_str() {
                        "cn" => kart_rho::rho5_archive::SALT_CN.to_string(),
                        "kr" => kart_rho::rho5_archive::SALT_KR.to_string(),
                        "tw" => kart_rho::rho5_archive::SALT_TW.to_string(),
                        other => other.to_string(),
                    };
                }
                "-h" | "--help" => {
                    Self::usage();
                    std::process::exit(0);
                }
                x if x.starts_with('-') => return Err(format!("unknown option: {x}")),
                x => pos.push(x.to_string()),
            }
            i += 1;
        }

        if client_mode {
            if pos.len() != 2 {
                return Err("--client mode requires <client-root> <out-root>".into());
            }
            Ok(Args {
                client_mode,
                input_root: PathBuf::from(&pos[0]),
                out_root: PathBuf::from(&pos[1]),
                anims,
                fmt,
                limit,
                rebuild,
                unpack_only,
                rho5_salt,
                clamp_outliers,
            })
        } else {
            let default_out = if fmt == Format::Usda {
                "pipeline/usda-rs"
            } else {
                "pipeline/glb-rs"
            };
            Ok(Args {
                client_mode,
                input_root: PathBuf::from(
                    pos.first()
                        .cloned()
                        .unwrap_or_else(|| "pipeline/unpacked".into()),
                ),
                out_root: PathBuf::from(pos.get(1).cloned().unwrap_or_else(|| default_out.into())),
                anims,
                fmt,
                limit,
                rebuild,
                unpack_only,
                rho5_salt,
                clamp_outliers,
            })
        }
    }

    fn usage() {
        eprintln!(
            "usage:\n  kart-pipeline <unpacked-root> <out-root> [--anims] [--usd] [--limit N]\n  kart-pipeline --client <client-root> <out-root> [--anims] [--usd] [--limit N] [--rebuild] [--unpack-only] [--clamp-outliers] [--rho5-salt cn|kr|tw|<salt>]"
        );
    }
}

fn run_client(args: &Args) -> Result<(), String> {
    let lk = args.input_root.join("LK.txf");
    let mut archives = parse_manifest(&lk, &args.input_root)?;
    let pack_stats = apply_pack_index(&mut archives, &args.input_root)?;
    if let Some(n) = args.limit {
        archives.truncate(n);
    }
    if archives.is_empty() {
        return Err(format!(
            "no .rho/.rho5 archives found from {}",
            lk.display()
        ));
    }
    println!(
        "[1] manifest: {} archive(s) from {}{}",
        archives.len(),
        lk.display(),
        pack_stats
            .as_ref()
            .map(|s| format!(
                " (aaa.pk mounted_rho={} rho5={} skipped_rho={} missing_rho={})",
                s.mounted_rho, s.rho5, s.skipped_rho, s.missing_rho
            ))
            .unwrap_or_default()
    );

    let unpack_root = args.out_root.join("unpacked");
    let converted_root = args.out_root.join(match args.fmt {
        Format::Glb => "glb-rs",
        Format::Usda => "usda-rs",
    });
    let state_path = args.out_root.join("state.tsv");
    std::fs::create_dir_all(&unpack_root)
        .map_err(|e| format!("mkdir {}: {e}", unpack_root.display()))?;

    let mut state = if args.rebuild {
        BTreeMap::new()
    } else {
        load_state(&state_path)
    };
    let mut unpacked = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    for a in &archives {
        let label = a.unpack_rel();
        let dest_rel = safe_rel_path(label)?;
        let dest = unpack_root.join(&dest_rel);
        let state_key = a.state_key();
        let up_to_date = !args.rebuild
            && dest.is_dir()
            && state.get(&state_key).is_some_and(|md5| md5 == &a.md5);
        if up_to_date {
            skipped += 1;
            continue;
        }
        state.remove(&state_key);
        if dest.exists() {
            std::fs::remove_dir_all(&dest).map_err(|e| format!("rm {}: {e}", dest.display()))?;
        }
        std::fs::create_dir_all(&dest).map_err(|e| format!("mkdir {}: {e}", dest.display()))?;
        match unpack_archive(a, &dest, &args.rho5_salt) {
            Ok(n) => {
                unpacked += 1;
                state.insert(state_key, a.md5.clone());
                if unpacked.is_multiple_of(100) {
                    println!("    ...unpacked {unpacked} archive(s)");
                }
                if n == 0 {
                    eprintln!("warn: {label} ({}) produced no files", a.name);
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("unpack failed {label} ({}): {e}", a.name);
            }
        }
    }
    save_state(&state_path, &state)?;
    println!(
        "[2] unpack: new/changed={unpacked} up-to-date={skipped} failed={failed} -> {}",
        unpack_root.display()
    );
    if failed > 0 {
        return Err(format!("{failed} archive(s) failed to unpack"));
    }
    if args.unpack_only {
        return Ok(());
    }
    run_convert(&unpack_root, &converted_root, args);
    Ok(())
}

fn run_convert(root: &Path, out_root: &Path, args: &Args) {
    let mut total = 0usize;
    let mut convertible = 0usize;
    let mut anim_other = 0usize;
    let mut converted = 0usize;
    let mut empty = 0usize;
    let mut failed = 0usize;
    let mut fail_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut fam: BTreeMap<String, FamStat> = BTreeMap::new();

    // The authoritative track registry (id → texture themes). Its `texTheme`
    // overrides the id-prefix theme on ~59 tracks (forest_R03 actually uses
    // beach, wkc_R03 uses village, …) and themes the DataPack-nested tracks whose
    // archive name carries no theme. Absent → None and texture resolution falls
    // back to the filename-derived theme (unchanged behaviour).
    let tex_themes = load_track_registry(root);
    if let Some(m) = &tex_themes {
        eprintln!(
            "track registry: {} tracks, texture themes resolved from track@zz.bml",
            m.len()
        );
    }

    let mut archives: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap_or_else(|e| panic!("read {root:?}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    archives.sort();
    if !args.client_mode {
        if let Some(n) = args.limit {
            archives.truncate(n);
        }
    }

    for arch in &archives {
        let arch_name = arch
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let family = family_of(&arch_name);

        for one_s in walk_1s(arch) {
            total += 1;
            let data = match std::fs::read(&one_s) {
                Ok(d) => d,
                Err(_) => continue,
            };
            // Convertibility mirrors C# IsConvertible: a raw model, or a
            // KartObject whose root stamp is a known class. Everything else
            // (animation clips, unregistered boss-anim classes, …) is anim/other
            // and excluded from the coverage denominator — exactly as C# does.
            if !kart_object::is_convertible(&data) {
                anim_other += 1;
                fam.entry(family.clone()).or_default().anim_other += 1;
                continue;
            }
            convertible += 1;
            fam.entry(family.clone()).or_default().convertible += 1;
            let rel = one_s.strip_prefix(root).unwrap_or(&one_s);
            let out = out_root.join(rel).with_extension(args.fmt.ext());
            let tex_dirs = track_tex_dirs(&one_s, &arch_name, root, tex_themes.as_ref());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                convert_one(
                    &one_s,
                    &data,
                    &tex_dirs,
                    args.anims,
                    args.fmt,
                    args.clamp_outliers,
                    &out,
                )
            }))
            .unwrap_or_else(|_| Err("panic (unsupported layout)".into()));
            match result {
                Ok(Outcome::Written) => {
                    converted += 1;
                    fam.entry(family.clone()).or_default().converted += 1;
                }
                Ok(Outcome::Empty) => {
                    // geometry-less dummy/placeholder — handled, no GLB written
                    empty += 1;
                    fam.entry(family.clone()).or_default().converted += 1;
                }
                Err(e) => {
                    failed += 1;
                    let r = reason_of(&e);
                    if std::env::var("KP_LOG_FAILS").is_ok() {
                        eprintln!("FAIL\t{r}\t{}", one_s.display());
                    }
                    *fail_reasons.entry(r).or_default() += 1;
                }
            }
        }
    }

    // Empties (geometry-less dummy/placeholder assets) are handled, not failed —
    // count them toward coverage alongside written files.
    let handled = converted + empty;
    let coverage = if convertible > 0 {
        100.0 * handled as f64 / convertible as f64
    } else {
        0.0
    };
    println!(
        "archives={}  total_1s={total}  convertible={convertible}  anim/other={anim_other}",
        archives.len()
    );
    println!("converted={converted}  empty={empty}  failed={failed}  coverage={coverage:.2}%");
    if !fail_reasons.is_empty() {
        println!("failures:");
        for (r, n) in &fail_reasons {
            println!("  [{n}] {r}");
        }
    }
    println!("by family (convertible desc):");
    let mut fams: Vec<_> = fam.into_iter().collect();
    fams.sort_by_key(|(_, s)| std::cmp::Reverse(s.convertible));
    for (f, s) in fams.iter().take(20) {
        println!(
            "  {f:18} converted={}/{} anim_other={}",
            s.converted, s.convertible, s.anim_other
        );
    }
}

fn parse_manifest(lk: &Path, client_root: &Path) -> Result<Vec<Archive>, String> {
    let text = std::fs::read_to_string(lk).map_err(|e| format!("read {}: {e}", lk.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 3 {
            continue;
        }
        let rel = fields[1]
            .trim()
            .trim_start_matches(['\\', '/'])
            .replace('\\', "/");
        let lower = rel.to_ascii_lowercase();
        if !lower.ends_with(".rho") && !lower.ends_with(".rho5") {
            continue;
        }
        let phys = client_root.join(&rel);
        if !phys.is_file() {
            continue;
        }
        let name = Path::new(&rel)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad archive path in LK.txf: {rel}"))?
            .to_string();
        out.push(Archive {
            phys,
            name,
            md5: fields[2].trim().to_string(),
            is_rho5: lower.ends_with(".rho5"),
            mount_path: None,
        });
    }
    Ok(out)
}

struct PackStats {
    mounted_rho: usize,
    rho5: usize,
    skipped_rho: usize,
    missing_rho: usize,
}

fn apply_pack_index(
    archives: &mut Vec<Archive>,
    client_root: &Path,
) -> Result<Option<PackStats>, String> {
    let aaa = client_root.join("Data/aaa.pk");
    if !aaa.is_file() {
        return Ok(None);
    }

    let index = kart_rho::pack::read_pack_index(&aaa)?;
    let mut by_name: HashMap<String, Archive> = archives
        .iter()
        .filter(|a| !a.is_rho5)
        .map(|a| (a.name.to_ascii_lowercase(), a.clone()))
        .collect();
    let mut ordered = Vec::new();
    let mut missing_rho = 0usize;
    for mount in &index.mounts {
        let key = mount.file_name.to_ascii_lowercase();
        if let Some(mut archive) = by_name.remove(&key) {
            archive.mount_path = Some(mount.mount_path.clone());
            ordered.push(archive);
        } else {
            missing_rho += 1;
        }
    }

    let skipped_rho = by_name.len();
    let mut rho5: Vec<Archive> = archives.iter().filter(|a| a.is_rho5).cloned().collect();
    rho5.sort_by(|a, b| a.name.cmp(&b.name));
    let rho5_count = rho5.len();

    let mounted_rho = ordered.len();
    ordered.extend(rho5);
    *archives = ordered;
    Ok(Some(PackStats {
        mounted_rho,
        rho5: rho5_count,
        skipped_rho,
        missing_rho,
    }))
}

fn unpack_archive(a: &Archive, dest: &Path, rho5_salt: &str) -> Result<usize, String> {
    let entries = if a.is_rho5 {
        kart_rho::rho5_archive::unpack_rho5(&a.phys, rho5_salt)?
    } else {
        kart_rho::archive::unpack(&a.phys)?
    };
    write_entries(dest, entries)
}

fn write_entries(dest: &Path, entries: Vec<kart_rho::archive::Entry>) -> Result<usize, String> {
    let mut written = 0usize;
    for e in entries {
        let rel = e.path.replace('\\', "/");
        if rel
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == "..")
        {
            eprintln!("skip suspicious path: {rel}");
            continue;
        }
        let out = dest.join(rel);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("mkdir {}: {err}", parent.display()))?;
        }
        std::fs::write(&out, e.data).map_err(|err| format!("write {}: {err}", out.display()))?;
        written += 1;
    }
    Ok(written)
}

fn safe_rel_path(rel: &str) -> Result<PathBuf, String> {
    if rel
        .split('/')
        .any(|c| c.is_empty() || c == "." || c == "..")
    {
        return Err(format!("unsafe archive output path: {rel}"));
    }
    Ok(PathBuf::from(rel))
}

fn load_state(path: &Path) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    text.lines().filter_map(parse_state_line).collect()
}

fn parse_state_line(line: &str) -> Option<(String, String)> {
    match line.split('\t').collect::<Vec<&str>>().as_slice() {
        [archive, md5] => Some((archive.to_string(), md5.to_string())),
        [archive, mount, md5] => Some((format!("{archive}|{mount}"), md5.to_string())),
        _ => None,
    }
}

/// Drop stray outlier triangles from a track's geometry (opt-in via
/// `--clamp-outliers`). A handful of tracks carry a vertex at an extreme
/// coordinate (e.g. z = -1.6e8) that the game culls at runtime but which blows up
/// the exported bbox. We compute a robust per-axis [2%, 98%] centre/extent over
/// all geometry, then drop any triangle with a vertex beyond 4× that extent from
/// the centre (expanding strips to a list so topology stays valid).
fn clamp_track_outliers(track: &mut kart_object::Track) {
    let mut axes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for n in &track.nodes {
        if let Some(g) = &n.geom {
            for v in &g.vertices {
                for k in 0..3 {
                    if v[k].is_finite() {
                        axes[k].push(v[k]);
                    }
                }
            }
        }
    }
    if axes[0].len() < 16 {
        return;
    }
    let (mut centre, mut half) = ([0f32; 3], [0f32; 3]);
    for k in 0..3 {
        let a = &mut axes[k];
        a.sort_by(f32::total_cmp);
        let lo = a[((a.len() as f32 * 0.02) as usize).min(a.len() - 1)];
        let hi = a[((a.len() as f32 * 0.98) as usize).min(a.len() - 1)];
        centre[k] = (lo + hi) * 0.5;
        half[k] = (hi - lo) * 0.5;
    }
    let limit = half.iter().cloned().fold(0.0f32, f32::max).max(1.0) * 4.0;
    let inside = |g: &kart_object::Geom, i: u32| -> bool {
        let Some(p) = g.vertices.get(i as usize) else {
            return false;
        };
        (0..3).all(|k| (p[k] - centre[k]).abs() <= limit)
    };
    for n in &mut track.nodes {
        if let Some(g) = &mut n.geom {
            let tris: Vec<[u32; 3]> = if g.strip {
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
            };
            let kept: Vec<u32> = tris
                .iter()
                .filter(|t| t.iter().all(|&i| inside(g, i)))
                .flatten()
                .copied()
                .collect();
            if kept.len() != g.indices.len() || g.strip {
                g.indices = kept;
                g.strip = false;
            }
        }
    }
}

fn save_state(path: &Path, state: &BTreeMap<String, String>) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let mut text = String::new();
    for (k, v) in state {
        text.push_str(k);
        text.push('\t');
        text.push_str(v);
        text.push('\n');
    }
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Convert a single `.1s` to the chosen backend, dispatching on format family.
/// Result of converting one `.1s`: a file was written, or the asset was a
/// geometry-less dummy/placeholder and intentionally skipped.
enum Outcome {
    Written,
    Empty,
}

fn convert_one(
    path: &Path,
    data: &[u8],
    tex_dirs: &[PathBuf],
    anims: bool,
    fmt: Format,
    clamp_outliers: bool,
    out: &Path,
) -> Result<Outcome, String> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
    let dir = path.parent().unwrap_or(Path::new("."));
    let bytes: Vec<u8> = if raw_model::is_raw_model(data) {
        let model = raw_model::parse(data);
        if model
            .all_nodes()
            .iter()
            .all(|n| !n.is_mesh || n.vertices.is_empty())
        {
            return Ok(Outcome::Empty); // dummy/placeholder model with no geometry
        }
        match fmt {
            Format::Glb => {
                let is_char = model.asset_type == "character";
                let (ca, ka) = if anims {
                    if is_char {
                        (load_char_anims(dir), Vec::new())
                    } else {
                        (Vec::new(), load_kart_anims(dir))
                    }
                } else {
                    (Vec::new(), Vec::new())
                };
                // model textures sit next to the .1s
                kart_gltf::scene::convert_raw(&model, name, dir, &ca, &ka).into_glb()
            }
            // USDA embeds the skeleton/skin from the model itself; no separate clips.
            Format::Usda => kart_usd::raw_model_usda(&model, name, dir).into_bytes(),
        }
    } else if data.len() >= 2 && data[0] == 0xAA && data[1] == 0x47 {
        // any KartObject asset (track/stage/effect/stuff/pet Relement tree)
        let mut track = kart_object::parse(data).map_err(|e| format!("kartobject parse: {e}"))?;
        if track.nodes.iter().all(|n| n.geom.is_none()) {
            return Ok(Outcome::Empty); // dummy/placeholder node tree with no geometry
        }
        if clamp_outliers {
            clamp_track_outliers(&mut track);
        }
        match fmt {
            // glTF resolves each texture across all theme dirs (texTheme + own
            // theme + commons), plus the .1s's own dir as a final fallback.
            Format::Glb => {
                let dirs: Vec<PathBuf> = tex_dirs
                    .iter()
                    .cloned()
                    .chain(std::iter::once(dir.to_path_buf()))
                    .collect();
                kart_gltf::scene::convert_track(&track, name, &dirs, anims).into_glb()
            }
            // USDA keeps single-dir resolution (first existing theme dir).
            Format::Usda => {
                let tex = tex_dirs
                    .iter()
                    .find(|d| d.exists())
                    .cloned()
                    .unwrap_or_else(|| dir.to_path_buf());
                kart_usd::track_usda(&track, name, &tex).into_bytes()
            }
        }
    } else {
        return Err("unsupported .1s format".into());
    };
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(out, bytes).map_err(|e| e.to_string())?;
    Ok(Outcome::Written)
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
            .and_then(|b| track_list::parse(&b))
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

fn walk_1s(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some("1s") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

/// Texture search dirs for one track `.1s`, theme-resolved via the BML track
/// registry when available. The registry's `texTheme` is authoritative — it
/// overrides the id-prefix theme on ~59 tracks and themes the DataPack-nested
/// tracks the archive name can't. Registry-resolved dirs are tried *first*, then
/// the filename-based dirs as a fallback, so a registry miss (or absent registry)
/// is byte-identical to the previous behaviour.
fn track_tex_dirs(
    one_s: &Path,
    arch_name: &str,
    root: &Path,
    registry: Option<&HashMap<String, Vec<String>>>,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let track_id = track_id_of(root, one_s);
    if let Some(themes) = track_id
        .as_ref()
        .and_then(|id| registry.and_then(|reg| reg.get(id)))
    {
        for t in themes {
            push_theme_dirs(&mut dirs, root, t);
        }
    }
    if let Some(theme) = track_id.as_deref().and_then(|id| id.split('_').next()) {
        push_theme_dirs(&mut dirs, root, theme);
    }
    if let Some(theme) = theme_from_archive_name(arch_name) {
        push_theme_dirs(&mut dirs, root, &theme);
    }
    push_common_theme_dirs(&mut dirs, root);
    dirs
}

/// Track id from a `.1s` path = its immediate parent directory name, stripped of
/// the `track_<id>.rho` archive decoration (DataPack-nested `track_/<id>/` dirs
/// are already bare). E.g. `track_village_R04.rho/track.1s` → `village_R04`.
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

/// Tracks/stages keep geometry textures in a shared theme archive
/// (`track_<theme>_*` / `stage_<theme>_*` → `theme_<theme>.rho`), plus commons.
fn theme_from_archive_name(arch_name: &str) -> Option<String> {
    arch_name
        .strip_suffix(".rho")
        .and_then(|s| {
            s.strip_prefix("track_")
                .or_else(|| s.strip_prefix("stage_"))
        })
        .and_then(|s| s.split('_').next())
        .map(|s| s.to_string())
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

fn family_of(arch_name: &str) -> String {
    arch_name
        .strip_suffix(".rho")
        .unwrap_or(arch_name)
        .split('_')
        .next()
        .unwrap_or("other")
        .to_string()
}

fn reason_of(msg: &str) -> String {
    if let Some(i) = msg.find("unknown class stamp ") {
        return msg[i..]
            .split_whitespace()
            .take(4)
            .collect::<Vec<_>>()
            .join(" ");
    }
    let m = msg.to_lowercase();
    if m.contains("unsupported") {
        "unsupported format".into()
    } else if m.contains("panic") {
        "panic (unsupported layout)".into()
    } else if m.contains("parse") {
        "parse error".into()
    } else {
        msg.chars().take(40).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_uses_mount_path_for_plain_rho() {
        let a = Archive {
            phys: PathBuf::from("Data/track_common.rho"),
            name: "track_common.rho".into(),
            md5: "abc".into(),
            is_rho5: false,
            mount_path: Some("track_/common".into()),
        };
        assert_eq!(a.unpack_rel(), "track_/common");
        assert_eq!(a.state_key(), "track_common.rho|track_/common");
    }

    #[test]
    fn archive_keeps_rho5_in_datapack_namespace() {
        let a = Archive {
            phys: PathBuf::from("Data/DataPack1_00000.rho5"),
            name: "DataPack1_00000.rho5".into(),
            md5: "abc".into(),
            is_rho5: true,
            mount_path: Some("etc_".into()),
        };
        assert_eq!(a.unpack_rel(), "DataPack1_00000.rho5");
    }

    #[test]
    fn track_id_supports_physical_and_virtual_layouts() {
        let root = Path::new("/tmp/unpacked");
        assert_eq!(
            track_id_of(
                root,
                Path::new("/tmp/unpacked/track_village_R04.rho/track.1s")
            ),
            Some("village_R04".into())
        );
        assert_eq!(
            track_id_of(root, Path::new("/tmp/unpacked/track_/gold/I03/track.1s")),
            Some("gold_I03".into())
        );
        assert_eq!(
            track_id_of(
                root,
                Path::new("/tmp/unpacked/DataPack3_00010.rho5/track_/wkc/R03/track.1s")
            ),
            Some("wkc_R03".into())
        );
    }

    #[test]
    fn texture_dirs_include_physical_and_virtual_theme_paths() {
        let root = Path::new("/tmp/unpacked");
        let dirs = track_tex_dirs(
            Path::new("/tmp/unpacked/track_/village/I01/track.1s"),
            "track_",
            root,
            None,
        );
        assert!(dirs.contains(&PathBuf::from("/tmp/unpacked/theme_village.rho/texture")));
        assert!(dirs.contains(&PathBuf::from("/tmp/unpacked/theme_/village/texture")));
        assert!(dirs.contains(&PathBuf::from("/tmp/unpacked/theme_common.rho/texture")));
        assert!(dirs.contains(&PathBuf::from("/tmp/unpacked/theme_/common/texture")));
    }
}
