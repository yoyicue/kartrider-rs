use std::collections::BTreeMap;
use std::env;
use std::f32::consts::PI;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod daemon;
mod navmesh;
mod win;
use daemon::{bool_text, read_tail_lines, read_tsv_header};
use kart_track_world::prelude::*;
use navmesh::NavMesh;
use win::{
    clear_known_ui_blockers, focus_kart_rider, input_desktop_context, launcher_post_key,
    post_key_to_kart_rider, process_integrity_level, InputDriver,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Key {
    W,
    A,
    S,
    D,
    V,
    Up,
    Down,
    Left,
    Right,
    Space,
    Shift,
    Ctrl,
    R,
    Esc,
    Tab,
    Enter,
    F5,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Command {
    Hold { keys: Vec<Key>, duration_ms: u64 },
    ReleaseAll,
    Sleep { duration_ms: u64 },
}

fn main() {
    if let Err(err) = run(env::args().skip(1).collect()) {
        eprintln!("kart-autoprobe: {err}");
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return Ok(());
    }

    match args[0].as_str() {
        "hold" => {
            let (keys, duration_ms) = parse_keys_duration(&args[1..])?;
            execute(
                &[Command::Hold { keys, duration_ms }],
                &mut EventLog::from_optional_path(None)?,
            )
        }
        "release-all" => execute(
            &[Command::ReleaseAll],
            &mut EventLog::from_optional_path(None)?,
        ),
        "script" => run_script(&args[1..]),
        "calibration-script" => run_calibration_script(&args[1..]),
        "fit-model" => run_fit_model(&args[1..]),
        "targets" => run_targets(&args[1..]),
        "kap-topology" => run_kap_topology(&args[1..]),
        "world-query-server" => run_world_query_server(&args[1..]),
        query if is_query_command(query) => run_command(query, &args[1..]),
        "kap-route-mine" => run_kap_route_mine(&args[1..]),
        "drive-to" => run_drive_to(&args[1..]),
        "drive-surface" => run_drive_surface(&args[1..]),
        "daemon" => daemon::run_daemon(&args[1..]),
        other => Err(format!("unknown command: {other}")),
    }
}

fn print_usage() {
    println!(
        "kart-autoprobe\n\
\n\
Usage:\n\
  kart-autoprobe hold <keys> <duration_ms>\n\
  kart-autoprobe release-all\n\
  kart-autoprobe script <script.tsv> [--log <probe_control.tsv>]\n\
  kart-autoprobe calibration-script [--out <script.tsv>]\n\
  kart-autoprobe fit-model <probe_pose.tsv> <probe_control.tsv> [--out <model.tsv>]\n\
  kart-autoprobe targets <track.1s> [--surface <token>] [--rail <token>] [--limit <n>]\n\
  kart-autoprobe kap-markers <path.kap>\n\
  kart-autoprobe kap-points <path.kap> [--step-m <n>] [--start-s <n>] [--end-s <n>]\n\
  kart-autoprobe kap-topology <track.1s> <path.kap>\n\
  kart-autoprobe world-node-info <track.1s> <node-name-substring>\n\
  kart-autoprobe world-dynamic-objects <track.1s>\n\
  kart-autoprobe world-dynamic-state <track.1s> <node-index> <time-ms> <delta-ms>\n\
  kart-autoprobe world-query-moving <track.1s> <time-ms> <x> <y> <z> [owner-index]\n\
  kart-autoprobe world-query-dynamic <track.1s> <time-ms> <x> <y> <z> [owner-index]\n\
  kart-autoprobe world-query <track.1s> <x> <y> <z>\n\
  kart-autoprobe world-rail-contexts <track.1s>\n\
  kart-autoprobe world-rail-entries <track.1s>\n\
  kart-autoprobe world-query-ground <track.1s> <x> <y> <z> <upx> <upy> <upz> <min-dot>\n\
  kart-autoprobe world-query-server <track.1s>\n\
  kart-autoprobe world-sweep <track.1s> <x0> <y0> <z0> <x1> <y1> <z1>\n\
  kart-autoprobe world-sweep-sphere <track.1s> <radius> <x0> <y0> <z0> <x1> <y1> <z1>\n\
  kart-autoprobe world-sweep-obb <track.1s> <x0> <y0> <z0> <x1> <y1> <z1> <a0x> <a0y> <a0z> <a1x> <a1y> <a1z> <a2x> <a2y> <a2z> <hx> <hy> <hz>\n\
  kart-autoprobe world-sweep-obb-rotating <track.1s> <start(3)> <end(3)> <start_axes(9)> <end_axes(9)> <half_extents(3)>\n\
  kart-autoprobe world-sweep-obb-rotating-all <track.1s> <start(3)> <end(3)> <start_axes(9)> <end_axes(9)> <half_extents(3)>\n\
  kart-autoprobe world-overlap-obb-all <track.1s> <center(3)> <axes(9)> <half_extents(3)>\n\
  kart-autoprobe world-event-obstacles <track.1s>\n\
  kart-autoprobe world-event-obstacle-contacts <track.1s> <time-ms> <delta-ms> <center(3)> <axes(9)> <half_extents(3)>\n\
  kart-autoprobe kap-route-mine --kap-dir <aiPath-dir> --kap-track <track_id> [--kap-slot <n>] [--window-m <n>] [--step-m <n>] [--max-heading-change-rad <n>] [--min-score <n>] [--limit <n>] [--out <windows.tsv>]\n\
  kart-autoprobe drive-to <probe_pose.tsv> <x> <y> <z> [--model <model.tsv>] [--log <probe_control.tsv>] [--collision <probe_collision_face.tsv>]\n\
  kart-autoprobe drive-surface <track.1s> <probe_pose.tsv> (--surface <token>|--rail <token>|--kap-dir <aiPath-dir> [--kap-track <track_id>]) [--model <model.tsv>] [--physics-baseline <baseline.tsv>] [--control-mode legacy|mpc] [--log <probe_control.tsv>] [--collision <probe_collision_face.tsv>] [--wall-hard <probe_wall_hard.tsv>] [--speed-limit <n>] [--abort-above-speed <n>] [--abort-below-speed <n>] [--abort-below-speed-after-ms <ms>] [--fixed-keys <keys>] [--fixed-key-sequence <keys:ms,...>] [--safe-cross-track <n>] [--panic-cross-track <n>] [--stuck-timeout-ms <ms>] [--start-path-s <m>] [--start-window-m <m>] [--heading-mode auto|pose-yaw|velocity] [--stop-on-surface] [--stop-on-state-kind <n>] [--timeout-ok] [--ignore-reachability]\n\
  kart-autoprobe daemon [--bind 0.0.0.0:37731] [--dynout D:\\kr\\dynout] [--probe-root D:\\kr\\probe\\KRPhysicsProbe] [--package-root D:\\kr\\packages]\n\
\n\
Keys are joined with '+', for example Up, Up+Left, Down, Space, Shift, Ctrl, R.\n\
Driving control uses sustained arrow-key holds. Script TSV rows: hold<TAB>Up+Left<TAB>750 or sleep<TAB>250 or release-all.\n"
    );
}

fn run_fit_model(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("expected <probe_pose.tsv> <probe_control.tsv>".to_string());
    }
    let pose_path = PathBuf::from(&args[0]);
    let control_path = PathBuf::from(&args[1]);
    let mut out_path = None;
    let mut cursor = FlagCursor::new(args, 2);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--out" => out_path = Some(PathBuf::from(cursor.take_value("--out requires a path")?)),
            other => return Err(format!("unknown fit-model option: {other}")),
        }
        cursor.advance();
    }

    let poses = read_pose_samples(&pose_path)?;
    let controls = read_control_samples(&control_path)?;
    let model = fit_vehicle_model(&poses, &controls);
    let text = model.serialize_tsv();
    match out_path {
        Some(path) => {
            fs::write(&path, text).map_err(|err| format!("write {}: {err}", path.display()))
        }
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn run_calibration_script(args: &[String]) -> Result<(), String> {
    let mut out_path = None;
    let mut cursor = FlagCursor::new(args, 0);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--out" => out_path = Some(PathBuf::from(cursor.take_value("--out requires a path")?)),
            other => return Err(format!("unknown calibration-script option: {other}")),
        }
        cursor.advance();
    }

    let rows = [
        "# low-speed vehicle response calibration",
        "# phase: stop, forward accel, coast drag, left/right turn, brake",
        "release-all",
        "sleep\t300",
        "hold\tUp\t1200",
        "sleep\t1000",
        "hold\tUp+Left\t1500",
        "sleep\t500",
        "hold\tUp+Right\t1500",
        "sleep\t500",
        "hold\tDown\t800",
        "release-all",
    ];
    let text = rows.join("\n") + "\n";
    match out_path {
        Some(path) => {
            fs::write(&path, text).map_err(|err| format!("write {}: {err}", path.display()))
        }
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn run_targets(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("track.1s path is required".to_string());
    }
    let track_path = PathBuf::from(&args[0]);
    let mut surface_filter = None;
    let mut rail_filter = None;
    let mut limit = usize::MAX;
    let mut cursor = FlagCursor::new(args, 1);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--surface" => {
                surface_filter = Some(cursor.take_value("--surface requires a token")?.to_string());
            }
            "--rail" => {
                rail_filter = Some(cursor.take_value("--rail requires a token")?.to_string());
            }
            "--limit" => {
                limit = cursor
                    .take_value("--limit requires a count")?
                    .parse::<usize>()
                    .map_err(|_| "--limit must be an integer".to_string())?;
            }
            other => return Err(format!("unknown targets option: {other}")),
        }
        cursor.advance();
    }

    let targets = kap_targets(
        &track_path,
        surface_filter.as_deref(),
        rail_filter.as_deref(),
    )?;
    println!(
        "target_id\ttrack_id\tnode\troad_surface\tsurface_id\troad_rail\ttri_index\tcx\tcy\tcz\tnx\tny\tnz"
    );
    for target in targets.iter().take(limit) {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            clean(&target.target_id),
            clean(&target.track_id),
            clean(&target.node),
            clean(&target.road_surface),
            target.surface_id,
            clean(&target.road_rail),
            target.tri_index,
            target.center[0],
            target.center[1],
            target.center[2],
            target.normal[0],
            target.normal[1],
            target.normal[2]
        );
    }
    Ok(())
}

fn run_world_query_server(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("expected <track.1s>".to_string());
    }
    let track_path = PathBuf::from(&args[0]);
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_world_query_server_session(&track_path, stdin.lock(), &mut stdout.lock())
}

fn run_kap_route_mine(args: &[String]) -> Result<(), String> {
    let mut request = KapFollowRequest::default();
    let mut options = KapRouteMineOptions::default();
    let mut out_path = None;
    let mut cursor = FlagCursor::new(args, 0);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--kap-dir" => {
                request.kap_dir = Some(PathBuf::from(
                    cursor.take_value("--kap-dir requires a path")?,
                ));
            }
            "--kap-track" => {
                request.track_id = Some(
                    cursor
                        .take_value("--kap-track requires a track id")?
                        .to_string(),
                );
            }
            "--kap-slot" => {
                request.slot = Some(
                    cursor
                        .take_value("--kap-slot requires a slot")?
                        .parse::<usize>()
                        .map_err(|_| "--kap-slot must be an integer".to_string())?,
                );
            }
            "--window-m" => {
                options.window_m = parse_f32_arg(
                    cursor.take_value("--window-m requires a distance")?,
                    "window-m",
                )?;
            }
            "--step-m" => {
                options.step_m =
                    parse_f32_arg(cursor.take_value("--step-m requires a distance")?, "step-m")?;
            }
            "--max-heading-change-rad" => {
                options.max_heading_change_rad = parse_f32_arg(
                    cursor.take_value("--max-heading-change-rad requires radians")?,
                    "max-heading-change-rad",
                )?;
            }
            "--min-score" => {
                options.min_score = parse_f32_arg(
                    cursor.take_value("--min-score requires a score")?,
                    "min-score",
                )?;
            }
            "--limit" => {
                options.limit = cursor
                    .take_value("--limit requires a count")?
                    .parse::<usize>()
                    .map_err(|_| "--limit must be an integer".to_string())?;
            }
            "--out" => {
                out_path = Some(PathBuf::from(cursor.take_value("--out requires a path")?));
            }
            other => return Err(format!("unknown kap-route-mine option: {other}")),
        }
        cursor.advance();
    }

    let paths = load_kap_follow_paths(&request)?;
    let mut windows = Vec::new();
    for path in &paths {
        windows.extend(mine_kap_route_windows(path, &options));
    }
    windows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path_id.cmp(&b.path_id))
            .then_with(|| {
                a.start_s
                    .partial_cmp(&b.start_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    if options.limit != usize::MAX {
        windows.truncate(options.limit);
    }

    let mut text = String::from(
        "rank\ttrack_id\tslot\tpath_id\tstart_s\tend_s\tcenter_s\tlength_m\tprogress_start\tprogress_end\theading_change_rad\tmax_segment_heading_delta_rad\tstraightness\tgrade_change\tgrade_abs_max\tscore\tkind\tverdict\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\n",
    );
    for (rank, window) in windows.iter().enumerate() {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            rank + 1,
            clean(&window.track_id),
            window
                .slot
                .map(|slot| slot.to_string())
                .unwrap_or_else(|| "".to_string()),
            clean(&window.path_id),
            window.start_s,
            window.end_s,
            window.center_s,
            window.length_m,
            window.progress_start,
            window.progress_end,
            window.heading_change_rad,
            window.max_segment_heading_delta_rad,
            window.straightness,
            window.grade_change,
            window.grade_abs_max,
            window.score,
            window.kind,
            window.verdict,
            window.start[0],
            window.start[1],
            window.start[2],
            window.end[0],
            window.end[1],
            window.end[2],
        ));
    }

    match out_path {
        Some(path) => {
            fs::write(&path, text).map_err(|err| format!("write {}: {err}", path.display()))
        }
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn run_drive_to(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err("expected <probe_pose.tsv> <x> <y> <z>".to_string());
    }
    let pose_path = PathBuf::from(&args[0]);
    let target = [
        parse_f32_arg(&args[1], "x")?,
        parse_f32_arg(&args[2], "y")?,
        parse_f32_arg(&args[3], "z")?,
    ];
    let options = parse_drive_options(&args[4..], "drive-to")?;
    let mut log = EventLog::from_optional_path(options.log_path.clone())?;
    drive_loop(&pose_path, target, options, &mut log)
}

fn run_drive_surface(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("expected <track.1s> <probe_pose.tsv>".to_string());
    }
    let track_path = PathBuf::from(&args[0]);
    let pose_path = PathBuf::from(&args[1]);
    let mut surface_filter = None;
    let mut rail_filter = None;
    let mut kap_follow = KapFollowRequest::default();
    let mut ignore_reachability = false;
    let mut passthrough = Vec::new();
    let mut cursor = FlagCursor::new(args, 2);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--surface" => {
                surface_filter = Some(cursor.take_value("--surface requires a token")?.to_string());
            }
            "--rail" => {
                rail_filter = Some(cursor.take_value("--rail requires a token")?.to_string());
            }
            "--kap-dir" => {
                kap_follow.kap_dir = Some(PathBuf::from(
                    cursor.take_value("--kap-dir requires a path")?,
                ));
            }
            "--kap-track" => {
                kap_follow.track_id = Some(
                    cursor
                        .take_value("--kap-track requires a track id")?
                        .to_string(),
                );
            }
            "--kap-slot" => {
                kap_follow.slot = Some(
                    cursor
                        .take_value("--kap-slot requires a slot")?
                        .parse::<usize>()
                        .map_err(|_| "--kap-slot must be an integer".to_string())?,
                );
            }
            "--lookahead" => {
                kap_follow.lookahead = cursor
                    .take_value("--lookahead requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--lookahead must be a number".to_string())?;
            }
            "--max-cross-track" => {
                kap_follow.max_cross_track = cursor
                    .take_value("--max-cross-track requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--max-cross-track must be a number".to_string())?;
            }
            "--target-progress" => {
                kap_follow.target_progress = cursor
                    .take_value("--target-progress requires a lap fraction")?
                    .parse::<f32>()
                    .map_err(|_| "--target-progress must be a number".to_string())?;
            }
            "--start-path-s" => {
                kap_follow.start_path_s = Some(
                    cursor
                        .take_value("--start-path-s requires a path distance")?
                        .parse::<f32>()
                        .map_err(|_| "--start-path-s must be a number".to_string())?,
                );
            }
            "--start-window-m" => {
                kap_follow.start_window_m = cursor
                    .take_value("--start-window-m requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--start-window-m must be a number".to_string())?;
            }
            "--lateral-offset" => {
                kap_follow.lateral_offset = cursor
                    .take_value("--lateral-offset requires a signed distance")?
                    .parse::<f32>()
                    .map_err(|_| "--lateral-offset must be a number".to_string())?;
            }
            "--offset-start-s" => {
                kap_follow.offset_start_s = Some(
                    cursor
                        .take_value("--offset-start-s requires a path distance")?
                        .parse::<f32>()
                        .map_err(|_| "--offset-start-s must be a number".to_string())?,
                );
            }
            "--offset-end-s" => {
                kap_follow.offset_end_s = Some(
                    cursor
                        .take_value("--offset-end-s requires a path distance")?
                        .parse::<f32>()
                        .map_err(|_| "--offset-end-s must be a number".to_string())?,
                );
            }
            "--offset-ramp-m" => {
                kap_follow.offset_ramp_m = cursor
                    .take_value("--offset-ramp-m requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--offset-ramp-m must be a number".to_string())?;
            }
            "--lateral-offset-zones" => {
                kap_follow.lateral_offset_zones =
                    parse_kap_lateral_offset_zones(&cursor.take_value(
                        "--lateral-offset-zones requires start:end:offset:ramp entries",
                    )?)?;
            }
            "--heading-mode" => {
                kap_follow.heading_mode = HeadingMode::parse(
                    cursor.take_value("--heading-mode requires auto, pose-yaw, or velocity")?,
                )?;
            }
            "--speed-limit" => {
                kap_follow.speed_limit = cursor
                    .take_value("--speed-limit requires a speed")?
                    .parse::<f32>()
                    .map_err(|_| "--speed-limit must be a number".to_string())?;
            }
            "--speed-zones" => {
                kap_follow.speed_zones = parse_kap_speed_zones(
                    &cursor.take_value("--speed-zones requires path_s:speed pairs")?,
                )?;
            }
            "--kap-gap-transfer" => {
                kap_follow.gap_transfer = parse_bool_arg(
                    cursor.take_value("--kap-gap-transfer requires true or false")?,
                    "kap-gap-transfer",
                )?;
            }
            "--abort-above-speed" => {
                kap_follow.abort_above_speed = cursor
                    .take_value("--abort-above-speed requires a speed")?
                    .parse::<f32>()
                    .map_err(|_| "--abort-above-speed must be a number".to_string())?;
            }
            "--abort-below-speed" => {
                kap_follow.abort_below_speed = cursor
                    .take_value("--abort-below-speed requires a speed")?
                    .parse::<f32>()
                    .map_err(|_| "--abort-below-speed must be a number".to_string())?;
            }
            "--abort-below-speed-after-ms" => {
                kap_follow.abort_below_speed_after_ms = cursor
                    .take_value("--abort-below-speed-after-ms requires milliseconds")?
                    .parse::<u64>()
                    .map_err(|_| "--abort-below-speed-after-ms must be an integer".to_string())?;
            }
            "--longitudinal-cooldown-ms" => {
                kap_follow.longitudinal_cooldown_ms = cursor
                    .take_value("--longitudinal-cooldown-ms requires milliseconds")?
                    .parse::<u64>()
                    .map_err(|_| "--longitudinal-cooldown-ms must be an integer".to_string())?;
            }
            "--fixed-keys" => {
                kap_follow.fixed_keys =
                    parse_keys(cursor.take_value("--fixed-keys requires keys like Up+Right")?)?;
            }
            "--fixed-add-keys" => {
                kap_follow.fixed_add_keys = parse_keys(
                    cursor.take_value("--fixed-add-keys requires keys like Up or Ctrl")?,
                )?;
            }
            "--fixed-add-key-sequence" => {
                kap_follow.fixed_add_key_sequence = parse_key_sequence(cursor.take_value(
                    "--fixed-add-key-sequence requires entries like None:1800,Up:5000",
                )?)?;
            }
            "--fixed-key-sequence" => {
                kap_follow.fixed_key_sequence = parse_key_sequence(cursor.take_value(
                    "--fixed-key-sequence requires entries like Up+Right:800,Right:5000",
                )?)?;
            }
            "--safe-cross-track" => {
                kap_follow.safe_cross_track = cursor
                    .take_value("--safe-cross-track requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--safe-cross-track must be a number".to_string())?;
            }
            "--panic-cross-track" => {
                kap_follow.panic_cross_track = cursor
                    .take_value("--panic-cross-track requires a distance")?
                    .parse::<f32>()
                    .map_err(|_| "--panic-cross-track must be a number".to_string())?;
            }
            "--stuck-timeout-ms" => {
                kap_follow.stuck_timeout_ms = cursor
                    .take_value("--stuck-timeout-ms requires milliseconds")?
                    .parse::<u64>()
                    .map_err(|_| "--stuck-timeout-ms must be an integer".to_string())?;
            }
            "--cross-track-timeout-ms" => {
                kap_follow.cross_track_timeout_ms = cursor
                    .take_value("--cross-track-timeout-ms requires milliseconds")?
                    .parse::<u64>()
                    .map_err(|_| "--cross-track-timeout-ms must be an integer".to_string())?;
            }
            "--log"
            | "--model"
            | "--physics-baseline"
            | "--control-mode"
            | "--max-ms"
            | "--step-ms"
            | "--stop-distance"
            | "--collision"
            | "--wall-hard"
            | "--cancel-file"
            | "--stop-on-surface-id"
            | "--stop-on-state-kind"
            | "--terminal-brake-ms"
            | "--terminal-brake-step-ms"
            | "--terminal-stop-speed"
            | "--terminal-settle-ms"
            | "--input-backend" => {
                let flag = flag.to_string();
                passthrough.push(flag.clone());
                passthrough.push(
                    cursor
                        .take_value(&format!("{flag} requires a value"))?
                        .to_string(),
                );
            }
            "--stop-on-surface" | "--timeout-ok" => passthrough.push(flag.to_string()),
            "--ignore-reachability" => ignore_reachability = true,
            other => return Err(format!("unknown drive-surface option: {other}")),
        }
        cursor.advance();
    }
    let mut options = parse_drive_options(&passthrough, "drive-surface")?;
    let pose = read_latest_drive_pose(&pose_path)?;
    if kap_follow.kap_dir.is_some() {
        if kap_follow.track_id.is_none() {
            kap_follow.track_id = Some(track_id(&track_path));
        }
        let mut log = EventLog::from_optional_path(options.log_path.clone())?;
        return drive_kap_follow_loop(&track_path, &pose_path, pose, kap_follow, options, &mut log);
    }
    if surface_filter.is_none() && rail_filter.is_none() {
        return Err("drive-surface requires --surface, --rail, or --kap-dir".to_string());
    }
    let targets = kap_targets(
        &track_path,
        surface_filter.as_deref(),
        rail_filter.as_deref(),
    )?;
    let nav_mesh = if ignore_reachability {
        None
    } else {
        Some(NavMesh::from_track(&track_path)?)
    };
    let target = choose_target(&targets, pose, nav_mesh.as_ref()).ok_or_else(|| {
        format!(
            "no KAP targets matched surface={:?} rail={:?}",
            surface_filter, rail_filter
        )
    })?;

    let mut log = EventLog::from_optional_path(options.log_path.clone())?;
    let reachability = if ignore_reachability {
        "ignored"
    } else {
        "checked"
    };
    log.write_target("target_select", pose, target, reachability)?;
    if options.stop_on_surface && options.stop_surface_id.is_none() && target.surface_id != 0 {
        options.stop_surface_id = Some(target.surface_id);
    }
    eprintln!(
        "selected target {} surface={} rail={} center=({:.3},{:.3},{:.3}) reachable={}",
        target.target_id,
        target.road_surface,
        target.road_rail,
        target.center[0],
        target.center[1],
        target.center[2],
        reachability
    );
    drive_loop(&pose_path, target.center, options, &mut log)
}

fn drive_kap_follow_loop(
    track_path: &Path,
    pose_path: &Path,
    initial_pose: Pose,
    request: KapFollowRequest,
    options: DriveOptions,
    log: &mut EventLog,
) -> Result<(), String> {
    let mut model = if let Some(path) = &options.model_path {
        VehicleModel::from_tsv(path)?
    } else {
        VehicleModel::low_speed_default()
    };
    let physics_baseline = options
        .physics_baseline_path
        .as_ref()
        .map(|path| PhysicsBaseline::from_tsv(path))
        .transpose()?;
    apply_request_speed_limit(&mut model, &request);
    let paths = load_kap_follow_paths(&request)?;
    let path = choose_kap_follow_path(&paths, initial_pose, &request)
        .ok_or_else(|| "no KAP path could be selected".to_string())?;
    let topology = if request.gap_transfer {
        let nav_mesh = NavMesh::from_track(track_path)?;
        KapTopologyProfile::from_path(path, &nav_mesh)
    } else {
        KapTopologyProfile::default()
    };
    log.write_kap_select("kap_select", initial_pose, path, &request)?;
    eprintln!(
        "selected KAP path {} points={} length={:.3} closed={} lookahead={:.3}",
        path.path_id,
        path.points.len(),
        path.length,
        path.closed,
        request.lookahead
    );

    let mut input = InputDriver::new(options.input_backend);
    let start = tick_millis();
    let mut cursor_s = initial_kap_projection(path, initial_pose, &request)
        .map(|projection| projection.path_s)
        .ok_or_else(|| {
            format!(
                "KAP path {} cannot project initial pose near start_path_s={:?}",
                path.path_id, request.start_path_s
            )
        })?;
    let mut credited_cursor_s = cursor_s;
    let mut progress_s = 0.0f32;
    let mut last_progress_ms = tick_millis();
    let mut last_material_progress_ms = last_progress_ms;
    let mut material_progress_checkpoint_s = 0.0f32;
    let mut wall_stall_recovery_latched = false;
    let mut last_activity_pos = initial_pose.pos;
    let mut last_longitudinal_ms: Option<u128> = None;
    let target_progress_s = if request.target_progress > 0.0 {
        Some(request.target_progress * path.length)
    } else {
        None
    };
    let projection_window = request
        .max_cross_track
        .max(request.lookahead * 4.0)
        .max(60.0);
    let seam_completion_tolerance_s = if path.closed {
        request
            .lookahead
            .max(request.safe_cross_track)
            .max(request.panic_cross_track)
            .max(20.0)
            * 3.0
    } else {
        0.0
    };
    let projection_backward_tolerance = (request.lookahead * 0.5).clamp(4.0, 12.0);
    let ground_drive_path = pose_path
        .parent()
        .map(|dir| dir.join("probe_ground_drive.tsv"))
        .unwrap_or_else(|| PathBuf::from("probe_ground_drive.tsv"));
    let wall_hard_path = options.wall_hard_path.clone().or_else(|| {
        pose_path
            .parent()
            .map(|dir| dir.join("probe_wall_hard.tsv"))
    });
    let state_path = pose_path
        .parent()
        .map(|dir| dir.join("probe_state.tsv"))
        .unwrap_or_else(|| PathBuf::from("probe_state.tsv"));
    let stop_state_baseline = options
        .stop_state_kind
        .and_then(|kind| read_latest_state_kind_seq(&state_path, kind).ok().flatten())
        .unwrap_or(0);
    let mut cross_track_violation_since = None;
    // Once a stacked-road downward transfer is physically confirmed, a
    // nearest-point recovery must not jump back across that topology edge.
    // Doing so alternated the steering target between factory_I02's upper
    // approach and lower continuation, making an otherwise identical run
    // either recover or oscillate depending on sampling timing.
    let mut downward_transfer_floor_s: Option<f32> = None;
    let control_step_ms = kap_control_step_ms(options.step_ms, request.speed_limit);
    let base_control_speed = model.max_control_speed;
    while tick_millis().saturating_sub(start) < u128::from(options.max_ms) {
        if options
            .cancel_path
            .as_ref()
            .is_some_and(|path| path.exists())
        {
            input.release_all()?;
            return Err("experiment_cancelled".to_string());
        }
        let pose = read_latest_drive_pose(pose_path)?;
        let ground_feedback = read_latest_ground_drive_feedback(&ground_drive_path, pose);
        let wall_feedback = wall_hard_path
            .as_ref()
            .and_then(|path| read_latest_wall_contact_feedback(path, pose.tick_ms));
        let max_progress_step_s = projection_window.max(request.lookahead * 4.0).max(60.0);
        let directional_projection = path
            .project_near_forward(
                pose.pos,
                cursor_s,
                max_progress_step_s,
                projection_backward_tolerance,
            )
            .ok_or_else(|| format!("KAP path {} cannot project pose", path.path_id))?;
        // The forward-only cursor prevents branch hopping during normal
        // driving, but after a real recovery arc the kart can be tens of
        // metres behind the retained cursor.  Continuing to reject that
        // bounded local regression makes the controller chase a target far
        // ahead and oscillate across the same road.  Once forward progress
        // has genuinely stalled, accept a materially better projection
        // inside the same local topology window.  This changes the estimator
        // cursor only; it never credits backward motion as route progress.
        let recovery_projection = if tick_millis().saturating_sub(last_progress_ms) >= 1_200 {
            path.project_near_strict(pose.pos, cursor_s, projection_window)
                .filter(|candidate| {
                    kap_recovery_projection_is_better(
                        path,
                        *candidate,
                        directional_projection,
                        cursor_s,
                        projection_window,
                        request.safe_cross_track,
                    )
                })
                .filter(|candidate| {
                    kap_recovery_projection_respects_topology_floor(
                        path,
                        *candidate,
                        downward_transfer_floor_s,
                        projection_window,
                    )
                })
        } else {
            None
        };
        let recovery_cursor_regression = recovery_projection.is_some();
        let raw_projection = recovery_projection.unwrap_or(directional_projection);
        let previous_cursor_s = cursor_s;
        let delta_s =
            path.forward_delta_bounded(cursor_s, raw_projection.path_s, max_progress_step_s);
        let raw_control_cross_track = kap_control_cross_track(
            path,
            pose.pos,
            raw_projection.path_s,
            raw_projection.cross_track,
            &request,
        );
        // A directional s-window alone is insufficient at stacked road
        // topology.  factory_I02's giant-gear transfer has an upper road and
        // a lower continuation inside the same forward window; once the kart
        // misses the moving transfer, the nearest allowed segment is still
        // "forward" but can be tens of metres below and hundreds of metres
        // away in weighted 3-D distance.  Never let such a remote candidate
        // advance the topology cursor.
        let projection_acceptance = kap_projection_acceptance(&request);
        let raw_vertical_gap = path
            .point_at(raw_projection.path_s)
            .map(|point| (pose.pos[2] - point[2]).abs())
            .unwrap_or(f32::INFINITY);
        let cursor_vertical_drop = path
            .point_at(cursor_s)
            .map(|point| point[2] - pose.pos[2])
            .unwrap_or(0.0);
        let downward_topology_transfer = kap_downward_topology_transfer(
            delta_s,
            cursor_vertical_drop,
            raw_vertical_gap,
            max_progress_step_s,
        );
        let remote_projection = !downward_topology_transfer
            && kap_projection_is_remote(
                KapProjection {
                    path_s: raw_projection.path_s,
                    cross_track: raw_control_cross_track,
                },
                raw_vertical_gap,
                &request,
            );
        let forward_delta_s = if remote_projection {
            0.0
        } else {
            delta_s.max(0.0)
        };
        // forward_delta_bounded returns 0 for both backward projections and
        // over-large forward jumps; hold the cursor in either case so it only
        // ever advances by an accepted forward step.
        let projection_held = remote_projection
            || (delta_s <= 0.0
                && path.path_s_distance(cursor_s, raw_projection.path_s) > 0.001
                && !recovery_cursor_regression);
        if !projection_held {
            cursor_s = raw_projection.path_s;
        }
        if downward_topology_transfer {
            downward_transfer_floor_s = Some(cursor_s);
        }
        let held_cross_track = if remote_projection {
            kap_control_cross_track(
                path,
                pose.pos,
                cursor_s,
                raw_projection.cross_track,
                &request,
            )
        } else {
            raw_control_cross_track
        };
        let projection = KapProjection {
            path_s: cursor_s,
            cross_track: held_cross_track,
        };
        model.max_control_speed =
            kap_speed_limit_at(&request.speed_zones, projection.path_s, base_control_speed);
        let progress_credit_s = kap_frontier_progress_credit(
            path,
            credited_cursor_s,
            cursor_s,
            max_progress_step_s,
            recovery_cursor_regression,
        );
        if progress_credit_s > 0.0 {
            progress_s += progress_credit_s;
            credited_cursor_s = cursor_s;
        }
        let seam_delta_s = path.seam_completion_delta(cursor_s, seam_completion_tolerance_s);
        if seam_delta_s > 0.0 && projection.cross_track <= request.max_cross_track {
            progress_s += seam_delta_s;
            cursor_s = 0.0;
            credited_cursor_s = 0.0;
            downward_transfer_floor_s = None;
        }
        let activity_delta = point_distance_2d(pose.pos, last_activity_pos);
        let position_replacement = kap_position_replacement_detected(activity_delta, &request);
        // Only forward path progress may refresh the KAP stuck watchdog.
        // Cross-track oscillation and driving backwards are physical activity,
        // but they do not move the experiment toward its route target.  Letting
        // activity_delta refresh this timer made a controller with an inverted
        // heading axis run until max_ms while repeatedly crossing the same
        // few metres of road.
        //
        // A large discontinuity is different: the game has authoritatively
        // replaced the pose (fall/reset). Keep the topology cursor so recovery
        // still targets the interrupted route, but restart the watchdog. On
        // factory_I02 the 2670 m fall replacement moved the kart 93 m and the
        // recovery converged from 94.8 m to 14.2 m cross-track before the old,
        // pre-reset timeout killed it.
        if forward_delta_s >= 0.25 || seam_delta_s > 0.0 || position_replacement {
            last_progress_ms = tick_millis();
        }
        // Sub-metre cursor creep while pushing into a wall is not meaningful
        // route progress. Keep last_progress_ms for immediate direction
        // authority, but only refresh recovery after a two-metre frontier gain.
        if kap_material_progress_advanced(progress_s, material_progress_checkpoint_s)
            || seam_delta_s > 0.0
            || position_replacement
        {
            material_progress_checkpoint_s = progress_s;
            last_material_progress_ms = tick_millis();
        }
        if activity_delta >= 0.5 {
            last_activity_pos = pose.pos;
        }
        let projection = KapProjection {
            path_s: cursor_s,
            cross_track: held_cross_track,
        };
        if let Some(kind) = options.stop_state_kind {
            if let Some(seq) = read_latest_state_kind_seq(&state_path, kind)? {
                if seq > stop_state_baseline {
                    input.release_all()?;
                    let decision = DriveDecision {
                        keys: Vec::new(),
                        done: true,
                        heading_error: 0.0,
                        reason: format!("state_kind_hit:{kind}:seq={seq}"),
                    };
                    log.write_kap_drive(
                        "kap_state_kind_hit",
                        &[],
                        0,
                        pose,
                        path.point_at(projection.path_s).unwrap_or(pose.pos),
                        &decision,
                        path,
                        projection,
                    )?;
                    terminal_safety_release(
                        &mut input,
                        log,
                        pose,
                        path.point_at(projection.path_s).unwrap_or(pose.pos),
                        "state_kind_hit",
                    )?;
                    return Ok(());
                }
            }
        }
        let effective_lookahead =
            adaptive_kap_lookahead(path, projection.path_s, request.lookahead);
        let target_s = projection.path_s + effective_lookahead;
        let target = path
            .point_at_lateral_offset(target_s, lateral_offset_at(path, target_s, &request))
            .ok_or_else(|| format!("KAP path {} cannot select lookahead target", path.path_id))?;
        // `model.max_control_speed` is updated from the active authored speed
        // zone above.  The planner and recovery layers must observe that same
        // value rather than the route-wide fallback in `request.speed_limit`.
        // Otherwise a 4 m/s bridge zone still plans at the global 10 m/s
        // cruise speed whenever adaptive lookahead remains at full length.
        let mut active_control_request = request.clone();
        active_control_request.speed_limit = model.max_control_speed;
        if projection.cross_track > request.max_cross_track {
            if downward_topology_transfer {
                // The active path owner changed from the upper branch to its
                // authored lower continuation. Recovery time must start at
                // that boundary, not at the earlier support-loss instant.
                cross_track_violation_since = None;
            }
            let since = *cross_track_violation_since.get_or_insert_with(tick_millis);
            let violation_ms = tick_millis().saturating_sub(since);
            // Panic recovery is already throttle-free above its recovery
            // speed. One second was shorter than the physical steering time
            // needed to rejoin factory_I02's broad elevated bend; it aborted
            // while still on the same z≈65 road surface. Keep the bounded
            // release, but allow a real recovery arc to complete.
            if violation_ms >= u128::from(request.cross_track_timeout_ms) {
                let abort_decision = DriveDecision {
                    keys: Vec::new(),
                    done: true,
                    heading_error: 0.0,
                    reason: format!(
                        "cross_track_abort:{:.3}>{:.3}:held={}ms:progress_s={:.3}",
                        projection.cross_track, request.max_cross_track, violation_ms, progress_s
                    ),
                };
                input.release_all()?;
                log.write_kap_drive(
                    "kap_cross_track_abort",
                    &[],
                    0,
                    pose,
                    target,
                    &abort_decision,
                    path,
                    projection,
                )?;
                terminal_safety_release(&mut input, log, pose, target, "cross_track_abort")?;
                return if options.timeout_ok {
                    Ok(())
                } else {
                    Err(abort_decision.reason)
                };
            }
        } else {
            cross_track_violation_since = None;
        }
        let mut decision = match options.control_mode {
            ControlMode::Legacy => plan_step_with_physics_feedback(
                &model,
                pose,
                target,
                0.0,
                request.heading_mode,
                ground_feedback.as_ref(),
                physics_baseline.as_ref(),
            ),
            ControlMode::Mpc => plan_step_mpc(
                &model,
                physics_baseline.as_ref(),
                path,
                pose,
                target,
                projection,
                &active_control_request,
                ground_feedback.as_ref(),
                wall_feedback.as_ref(),
            ),
        };
        // Accepted forward KAP progress is a stronger direction observation
        // than an orientation-matrix axis that can be π-flipped on this
        // client. Without this override factory_I02 reported
        // pose_yaw_reverse_velocity while cursor_s was increasing, disabled
        // Down, and coasted off elevated roads above 20 m/s.
        let route_forward_recent =
            progress_s > 0.0 && tick_millis().saturating_sub(last_progress_ms) <= 250;
        let no_progress_ms = tick_millis().saturating_sub(last_material_progress_ms);
        let stalled_velocity_telemetry = no_progress_ms >= 1_200 && activity_delta < 0.1;
        let safety_pose =
            kap_direction_authority_pose(pose, route_forward_recent, stalled_velocity_telemetry);
        apply_kap_safety_with_model(
            &mut decision,
            safety_pose,
            projection,
            &active_control_request,
            &model,
            physics_baseline.as_ref(),
            wall_feedback.as_ref(),
        );
        apply_kap_direction_aware_overspeed_with_model(&mut decision, safety_pose, &model);
        apply_adaptive_lookahead_speed_guard_with_model(
            &mut decision,
            safety_pose,
            request.lookahead,
            effective_lookahead,
            model.max_control_speed,
            &model,
        );
        apply_kap_low_speed_recovery(
            &mut decision,
            pose,
            projection,
            &active_control_request,
            no_progress_ms,
            physics_baseline.as_ref(),
        );
        apply_kap_wall_stall_recovery(
            &mut decision,
            pose,
            no_progress_ms,
            route_forward_recent,
            &mut wall_stall_recovery_latched,
        );
        if request.gap_transfer {
            if let Some(gap) = topology.active_gap(projection.path_s) {
                apply_kap_gap_transfer(&mut decision, pose, projection, gap, &request);
            }
        }
        if let Some((sequence_index, sequence_step)) = active_key_sequence_step(
            &request.fixed_key_sequence,
            tick_millis().saturating_sub(start),
        ) {
            let fixed = sequence_step.key_name();
            decision.keys = sequence_step.keys.clone();
            decision.reason = format!(
                "{}:fixed_key_sequence={sequence_index}:{}ms:{fixed}",
                decision.reason, sequence_step.duration_ms
            );
        } else if !request.fixed_keys.is_empty() {
            let fixed = request
                .fixed_keys
                .iter()
                .map(|key| key.name())
                .collect::<Vec<_>>()
                .join("+");
            decision.keys = request.fixed_keys.clone();
            decision.reason = format!("{}:fixed_keys={fixed}", decision.reason);
        }
        if let Some((sequence_index, sequence_step)) = active_key_sequence_step(
            &request.fixed_add_key_sequence,
            tick_millis().saturating_sub(start),
        ) {
            apply_fixed_add_keys(&mut decision, &sequence_step.keys);
            decision.reason = format!(
                "{}:fixed_add_key_sequence={sequence_index}:{}ms",
                decision.reason, sequence_step.duration_ms
            );
        } else {
            apply_fixed_add_keys(&mut decision, &request.fixed_add_keys);
        }
        if options.input_backend != InputBackend::SendInput {
            decision.reason = format!("{}:input={}", decision.reason, options.input_backend.name());
        }
        // Sparse low-speed propulsion prevents a single Up pulse from carrying the
        // kart far past the requested speed.  Never apply that cooldown to Down:
        // braking must remain immediately available while momentum is still rising.
        // Low-speed recovery is specifically a request for sustained
        // propulsion after sparse pulses failed to move the kart. Applying
        // the generic cooldown afterwards removed that Up again and made the
        // recovery branch self-cancelling on factory_I02 inclines.
        if kap_throttle_cooldown_applies(&decision) {
            let now = tick_millis();
            // Authored speed zones are the runtime control authority. Computing
            // this once from request.speed_limit left a zero/default request at
            // the 200 ms sparse-pulse setting even while the active zone asked
            // for normal 9--10 m/s propulsion. On factory_I02 that made the kart
            // roll backwards on the opening incline and oscillate over the same
            // path window forever.
            let longitudinal_cooldown_ms = kap_longitudinal_cooldown_ms(
                request.longitudinal_cooldown_ms,
                model.max_control_speed,
            );
            if last_longitudinal_ms
                .is_some_and(|last| now.saturating_sub(last) < u128::from(longitudinal_cooldown_ms))
            {
                decision.keys.retain(|key| *key != Key::Up);
                decision.reason = format!(
                    "{}:throttle_cooldown={}ms",
                    decision.reason, longitudinal_cooldown_ms
                );
            }
        }
        if projection.cross_track > request.max_cross_track {
            decision.reason = format!(
                "{}:cross_track_high:{:.3}>{:.3}",
                decision.reason, projection.cross_track, request.max_cross_track
            );
        }
        decision.reason = format!(
            "{}:kap_progress:cursor={:.3}:delta={:.3}:credit={:.3}:total={:.3}:target={:.3}",
            decision.reason,
            cursor_s,
            delta_s,
            progress_credit_s,
            progress_s,
            target_progress_s.unwrap_or(0.0)
        );
        if effective_lookahead + 1.0e-3 < request.lookahead {
            decision.reason = format!(
                "{}:kap_adaptive_lookahead:{:.3}->{:.3}",
                decision.reason, request.lookahead, effective_lookahead
            );
        }
        if seam_delta_s > 0.0 {
            decision.reason = format!(
                "{}:kap_seam_wrap:delta={:.3}:tolerance={:.3}",
                decision.reason, seam_delta_s, seam_completion_tolerance_s
            );
        }
        if projection_held {
            decision.reason = format!(
                "{}:kap_projection_held:raw={:.3}:previous={:.3}:remote={}:raw_cross={:.3}:raw_dz={:.3}:accept={:.3}",
                decision.reason,
                raw_projection.path_s,
                previous_cursor_s,
                remote_projection,
                raw_projection.cross_track,
                raw_vertical_gap,
                projection_acceptance,
            );
        }
        if recovery_cursor_regression {
            decision.reason = format!(
                "{}:kap_recovery_cursor_regression:raw={:.3}:previous={:.3}:distance={:.3}",
                decision.reason,
                raw_projection.path_s,
                previous_cursor_s,
                path.path_s_distance(previous_cursor_s, raw_projection.path_s),
            );
        }
        if downward_topology_transfer {
            decision.reason = format!(
                "{}:kap_downward_topology_transfer:delta={:.3}:cursor_drop={:.3}:raw_dz={:.3}",
                decision.reason, delta_s, cursor_vertical_drop, raw_vertical_gap,
            );
        }
        if position_replacement {
            decision.reason = format!(
                "{}:kap_position_replacement_watchdog_reset:delta={:.3}:threshold={:.3}",
                decision.reason,
                activity_delta,
                kap_position_replacement_threshold(&request),
            );
        }
        if request.abort_above_speed > 0.0 && pose.speed > request.abort_above_speed {
            input.release_all()?;
            let abort_decision = DriveDecision {
                keys: Vec::new(),
                done: true,
                heading_error: decision.heading_error,
                reason: format!(
                    "speed_abort:{:.3}>{:.3}:progress_s={:.3}:cross_track={:.3}:activity_delta={:.3}",
                    pose.speed, request.abort_above_speed, progress_s, projection.cross_track, activity_delta
                ),
            };
            log.write_kap_drive(
                "kap_speed_abort",
                &[],
                0,
                pose,
                target,
                &abort_decision,
                path,
                projection,
            )?;
            terminal_safety_release(&mut input, log, pose, target, "speed_abort")?;
            return if options.timeout_ok {
                Ok(())
            } else {
                Err(abort_decision.reason)
            };
        }
        if request.abort_below_speed > 0.0
            && request.abort_below_speed_after_ms > 0
            && tick_millis().saturating_sub(start) >= u128::from(request.abort_below_speed_after_ms)
            && pose.speed <= request.abort_below_speed
        {
            input.release_all()?;
            let abort_decision = DriveDecision {
                keys: Vec::new(),
                done: true,
                heading_error: decision.heading_error,
                reason: format!(
                    "low_speed_abort:{:.3}<={:.3}:after={}ms:progress_s={:.3}:cross_track={:.3}:activity_delta={:.3}",
                    pose.speed,
                    request.abort_below_speed,
                    request.abort_below_speed_after_ms,
                    progress_s,
                    projection.cross_track,
                    activity_delta
                ),
            };
            log.write_kap_drive(
                "kap_low_speed_abort",
                &[],
                0,
                pose,
                target,
                &abort_decision,
                path,
                projection,
            )?;
            terminal_safety_release(&mut input, log, pose, target, "low_speed_abort")?;
            return if options.timeout_ok {
                Ok(())
            } else {
                Err(abort_decision.reason)
            };
        }
        if request.stuck_timeout_ms > 0 && no_progress_ms > u128::from(request.stuck_timeout_ms) {
            input.release_all()?;
            let stuck_decision = DriveDecision {
                keys: Vec::new(),
                done: true,
                heading_error: decision.heading_error,
                reason: format!(
                    "stuck_no_progress:{}ms:progress_s={:.3}:cross_track={:.3}:activity_delta={:.3}",
                    request.stuck_timeout_ms, progress_s, projection.cross_track, activity_delta
                ),
            };
            log.write_kap_drive(
                "kap_stuck",
                &[],
                0,
                pose,
                target,
                &stuck_decision,
                path,
                projection,
            )?;
            terminal_safety_release(&mut input, log, pose, target, "stuck_abort")?;
            return if options.timeout_ok {
                Ok(())
            } else {
                Err(stuck_decision.reason)
            };
        }

        if decision.keys.contains(&Key::Up) {
            last_longitudinal_ms = Some(tick_millis());
        }
        let applied_step_ms = kap_applied_step_ms(control_step_ms, &decision, pose);
        input.key_down_many(&decision.keys)?;
        log.write_kap_drive(
            "kap_step_begin",
            &decision.keys,
            applied_step_ms,
            pose,
            target,
            &decision,
            path,
            projection,
        )?;
        thread::sleep(Duration::from_millis(applied_step_ms));
        input.key_up_many(&decision.keys)?;
        log.write_kap_drive(
            "kap_step_end",
            &decision.keys,
            applied_step_ms,
            pose,
            target,
            &decision,
            path,
            projection,
        )?;
        if let Some(target_progress_s) = target_progress_s {
            if progress_s >= target_progress_s {
                input.release_all()?;
                let reached_decision = DriveDecision {
                    keys: Vec::new(),
                    done: true,
                    heading_error: 0.0,
                    reason: format!(
                        "target_progress_reached:{:.6}:progress_s={:.3}:target_s={:.3}",
                        request.target_progress, progress_s, target_progress_s
                    ),
                };
                log.write_kap_drive(
                    "kap_target_reached",
                    &[],
                    0,
                    pose,
                    target,
                    &reached_decision,
                    path,
                    projection,
                )?;
                terminal_brake(pose_path, target, &options, &mut input, log, None)?;
                return Ok(());
            }
        }
    }

    input.release_all()?;
    let pose = read_latest_pose(pose_path)?;
    let projection = path
        .project_near_forward(
            pose.pos,
            cursor_s,
            projection_window.max(request.lookahead * 4.0).max(60.0),
            projection_backward_tolerance,
        )
        .ok_or_else(|| format!("KAP path {} cannot project final pose", path.path_id))?;
    let target = path.point_at(projection.path_s).unwrap_or(pose.pos);
    let decision = DriveDecision {
        keys: Vec::new(),
        done: true,
        heading_error: 0.0,
        reason: format!("timeout:{}ms", options.max_ms),
    };
    log.write_kap_drive(
        "kap_timeout",
        &[],
        0,
        pose,
        target,
        &decision,
        path,
        projection,
    )?;
    terminal_brake(pose_path, target, &options, &mut input, log, None)?;
    if options.timeout_ok {
        Ok(())
    } else {
        Err(format!("KAP follow timed out after {}ms", options.max_ms))
    }
}

fn kap_control_step_ms(requested_step_ms: u64, speed_limit: f32) -> u64 {
    if speed_limit > 0.0 && speed_limit <= 8.0 {
        requested_step_ms.min(40)
    } else {
        requested_step_ms
    }
}

fn kap_material_progress_advanced(progress_s: f32, checkpoint_s: f32) -> bool {
    progress_s.is_finite() && checkpoint_s.is_finite() && progress_s - checkpoint_s >= 2.0
}

fn kap_applied_step_ms(control_step_ms: u64, decision: &DriveDecision, pose: Pose) -> u64 {
    if decision.keys.contains(&Key::Down) && pose.speed >= 10.0 && !pose_has_reverse_velocity(pose)
    {
        // A factory_I02 direction-response pulse did not enter measurable
        // braking until roughly 100 ms after Down.  A 160 ms planner pulse can
        // therefore expire inside the selector/actuator transient and alternate
        // immediately back to Up.  Keep steering intact but hold established
        // forward braking long enough to cross the observed latency.
        control_step_ms.max(320)
    } else if decision.keys.contains(&Key::Down)
        && pose.speed >= 5.0
        && !pose_has_reverse_velocity(pose)
    {
        // At 5--10 m/s the 320 ms pulse crossed zero in factory_I02's 5 m/s
        // descent zone and selected reverse. Retain enough dwell to cross the
        // actuator latency without applying a high-speed pulse near rest.
        control_step_ms.max(180)
    } else {
        control_step_ms
    }
}

fn kap_longitudinal_cooldown_ms(requested_cooldown_ms: u64, speed_limit: f32) -> u64 {
    if speed_limit > 0.0 && speed_limit <= 8.0 {
        requested_cooldown_ms.max(200)
    } else if speed_limit > 8.0 {
        // Sparse propulsion belongs to the low-speed acceptance regime. At
        // normal route speed it repeatedly lost factory_I02 uphill momentum
        // around 1 m/s; the planner's speed/overspeed guards own control here.
        0
    } else {
        requested_cooldown_ms
    }
}

fn kap_throttle_cooldown_applies(decision: &DriveDecision) -> bool {
    decision.keys.contains(&Key::Up)
        && !decision.reason.contains(":kap_low_speed_steering_recover:")
}

fn kap_direction_authority_pose(
    mut pose: Pose,
    route_forward_recent: bool,
    stalled_velocity_telemetry: bool,
) -> Pose {
    if route_forward_recent {
        pose.velocity_heading = None;
    }
    if stalled_velocity_telemetry {
        // Some collision/rest states retain the last nonzero velocity vector
        // while position is stationary. Let accepted position progress own
        // motion authority so corridor recovery can apply Up and unstick.
        pose.speed = 0.0;
        pose.velocity_heading = None;
    }
    pose
}

fn apply_request_speed_limit(model: &mut VehicleModel, request: &KapFollowRequest) {
    if request.speed_limit > 0.0 {
        model.max_control_speed = request.speed_limit;
    }
}

fn parse_kap_speed_zones(value: &str) -> Result<Vec<(f32, f32)>, String> {
    let mut zones = Vec::new();
    for raw in value.split(',').filter(|part| !part.trim().is_empty()) {
        let (path_s, speed) = raw
            .split_once(':')
            .ok_or_else(|| format!("invalid speed zone {raw:?}; expected path_s:speed"))?;
        let path_s = path_s
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid speed-zone path_s: {path_s:?}"))?;
        let speed = speed
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid speed-zone speed: {speed:?}"))?;
        if !path_s.is_finite() || path_s < 0.0 || !speed.is_finite() || speed <= 0.0 {
            return Err(format!(
                "speed zone values must be finite with path_s>=0 and speed>0: {raw:?}"
            ));
        }
        zones.push((path_s, speed));
    }
    zones.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if zones
        .windows(2)
        .any(|pair| (pair[0].0 - pair[1].0).abs() <= f32::EPSILON)
    {
        return Err("speed zones must have unique path_s boundaries".to_string());
    }
    Ok(zones)
}

fn parse_kap_lateral_offset_zones(value: &str) -> Result<Vec<KapLateralOffsetZone>, String> {
    let mut zones = Vec::new();
    for raw in value.split(',').filter(|part| !part.trim().is_empty()) {
        let fields = raw.split(':').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(format!(
                "invalid lateral offset zone {raw:?}; expected start:end:offset:ramp"
            ));
        }
        let parse = |field: &str, name: &str| {
            field
                .parse::<f32>()
                .map_err(|_| format!("invalid lateral-offset-zone {name}: {field:?}"))
        };
        let zone = KapLateralOffsetZone {
            start_s: parse(fields[0], "start")?,
            end_s: parse(fields[1], "end")?,
            offset: parse(fields[2], "offset")?,
            ramp_m: parse(fields[3], "ramp")?,
        };
        if !zone.start_s.is_finite()
            || !zone.end_s.is_finite()
            || !zone.offset.is_finite()
            || !zone.ramp_m.is_finite()
            || zone.start_s < 0.0
            || zone.end_s <= zone.start_s
            || zone.ramp_m < 0.0
        {
            return Err(format!(
                "lateral offset zone values must be finite with 0<=start<end and ramp>=0: {raw:?}"
            ));
        }
        zones.push(zone);
    }
    zones.sort_by(|a, b| {
        a.start_s
            .partial_cmp(&b.start_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if zones.windows(2).any(|pair| pair[0].end_s > pair[1].start_s) {
        return Err("lateral offset zones must not overlap".to_string());
    }
    Ok(zones)
}

fn kap_speed_limit_at(zones: &[(f32, f32)], path_s: f32, fallback: f32) -> f32 {
    zones
        .iter()
        .take_while(|(start_s, _)| *start_s <= path_s)
        .map(|(_, speed)| *speed)
        .last()
        .unwrap_or(fallback)
}

fn drive_loop(
    pose_path: &Path,
    target: P3,
    options: DriveOptions,
    log: &mut EventLog,
) -> Result<(), String> {
    let model = if let Some(path) = &options.model_path {
        VehicleModel::from_tsv(path)?
    } else {
        VehicleModel::low_speed_default()
    };
    let mut input = InputDriver::new(options.input_backend);
    let start = tick_millis();
    let state_path = pose_path
        .parent()
        .map(|dir| dir.join("probe_state.tsv"))
        .unwrap_or_else(|| PathBuf::from("probe_state.tsv"));
    let stop_state_baseline = options
        .stop_state_kind
        .and_then(|kind| read_latest_state_kind_seq(&state_path, kind).ok().flatten())
        .unwrap_or(0);
    let initial_pose = read_latest_pose(pose_path)?;
    let mut last_activity_pos = initial_pose.pos;
    let mut last_progress_ms = tick_millis();
    let mut best_distance = point_distance_2d(initial_pose.pos, target);
    let mut unstuck_attempts = 0u32;
    let heading_calibration_path = options.heading_calibration_path.clone().unwrap_or_else(|| {
        pose_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("probe_heading_calibration.tsv")
    });
    let mut pose_yaw_offset = read_heading_calibration(&heading_calibration_path)?;
    let mut previous_step_was_forward = false;
    while tick_millis().saturating_sub(start) < u128::from(options.max_ms) {
        if options
            .cancel_path
            .as_ref()
            .is_some_and(|path| path.exists())
        {
            input.release_all()?;
            return Err("experiment_cancelled".to_string());
        }
        let pose = read_latest_pose(pose_path)?;
        let previous_offset = pose_yaw_offset;
        let planning_pose =
            pose_with_learned_forward_yaw(pose, previous_step_was_forward, &mut pose_yaw_offset);
        if option_angle_changed(previous_offset, pose_yaw_offset) {
            write_heading_calibration(&heading_calibration_path, pose_yaw_offset.unwrap(), pose)?;
        }
        let now = tick_millis();
        let activity_delta = point_distance_2d(pose.pos, last_activity_pos);
        if activity_delta >= 0.5 {
            last_activity_pos = pose.pos;
            last_progress_ms = now;
        }
        let target_distance = point_distance_2d(pose.pos, target);
        if target_distance + 1.0 < best_distance {
            best_distance = target_distance;
            unstuck_attempts = 0;
        }
        if let Some(kind) = options.stop_state_kind {
            if let Some(seq) = read_latest_state_kind_seq(&state_path, kind)? {
                if seq > stop_state_baseline {
                    let decision = DriveDecision {
                        keys: Vec::new(),
                        done: true,
                        heading_error: 0.0,
                        reason: format!("state_kind_hit:{kind}:seq={seq}"),
                    };
                    log.write_drive(
                        "drive_state_kind_hit",
                        &[],
                        options.step_ms,
                        pose,
                        target,
                        &decision,
                    )?;
                    terminal_brake(
                        pose_path,
                        target,
                        &options,
                        &mut input,
                        log,
                        pose_yaw_offset,
                    )?;
                    return Ok(());
                }
            }
        }
        if let Some(surface_id) = options.stop_surface_id {
            if let Some(collision_path) = &options.collision_path {
                if read_latest_surface_id(collision_path)? == Some(surface_id) {
                    let decision = DriveDecision {
                        keys: Vec::new(),
                        done: true,
                        heading_error: 0.0,
                        reason: format!("surface_hit:{surface_id}"),
                    };
                    input.release_all()?;
                    log.write_drive(
                        "drive_surface_hit",
                        &[],
                        options.step_ms,
                        pose,
                        target,
                        &decision,
                    )?;
                    return Ok(());
                }
            }
        }
        let no_progress_ms = now.saturating_sub(last_progress_ms);
        if options.stuck_timeout_ms > 0 && no_progress_ms >= u128::from(options.stuck_timeout_ms) {
            if unstuck_attempts >= options.max_unstuck_attempts {
                input.release_all()?;
                let decision = DriveDecision {
                    keys: Vec::new(),
                    done: true,
                    heading_error: 0.0,
                    reason: format!(
                        "stuck_after_{}_attempts:no_progress={}ms:distance={:.3}",
                        unstuck_attempts, no_progress_ms, target_distance
                    ),
                };
                log.write_drive("drive_stuck", &[], 0, pose, target, &decision)?;
                return if options.timeout_ok {
                    Ok(())
                } else {
                    Err(decision.reason)
                };
            }
            unstuck_attempts += 1;
            let decision = DriveDecision {
                keys: vec![Key::Down],
                done: false,
                heading_error: 0.0,
                reason: format!(
                    "unstuck_reverse:attempt={}/{}:no_progress={}ms:activity={:.3}:distance={:.3}",
                    unstuck_attempts,
                    options.max_unstuck_attempts,
                    no_progress_ms,
                    activity_delta,
                    target_distance
                ),
            };
            input.release_all()?;
            input.key_down_many(&decision.keys)?;
            log.write_drive(
                "drive_unstuck_begin",
                &decision.keys,
                options.unstuck_reverse_ms,
                pose,
                target,
                &decision,
            )?;
            thread::sleep(Duration::from_millis(options.unstuck_reverse_ms));
            input.key_up_many(&decision.keys)?;
            log.write_drive(
                "drive_unstuck_end",
                &decision.keys,
                options.unstuck_reverse_ms,
                pose,
                target,
                &decision,
            )?;
            last_progress_ms = tick_millis();
            last_activity_pos = read_latest_pose(pose_path)?.pos;
            previous_step_was_forward = false;
            continue;
        }
        let mut decision = plan_step_with_heading_mode(
            &model,
            planning_pose,
            target,
            options.stop_distance,
            options.heading_mode,
        );
        if options.input_backend != InputBackend::SendInput {
            decision.reason = format!("{}:input={}", decision.reason, options.input_backend.name());
        }
        if decision.done {
            log.write_drive("drive_done", &[], options.step_ms, pose, target, &decision)?;
            terminal_brake(
                pose_path,
                target,
                &options,
                &mut input,
                log,
                pose_yaw_offset,
            )?;
            return Ok(());
        }

        input.key_down_many(&decision.keys)?;
        log.write_drive(
            "drive_step_begin",
            &decision.keys,
            options.step_ms,
            pose,
            target,
            &decision,
        )?;
        thread::sleep(Duration::from_millis(options.step_ms));
        input.key_up_many(&decision.keys)?;
        log.write_drive(
            "drive_step_end",
            &decision.keys,
            options.step_ms,
            pose,
            target,
            &decision,
        )?;
        previous_step_was_forward =
            decision.keys.contains(&Key::Up) && !decision.keys.contains(&Key::Down);
    }

    input.release_all()?;
    let decision = DriveDecision {
        keys: Vec::new(),
        done: true,
        heading_error: 0.0,
        reason: format!("timeout:{}ms", options.max_ms),
    };
    let pose = read_latest_pose(pose_path)?;
    log.write_drive("drive_timeout", &[], 0, pose, target, &decision)?;
    terminal_brake(
        pose_path,
        target,
        &options,
        &mut input,
        log,
        pose_yaw_offset,
    )?;
    if options.timeout_ok {
        Ok(())
    } else {
        Err(format!("drive timed out after {}ms", options.max_ms))
    }
}

fn terminal_safety_release(
    input: &mut InputDriver,
    log: &mut EventLog,
    pose: Pose,
    target: P3,
    cause: &str,
) -> Result<(), String> {
    input.release_all()?;
    let decision = DriveDecision {
        keys: Vec::new(),
        done: true,
        heading_error: 0.0,
        reason: format!(
            "terminal_safety_release:cause={cause}:speed={:.3}",
            pose.speed
        ),
    };
    log.write_drive(
        "drive_terminal_safety_release",
        &[],
        0,
        pose,
        target,
        &decision,
    )
}

fn terminal_brake(
    pose_path: &Path,
    target: P3,
    options: &DriveOptions,
    input: &mut InputDriver,
    log: &mut EventLog,
    _yaw_offset: Option<f32>,
) -> Result<(), String> {
    input.release_all()?;
    if options.terminal_brake_ms == 0 {
        return Ok(());
    }
    let start = tick_millis();
    let terminal_anchor = read_latest_pose(pose_path)?.pos;
    let mut brake_key = Key::Down;
    let mut direction_probed = false;
    let mut best_speed = f32::INFINITY;
    let mut low_speed_since = None;
    let mut low_speed_anchor = None;
    let mut roll_resume_count = 0u32;
    let mut coast_probe = true;
    while tick_millis().saturating_sub(start) < u128::from(options.terminal_brake_ms) {
        let pose = read_latest_pose(pose_path)?;
        let terminal_drift = point_distance_2d(terminal_anchor, pose.pos);
        if terminal_drift > 8.0 {
            input.release_all()?;
            let decision = DriveDecision {
                keys: Vec::new(),
                done: true,
                heading_error: 0.0,
                reason: format!(
                    "terminal_safety_abort:drift={terminal_drift:.3}:speed={:.3}",
                    pose.speed
                ),
            };
            log.write_drive(
                "drive_terminal_safety_abort",
                &[],
                0,
                pose,
                target,
                &decision,
            )?;
            return Ok(());
        }
        if pose.speed <= options.terminal_stop_speed {
            input.release_all()?;
            let drift = low_speed_anchor
                .map(|anchor| point_distance_2d(anchor, pose.pos))
                .unwrap_or(0.0);
            if low_speed_since.is_none() || drift > 0.5 {
                if low_speed_since.is_some() {
                    roll_resume_count += 1;
                }
                low_speed_since = Some(tick_millis());
                low_speed_anchor = Some(pose.pos);
            }
            let since = low_speed_since.unwrap();
            let held_ms = tick_millis().saturating_sub(since);
            if held_ms >= u128::from(options.terminal_settle_ms) {
                let decision = DriveDecision {
                    keys: Vec::new(),
                    done: true,
                    heading_error: 0.0,
                    reason: format!(
                        "terminal_settled:speed={:.3}:held={}ms",
                        pose.speed, held_ms
                    ),
                };
                log.write_drive("drive_terminal_settled", &[], 0, pose, target, &decision)?;
                return Ok(());
            }
            thread::sleep(Duration::from_millis(options.terminal_brake_step_ms));
            continue;
        } else if low_speed_since.take().is_some() {
            // The kart started rolling again during the release window. The
            // new motion may be in either direction, so probe braking afresh.
            brake_key = Key::Down;
            direction_probed = false;
            best_speed = pose.speed;
            low_speed_anchor = None;
            roll_resume_count += 1;
        } else if terminal_crossed_zero(best_speed, pose.speed) {
            // Speed is an unsigned norm. Once a brake pulse has taken the kart
            // close to rest, a later increase can mean that it crossed zero
            // and is now accelerating in reverse. Guessing the opposite key
            // from speed alone can alternate directions and cause a runaway.
            // Release safely; a future signed longitudinal-velocity model may
            // replace this conservative stop policy.
            input.release_all()?;
            let decision = DriveDecision {
                keys: Vec::new(),
                done: true,
                heading_error: 0.0,
                reason: format!(
                    "terminal_zero_cross_release:speed={:.3}:best={best_speed:.3}",
                    pose.speed
                ),
            };
            log.write_drive(
                "drive_terminal_zero_cross_release",
                &[],
                0,
                pose,
                target,
                &decision,
            )?;
            return Ok(());
        }
        if coast_probe && pose.speed <= 4.0 {
            if !best_speed.is_finite() || pose.speed <= best_speed + 0.08 {
                input.release_all()?;
                best_speed = best_speed.min(pose.speed);
                thread::sleep(Duration::from_millis(options.terminal_brake_step_ms));
                continue;
            }
            coast_probe = false;
            brake_key = Key::Down;
            direction_probed = false;
        } else if pose.speed > 4.0 {
            coast_probe = false;
        }
        let speed_before = pose.speed;
        best_speed = best_speed.min(speed_before);
        let keys = [brake_key];
        let decision = DriveDecision {
            keys: keys.to_vec(),
            done: false,
            heading_error: 0.0,
            reason: format!(
                "terminal_brake:speed={:.3}:key={}",
                pose.speed,
                brake_key.name()
            ),
        };
        input.key_down_many(&keys)?;
        log.write_drive(
            "drive_terminal_brake",
            &keys,
            options.terminal_brake_step_ms,
            pose,
            target,
            &decision,
        )?;
        thread::sleep(Duration::from_millis(options.terminal_brake_step_ms));
        input.key_up_many(&keys)?;
        let speed_after = read_latest_pose(pose_path)?.speed;
        if !direction_probed {
            // At road speed the unsigned speed norm can keep rising for a pulse
            // after Down is pressed because of momentum.  That is not evidence
            // that the kart is reversing, and flipping to Up here accelerates it.
            // Only direction-probe close to rest, where the response is decisive.
            if terminal_brake_should_flip(speed_before, speed_after) {
                brake_key = Key::Up;
            }
            direction_probed = true;
        }
        best_speed = best_speed.min(speed_after);
    }
    input.release_all()?;
    let pose = read_latest_pose(pose_path)?;
    let outcome = terminal_timeout_outcome(roll_resume_count);
    let decision = DriveDecision {
        keys: Vec::new(),
        done: true,
        heading_error: 0.0,
        reason: format!(
            "{outcome}:speed={:.3}:best={:.3}:roll_resumes={roll_resume_count}",
            pose.speed, best_speed
        ),
    };
    let event = if roll_resume_count > 0 {
        "drive_terminal_grade_roll"
    } else {
        "drive_terminal_brake_timeout"
    };
    log.write_drive(event, &[], 0, pose, target, &decision)
}

fn terminal_crossed_zero(best_speed: f32, current_speed: f32) -> bool {
    best_speed.is_finite() && best_speed < 1.0 && current_speed > best_speed + 0.12
}

fn terminal_brake_should_flip(speed_before: f32, speed_after: f32) -> bool {
    speed_before <= 4.0 && speed_after > speed_before + 0.08
}

fn terminal_timeout_outcome(roll_resume_count: u32) -> &'static str {
    if roll_resume_count > 0 {
        "grade_roll_no_static_equilibrium"
    } else {
        "brake_timeout"
    }
}

fn run_script(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("script path is required".to_string());
    }

    let script_path = PathBuf::from(&args[0]);
    let mut log_path = None;
    let mut cursor = FlagCursor::new(args, 1);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--log" => log_path = Some(PathBuf::from(cursor.take_value("--log requires a path")?)),
            other => return Err(format!("unknown script option: {other}")),
        }
        cursor.advance();
    }

    let commands = read_script(&script_path)?;
    let mut log = EventLog::from_optional_path(log_path)?;
    execute(&commands, &mut log)
}

fn parse_keys_duration(args: &[String]) -> Result<(Vec<Key>, u64), String> {
    if args.len() != 2 {
        return Err("expected <keys> <duration_ms>".to_string());
    }
    let keys = parse_keys(&args[0])?;
    let duration_ms = args[1]
        .parse::<u64>()
        .map_err(|_| format!("invalid duration_ms: {}", args[1]))?;
    Ok((keys, duration_ms))
}

fn read_script(path: &PathBuf) -> Result<Vec<Command>, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut commands = Vec::new();

    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("read {}: {err}", path.display()))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let cols: Vec<_> = line.split('\t').collect();
        match cols.as_slice() {
            ["hold", keys, duration_ms] => {
                let duration_ms = duration_ms.parse::<u64>().map_err(|_| {
                    format!("{}:{} invalid duration_ms", path.display(), line_no + 1)
                })?;
                commands.push(Command::Hold {
                    keys: parse_keys(keys)?,
                    duration_ms,
                });
            }
            ["sleep", duration_ms] => {
                let duration_ms = duration_ms.parse::<u64>().map_err(|_| {
                    format!("{}:{} invalid duration_ms", path.display(), line_no + 1)
                })?;
                commands.push(Command::Sleep { duration_ms });
            }
            ["release-all"] => commands.push(Command::ReleaseAll),
            _ => {
                return Err(format!(
                    "{}:{} invalid command row: {line}",
                    path.display(),
                    line_no + 1
                ));
            }
        }
    }

    Ok(commands)
}

fn parse_keys(raw: &str) -> Result<Vec<Key>, String> {
    raw.split('+').map(parse_key).collect()
}

fn parse_key_sequence(raw: &str) -> Result<Vec<KeySequenceStep>, String> {
    let mut steps = Vec::new();
    for (index, entry) in raw.split(',').enumerate() {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (keys_raw, duration_raw) = entry
            .split_once(':')
            .ok_or_else(|| format!("fixed-key-sequence entry {index} must be keys:duration_ms"))?;
        let keys = parse_sequence_keys(keys_raw.trim())?;
        let duration_ms = duration_raw.trim().parse::<u64>().map_err(|_| {
            format!("fixed-key-sequence entry {index} duration must be milliseconds")
        })?;
        if duration_ms == 0 {
            return Err(format!(
                "fixed-key-sequence entry {index} duration must be > 0"
            ));
        }
        steps.push(KeySequenceStep { keys, duration_ms });
    }
    if steps.is_empty() {
        return Err("fixed-key-sequence must contain at least one step".to_string());
    }
    Ok(steps)
}

fn parse_key(raw: &str) -> Result<Key, String> {
    match raw.to_ascii_lowercase().as_str() {
        "w" => Ok(Key::W),
        "a" => Ok(Key::A),
        "s" => Ok(Key::S),
        "d" => Ok(Key::D),
        "v" => Ok(Key::V),
        "up" | "arrowup" => Ok(Key::Up),
        "down" | "arrowdown" => Ok(Key::Down),
        "left" | "arrowleft" => Ok(Key::Left),
        "right" | "arrowright" => Ok(Key::Right),
        "space" => Ok(Key::Space),
        "shift" => Ok(Key::Shift),
        "ctrl" | "control" => Ok(Key::Ctrl),
        "r" => Ok(Key::R),
        "esc" | "escape" => Ok(Key::Esc),
        "tab" => Ok(Key::Tab),
        "enter" | "return" => Ok(Key::Enter),
        "f5" => Ok(Key::F5),
        _ => Err(format!("unknown key: {raw}")),
    }
}

fn parse_sequence_keys(raw: &str) -> Result<Vec<Key>, String> {
    match raw.to_ascii_lowercase().as_str() {
        "" | "none" | "release" | "released" | "off" => Ok(Vec::new()),
        _ => parse_keys(raw),
    }
}

fn active_key_sequence_step(
    steps: &[KeySequenceStep],
    elapsed_ms: u128,
) -> Option<(usize, &KeySequenceStep)> {
    let mut end_ms = 0u128;
    for (index, step) in steps.iter().enumerate() {
        end_ms = end_ms.saturating_add(u128::from(step.duration_ms));
        if elapsed_ms < end_ms {
            return Some((index, step));
        }
    }
    None
}

fn execute(commands: &[Command], log: &mut EventLog) -> Result<(), String> {
    execute_with_backend(InputBackend::SendInput, commands, log)
}

fn execute_with_backend(
    backend: InputBackend,
    commands: &[Command],
    log: &mut EventLog,
) -> Result<(), String> {
    let mut input = InputDriver::new(backend);

    for command in commands {
        match command {
            Command::Hold { keys, duration_ms } => {
                input.key_down_many(keys)?;
                log.write("hold_begin", keys, *duration_ms)?;
                thread::sleep(Duration::from_millis(*duration_ms));
                input.key_up_many(keys)?;
                log.write("hold_end", keys, *duration_ms)?;
            }
            Command::ReleaseAll => {
                input.release_all()?;
                log.write("release_all", &[], 0)?;
            }
            Command::Sleep { duration_ms } => {
                log.write("sleep", &[], *duration_ms)?;
                thread::sleep(Duration::from_millis(*duration_ms));
            }
        }
    }

    input.release_all()
}

struct EventLog {
    writer: Box<dyn Write>,
}

impl EventLog {
    const HEADER: &'static [u8] =
        b"time_ms\tevent\tkeys\tduration_ms\targ0\targ1\targ2\targ3\targ4\targ5\targ6\targ7\targ8\targ9\targ10\targ11\targ12\n";

    fn from_optional_path(path: Option<PathBuf>) -> Result<Self, String> {
        let writer: Box<dyn Write> = match path {
            Some(path) => {
                let needs_header = fs::metadata(&path)
                    .map(|meta| meta.len() == 0)
                    .unwrap_or(true);
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|err| format!("open log {}: {err}", path.display()))?;
                if needs_header {
                    file.write_all(Self::HEADER)
                        .map_err(|err| format!("write log header {}: {err}", path.display()))?;
                }
                Box::new(file)
            }
            None => Box::new(io::sink()),
        };
        Ok(Self { writer })
    }

    fn write(&mut self, event: &str, keys: &[Key], duration_ms: u64) -> Result<(), String> {
        let keys = keys
            .iter()
            .map(|key| key.name())
            .collect::<Vec<_>>()
            .join("+");
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}",
            tick_millis(),
            event,
            keys,
            duration_ms
        )
        .map_err(|err| format!("write control log: {err}"))
    }

    fn write_drive(
        &mut self,
        event: &str,
        keys: &[Key],
        duration_ms: u64,
        pose: Pose,
        target: [f32; 3],
        decision: &DriveDecision,
    ) -> Result<(), String> {
        let keys = keys
            .iter()
            .map(|key| key.name())
            .collect::<Vec<_>>()
            .join("+");
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
            tick_millis(),
            event,
            keys,
            duration_ms,
            pose.pos[0],
            pose.pos[1],
            pose.pos[2],
            pose.yaw,
            pose.speed,
            target[0],
            target[1],
            target[2],
            decision.heading_error,
            decision.reason
        )
        .map_err(|err| format!("write drive control log: {err}"))
    }

    fn write_target(
        &mut self,
        event: &str,
        pose: Pose,
        target: &KapTarget,
        reachability: &str,
    ) -> Result<(), String> {
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            tick_millis(),
            event,
            clean(&target.target_id),
            clean(&target.track_id),
            clean(&target.node),
            clean(&target.road_surface),
            target.surface_id,
            clean(&target.road_rail),
            reachability,
            target.center[0],
            target.center[1],
            target.center[2],
            pose.pos[0],
            pose.pos[1],
            pose.pos[2]
        )
        .map_err(|err| format!("write target log: {err}"))
    }

    fn write_kap_select(
        &mut self,
        event: &str,
        pose: Pose,
        path: &KapFollowPath,
        request: &KapFollowRequest,
    ) -> Result<(), String> {
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            tick_millis(),
            event,
            clean(&path.path_id),
            clean(&path.track_id),
            path.slot.map(|slot| slot.to_string()).unwrap_or_default(),
            path.points.len(),
            path.length,
            bool_text(path.closed),
            request.lookahead,
            request.target_progress,
            pose.pos[0],
            pose.pos[1],
        )
        .map_err(|err| format!("write KAP select log: {err}"))
    }

    fn write_kap_drive(
        &mut self,
        event: &str,
        keys: &[Key],
        duration_ms: u64,
        pose: Pose,
        target: P3,
        decision: &DriveDecision,
        path: &KapFollowPath,
        projection: KapProjection,
    ) -> Result<(), String> {
        let keys = keys
            .iter()
            .map(|key| key.name())
            .collect::<Vec<_>>()
            .join("+");
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
            tick_millis(),
            event,
            keys,
            duration_ms,
            clean(&path.path_id),
            projection.path_s,
            projection.cross_track,
            pose.pos[0],
            pose.pos[1],
            pose.pos[2],
            pose.yaw,
            pose.speed,
            target[0],
            target[1],
            target[2],
            decision.heading_error,
            decision.reason
        )
        .map_err(|err| format!("write KAP drive log: {err}"))
    }
}

#[cfg(not(windows))]
fn tick_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(windows)]
fn tick_millis() -> u128 {
    unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() as u128 }
}

#[derive(Clone, Debug)]
struct KapTarget {
    target_id: String,
    track_id: String,
    node: String,
    road_surface: String,
    surface_id: u32,
    road_rail: String,
    tri_index: usize,
    center: P3,
    normal: P3,
}

#[derive(Clone, Copy, Debug)]
struct Pose {
    pos: P3,
    yaw: f32,
    speed: f32,
    velocity_heading: Option<f32>,
    tick_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct PoseSample {
    tick_ms: u128,
    pose: Pose,
}

#[derive(Clone, Copy, Debug)]
struct GroundDriveFeedback {
    kind40_rows: usize,
    tick0_ms: f32,
    tick1_ms: f32,
    delta_xy: [f32; 2],
    delta_speed: f32,
    accel_along_velocity: f32,
}

#[derive(Clone, Debug)]
struct WallContactFeedback {
    kind: u32,
    kind_label: &'static str,
    site_rva: String,
    tick_ms: Option<f32>,
    hit_normal: [f32; 3],
    hit_dot_vel: Option<f32>,
}

impl WallContactFeedback {
    fn is_response(&self) -> bool {
        matches!(self.kind, 4 | 5 | 8 | 9 | 10)
    }

    fn reason_tag(&self) -> String {
        format!(
            "wall_feedback={}:site={}:tick={:.3}:dot={}:n=({:.3},{:.3},{:.3})",
            self.kind_label,
            self.site_rva,
            self.tick_ms.unwrap_or(0.0),
            self.hit_dot_vel
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "na".to_string()),
            self.hit_normal[0],
            self.hit_normal[1],
            self.hit_normal[2]
        )
    }
}

impl GroundDriveFeedback {
    fn from_kind40_rows(rows: &[GroundDriveKind40Row], pose: Pose) -> Option<Self> {
        if rows.len() < 2 {
            return None;
        }
        let tick0_ms = rows.first()?.tick_ms;
        let tick1_ms = rows.last()?.tick_ms;
        let dt_s = ((tick1_ms - tick0_ms) / 1000.0).max(0.001);
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return None;
        }
        let delta_xy = rows.iter().fold([0.0f32, 0.0f32], |mut acc, row| {
            acc[0] += row.delta[0];
            acc[1] += row.delta[1];
            acc
        });
        let heading = pose.velocity_heading.unwrap_or(pose.yaw);
        let unit = [heading.cos(), heading.sin()];
        let delta_speed = delta_xy[0] * unit[0] + delta_xy[1] * unit[1];
        let accel_along_velocity = (delta_speed / dt_s).clamp(-30.0, 30.0);
        Some(Self {
            kind40_rows: rows.len(),
            tick0_ms,
            tick1_ms,
            delta_xy,
            delta_speed,
            accel_along_velocity,
        })
    }

    fn predicted_speed(
        self,
        model_speed_prediction: f32,
        current_speed: f32,
        horizon_s: f32,
    ) -> f32 {
        if horizon_s <= 0.0 || !self.accel_along_velocity.is_finite() {
            return model_speed_prediction;
        }
        let physics_prediction = (current_speed + self.accel_along_velocity * horizon_s).max(0.0);
        (0.5 * model_speed_prediction + 0.5 * physics_prediction).max(0.0)
    }

    fn reason_fragment(self) -> String {
        format!(
            "phys=kind40:rows={}:dt={:.0}ms:dv={:.4}:acc={:.3}:dxy={:.4},{:.4}",
            self.kind40_rows,
            self.tick1_ms - self.tick0_ms,
            self.delta_speed,
            self.accel_along_velocity,
            self.delta_xy[0],
            self.delta_xy[1]
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct GroundDriveKind40Row {
    tick_ms: f32,
    delta: [f32; 3],
}

#[derive(Clone, Debug)]
struct PhysicsBaseline {
    version: String,
    low_speed_near_zero_threshold: f32,
    low_speed_recovery_max_speed: f32,
    low_speed_timeout_min_ms: u64,
    low_speed_timeout_max_ms: u64,
    wall_risk_cross_track_scale: f32,
    wall_risk_heading_error_rad: f32,
    wall_risk_speed_floor: f32,
    mpc_wall_risk_throttle_penalty: f32,
    coast_lowproj_k_lt2: f32,
    coast_lowproj_k_2_5: f32,
    coast_highproj_k_5_10: f32,
    coast_highproj_k_10_20: f32,
    coast_highproj_k_20p: f32,
}

impl PhysicsBaseline {
    fn default_closed_clean() -> Self {
        Self {
            version: "builtin_clean_baseline".to_string(),
            low_speed_near_zero_threshold: 0.05,
            low_speed_recovery_max_speed: 0.7,
            low_speed_timeout_min_ms: 500,
            low_speed_timeout_max_ms: 1200,
            wall_risk_cross_track_scale: 0.85,
            wall_risk_heading_error_rad: 1.0,
            wall_risk_speed_floor: 2.0,
            mpc_wall_risk_throttle_penalty: 0.45,
            coast_lowproj_k_lt2: 0.0415146991,
            coast_lowproj_k_2_5: 0.059738974,
            coast_highproj_k_5_10: 0.0891539334,
            coast_highproj_k_10_20: 0.15138862,
            coast_highproj_k_20p: 0.202590069,
        }
    }

    fn from_tsv(path: &Path) -> Result<Self, String> {
        let mut baseline = Self::default_closed_clean();
        let file = File::open(path)
            .map_err(|err| format!("open physics baseline {}: {err}", path.display()))?;
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line =
                line.map_err(|err| format!("read physics baseline {}: {err}", path.display()))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<_> = line.split('\t').collect();
            if cols.len() != 2 {
                return Err(format!(
                    "{}:{} expected key<TAB>value",
                    path.display(),
                    line_no + 1
                ));
            }
            match cols[0] {
                "version" => baseline.version = cols[1].to_string(),
                "low_speed_near_zero_threshold" => {
                    baseline.low_speed_near_zero_threshold =
                        parse_baseline_f32(path, line_no, cols[1])?
                }
                "low_speed_recovery_max_speed" => {
                    baseline.low_speed_recovery_max_speed =
                        parse_baseline_f32(path, line_no, cols[1])?
                }
                "low_speed_timeout_min_ms" => {
                    baseline.low_speed_timeout_min_ms = parse_baseline_u64(path, line_no, cols[1])?
                }
                "low_speed_timeout_max_ms" => {
                    baseline.low_speed_timeout_max_ms = parse_baseline_u64(path, line_no, cols[1])?
                }
                "wall_risk_cross_track_scale" => {
                    baseline.wall_risk_cross_track_scale =
                        parse_baseline_f32(path, line_no, cols[1])?
                }
                "wall_risk_heading_error_rad" => {
                    baseline.wall_risk_heading_error_rad =
                        parse_baseline_f32(path, line_no, cols[1])?
                }
                "wall_risk_speed_floor" => {
                    baseline.wall_risk_speed_floor = parse_baseline_f32(path, line_no, cols[1])?
                }
                "mpc_wall_risk_throttle_penalty" => {
                    baseline.mpc_wall_risk_throttle_penalty =
                        parse_baseline_f32(path, line_no, cols[1])?
                }
                "coast_lowproj_k_lt2" => {
                    baseline.coast_lowproj_k_lt2 = parse_baseline_f32(path, line_no, cols[1])?
                }
                "coast_lowproj_k_2_5" => {
                    baseline.coast_lowproj_k_2_5 = parse_baseline_f32(path, line_no, cols[1])?
                }
                "coast_highproj_k_5_10" => {
                    baseline.coast_highproj_k_5_10 = parse_baseline_f32(path, line_no, cols[1])?
                }
                "coast_highproj_k_10_20" => {
                    baseline.coast_highproj_k_10_20 = parse_baseline_f32(path, line_no, cols[1])?
                }
                "coast_highproj_k_20p" => {
                    baseline.coast_highproj_k_20p = parse_baseline_f32(path, line_no, cols[1])?
                }
                _ => {}
            }
        }
        Ok(baseline)
    }

    fn coast_decel_for_speed(&self, speed: f32) -> f32 {
        if speed < 0.2 {
            return 0.0;
        }
        let k = if speed < 2.0 {
            self.coast_lowproj_k_lt2
        } else if speed < 5.0 {
            self.coast_lowproj_k_2_5
        } else if speed < 10.0 {
            self.coast_highproj_k_5_10
        } else if speed < 20.0 {
            self.coast_highproj_k_10_20
        } else {
            self.coast_highproj_k_20p
        };
        (k * speed).max(0.0)
    }

    fn timeout_threshold_ms(&self, requested_timeout_ms: u64) -> u128 {
        let half = requested_timeout_ms / 2;
        let clamped = half.clamp(self.low_speed_timeout_min_ms, self.low_speed_timeout_max_ms);
        u128::from(clamped)
    }

    fn wall_risk_active(
        &self,
        projection: KapProjection,
        request: &KapFollowRequest,
        speed: f32,
        heading_error: f32,
    ) -> bool {
        if request.safe_cross_track <= 0.0 || speed < self.wall_risk_speed_floor {
            return false;
        }
        let threshold = request.safe_cross_track * self.wall_risk_cross_track_scale.max(0.0);
        projection.cross_track.abs() >= threshold
            && heading_error.abs() >= self.wall_risk_heading_error_rad
    }

    fn reason_fragment(&self) -> String {
        format!("baseline={}", self.version)
    }
}

fn parse_baseline_f32(path: &Path, line_no: usize, value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("{}:{} invalid baseline float", path.display(), line_no + 1))
}

fn parse_baseline_u64(path: &Path, line_no: usize, value: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| {
        format!(
            "{}:{} invalid baseline integer",
            path.display(),
            line_no + 1
        )
    })
}

#[derive(Clone, Debug)]
struct ControlSample {
    tick_ms: u128,
    event: String,
    keys: Vec<Key>,
}

#[derive(Clone, Copy, Debug)]
struct VehicleModel {
    max_control_speed: f32,
    hard_turn_rad: f32,
    soft_turn_rad: f32,
    forward_accel: f32,
    forward_accel_stationary: f32,
    forward_accel_low: f32,
    forward_accel_low_mid: f32,
    coast_decel: f32,
    brake_decel: f32,
    down_brake_min_speed: f32,
    down_low_speed_response_accel: f32,
    brake_decel_low_mid: f32,
    brake_decel_mid: f32,
    turn_rate_low: f32,
    turn_rate_low_mid: f32,
    turn_rate_mid: f32,
    turn_rate_high: f32,
    combo_left_accel_low: f32,
    combo_left_accel_low_mid: f32,
    combo_right_accel_low: f32,
    combo_right_accel_low_mid: f32,
    combo_left_turn_rate_low: f32,
    combo_left_turn_rate_low_mid: f32,
    combo_right_turn_rate_low: f32,
    combo_right_turn_rate_low_mid: f32,
}

#[derive(Clone, Copy, Debug)]
enum DownEffect {
    ReverseOrSettle,
    Brake { decel: f32 },
}

impl VehicleModel {
    fn low_speed_default() -> Self {
        Self {
            max_control_speed: 18.0,
            hard_turn_rad: 35.0_f32.to_radians(),
            soft_turn_rad: 10.0_f32.to_radians(),
            forward_accel: 10.0,
            forward_accel_stationary: 10.0,
            forward_accel_low: 10.0,
            forward_accel_low_mid: 10.0,
            coast_decel: 2.0,
            brake_decel: 18.0,
            // Runtime factory_I02 evidence shows Down can select reverse
            // rather than braking throughout the old 2--10 m/s band. Keep
            // braking conservative until the exact static selector threshold
            // is recovered.
            down_brake_min_speed: 12.0,
            down_low_speed_response_accel: 0.0,
            brake_decel_low_mid: 18.0,
            brake_decel_mid: 18.0,
            turn_rate_low: 1.1,
            turn_rate_low_mid: 0.8,
            turn_rate_mid: 0.7,
            turn_rate_high: 0.7,
            combo_left_accel_low: 10.0,
            combo_left_accel_low_mid: 10.0,
            combo_right_accel_low: 10.0,
            combo_right_accel_low_mid: 10.0,
            combo_left_turn_rate_low: 1.1,
            combo_left_turn_rate_low_mid: 0.8,
            combo_right_turn_rate_low: 1.1,
            combo_right_turn_rate_low_mid: 0.8,
        }
    }

    fn turn_rate_for_speed(&self, speed: f32) -> f32 {
        if speed < 2.0 {
            self.turn_rate_low
        } else if speed < 5.0 {
            self.turn_rate_low_mid
        } else if speed < 10.0 {
            self.turn_rate_mid
        } else {
            self.turn_rate_high
        }
    }

    fn forward_accel_for_speed(&self, speed: f32) -> f32 {
        if speed < 0.2 {
            self.forward_accel_stationary
        } else if speed < 2.0 {
            self.forward_accel_low
        } else if speed < 5.0 {
            self.forward_accel_low_mid
        } else {
            self.forward_accel
        }
    }

    fn predicted_speed_after_throttle(&self, speed: f32, horizon_s: f32) -> f32 {
        (speed + self.forward_accel_for_speed(speed) * horizon_s).max(0.0)
    }

    fn combo_accel_for_speed(&self, turn_key: Key, speed: f32) -> f32 {
        match turn_key {
            Key::Left if speed < 2.0 => self.combo_left_accel_low,
            Key::Left if speed < 5.0 => self.combo_left_accel_low_mid,
            Key::Right if speed < 2.0 => self.combo_right_accel_low,
            Key::Right if speed < 5.0 => self.combo_right_accel_low_mid,
            _ => self.forward_accel_for_speed(speed),
        }
    }

    fn combo_turn_rate_for_speed(&self, turn_key: Key, speed: f32) -> f32 {
        match turn_key {
            Key::Left if speed < 2.0 => self.combo_left_turn_rate_low,
            Key::Left if speed < 5.0 => self.combo_left_turn_rate_low_mid,
            Key::Right if speed < 2.0 => self.combo_right_turn_rate_low,
            Key::Right if speed < 5.0 => self.combo_right_turn_rate_low_mid,
            _ => self.turn_rate_for_speed(speed),
        }
    }

    fn predicted_speed_after_combo_throttle(
        &self,
        turn_key: Key,
        speed: f32,
        horizon_s: f32,
    ) -> f32 {
        (speed + self.combo_accel_for_speed(turn_key, speed) * horizon_s).max(0.0)
    }

    fn predicted_speed_after_coast(
        &self,
        speed: f32,
        horizon_s: f32,
        physics_baseline: Option<&PhysicsBaseline>,
    ) -> f32 {
        let decel = physics_baseline
            .map(|baseline| baseline.coast_decel_for_speed(speed))
            .unwrap_or(self.coast_decel);
        (speed - decel * horizon_s).max(0.0)
    }

    fn down_effect_for_speed(&self, speed: f32) -> DownEffect {
        if speed < self.down_brake_min_speed {
            DownEffect::ReverseOrSettle
        } else if speed < 10.0 {
            DownEffect::Brake {
                decel: self.brake_decel_low_mid,
            }
        } else {
            DownEffect::Brake {
                decel: self.brake_decel_mid,
            }
        }
    }

    fn from_tsv(path: &Path) -> Result<Self, String> {
        let mut model = Self::low_speed_default();
        let file =
            File::open(path).map_err(|err| format!("open model {}: {err}", path.display()))?;
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|err| format!("read model {}: {err}", path.display()))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<_> = line.split('\t').collect();
            if cols.len() != 2 {
                return Err(format!(
                    "{}:{} expected key<TAB>value",
                    path.display(),
                    line_no + 1
                ));
            }
            let value = cols[1]
                .parse::<f32>()
                .map_err(|_| format!("{}:{} invalid value", path.display(), line_no + 1))?;
            match cols[0] {
                "max_control_speed" => model.max_control_speed = value,
                "hard_turn_deg" => model.hard_turn_rad = value.to_radians(),
                "soft_turn_deg" => model.soft_turn_rad = value.to_radians(),
                "forward_accel" => model.forward_accel = value,
                "forward_accel_stationary" => model.forward_accel_stationary = value,
                "forward_accel_low" => model.forward_accel_low = value,
                "forward_accel_low_mid" => model.forward_accel_low_mid = value,
                "coast_decel" => model.coast_decel = value,
                "brake_decel" => model.brake_decel = value,
                "down_brake_min_speed" => model.down_brake_min_speed = value,
                "down_low_speed_response_accel" => model.down_low_speed_response_accel = value,
                "brake_decel_low_mid" => model.brake_decel_low_mid = value,
                "brake_decel_mid" => model.brake_decel_mid = value,
                "turn_rate_low" => model.turn_rate_low = value,
                "turn_rate_low_mid" => model.turn_rate_low_mid = value,
                "turn_rate_mid" => model.turn_rate_mid = value,
                "turn_rate_high" => model.turn_rate_high = value,
                "combo_left_accel_low" => model.combo_left_accel_low = value,
                "combo_left_accel_low_mid" => model.combo_left_accel_low_mid = value,
                "combo_right_accel_low" => model.combo_right_accel_low = value,
                "combo_right_accel_low_mid" => model.combo_right_accel_low_mid = value,
                "combo_left_turn_rate_low" => model.combo_left_turn_rate_low = value,
                "combo_left_turn_rate_low_mid" => model.combo_left_turn_rate_low_mid = value,
                "combo_right_turn_rate_low" => model.combo_right_turn_rate_low = value,
                "combo_right_turn_rate_low_mid" => model.combo_right_turn_rate_low_mid = value,
                _ => {}
            }
        }
        Ok(model)
    }

    fn serialize_tsv(&self) -> String {
        format!(
            "# kart-autoprobe vehicle-model v1\n\
max_control_speed\t{:.6}\n\
hard_turn_deg\t{:.6}\n\
soft_turn_deg\t{:.6}\n\
forward_accel\t{:.6}\n\
forward_accel_stationary\t{:.6}\n\
forward_accel_low\t{:.6}\n\
forward_accel_low_mid\t{:.6}\n\
coast_decel\t{:.6}\n\
brake_decel\t{:.6}\n\
down_brake_min_speed\t{:.6}\n\
down_low_speed_response_accel\t{:.6}\n\
brake_decel_low_mid\t{:.6}\n\
brake_decel_mid\t{:.6}\n\
turn_rate_low\t{:.6}\n\
turn_rate_low_mid\t{:.6}\n\
turn_rate_mid\t{:.6}\n\
turn_rate_high\t{:.6}\n\
combo_left_accel_low\t{:.6}\n\
combo_left_accel_low_mid\t{:.6}\n\
combo_right_accel_low\t{:.6}\n\
combo_right_accel_low_mid\t{:.6}\n\
combo_left_turn_rate_low\t{:.6}\n\
combo_left_turn_rate_low_mid\t{:.6}\n\
combo_right_turn_rate_low\t{:.6}\n\
combo_right_turn_rate_low_mid\t{:.6}\n",
            self.max_control_speed,
            self.hard_turn_rad.to_degrees(),
            self.soft_turn_rad.to_degrees(),
            self.forward_accel,
            self.forward_accel_stationary,
            self.forward_accel_low,
            self.forward_accel_low_mid,
            self.coast_decel,
            self.brake_decel,
            self.down_brake_min_speed,
            self.down_low_speed_response_accel,
            self.brake_decel_low_mid,
            self.brake_decel_mid,
            self.turn_rate_low,
            self.turn_rate_low_mid,
            self.turn_rate_mid,
            self.turn_rate_high,
            self.combo_left_accel_low,
            self.combo_left_accel_low_mid,
            self.combo_right_accel_low,
            self.combo_right_accel_low_mid,
            self.combo_left_turn_rate_low,
            self.combo_left_turn_rate_low_mid,
            self.combo_right_turn_rate_low,
            self.combo_right_turn_rate_low_mid
        )
    }
}

#[derive(Clone, Debug)]
struct DriveDecision {
    keys: Vec<Key>,
    done: bool,
    heading_error: f32,
    reason: String,
}

fn apply_fixed_add_keys(decision: &mut DriveDecision, fixed_add_keys: &[Key]) {
    if fixed_add_keys.is_empty() {
        return;
    }
    if fixed_add_keys.contains(&Key::Up) {
        decision.keys.retain(|key| *key != Key::Down);
    }
    if fixed_add_keys.contains(&Key::Down) {
        decision.keys.retain(|key| *key != Key::Up);
    }
    for key in fixed_add_keys {
        if !decision.keys.contains(key) {
            decision.keys.push(*key);
        }
    }
    let fixed = fixed_add_keys
        .iter()
        .map(|key| key.name())
        .collect::<Vec<_>>()
        .join("+");
    decision.reason = format!("{}:fixed_add_keys={fixed}", decision.reason);
}

#[derive(Clone, Debug)]
struct DriveOptions {
    model_path: Option<PathBuf>,
    physics_baseline_path: Option<PathBuf>,
    log_path: Option<PathBuf>,
    collision_path: Option<PathBuf>,
    wall_hard_path: Option<PathBuf>,
    cancel_path: Option<PathBuf>,
    heading_calibration_path: Option<PathBuf>,
    control_mode: ControlMode,
    input_backend: InputBackend,
    heading_mode: HeadingMode,
    stop_on_surface: bool,
    stop_surface_id: Option<u32>,
    stop_state_kind: Option<u32>,
    timeout_ok: bool,
    max_ms: u64,
    step_ms: u64,
    stop_distance: f32,
    stuck_timeout_ms: u64,
    unstuck_reverse_ms: u64,
    max_unstuck_attempts: u32,
    terminal_brake_ms: u64,
    terminal_brake_step_ms: u64,
    terminal_stop_speed: f32,
    terminal_settle_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlMode {
    Legacy,
    Mpc,
}

impl ControlMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "legacy" | "" => Ok(Self::Legacy),
            "mpc" | "one-step-mpc" | "one_step_mpc" => Ok(Self::Mpc),
            other => Err(format!("unknown control mode: {other}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Mpc => "mpc",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeadingMode {
    Auto,
    PoseYaw,
    Velocity,
}

impl HeadingMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" | "" => Ok(Self::Auto),
            "pose-yaw" | "pose_yaw" => Ok(Self::PoseYaw),
            "velocity" => Ok(Self::Velocity),
            other => Err(format!("unknown heading mode: {other}")),
        }
    }
}

#[derive(Clone, Debug)]
struct KeySequenceStep {
    keys: Vec<Key>,
    duration_ms: u64,
}

impl KeySequenceStep {
    fn key_name(&self) -> String {
        if self.keys.is_empty() {
            "None".to_string()
        } else {
            self.keys
                .iter()
                .map(|key| key.name())
                .collect::<Vec<_>>()
                .join("+")
        }
    }
}

#[derive(Clone, Debug)]
struct KapFollowRequest {
    kap_dir: Option<PathBuf>,
    track_id: Option<String>,
    slot: Option<usize>,
    lookahead: f32,
    max_cross_track: f32,
    target_progress: f32,
    start_path_s: Option<f32>,
    start_window_m: f32,
    lateral_offset: f32,
    offset_start_s: Option<f32>,
    offset_end_s: Option<f32>,
    offset_ramp_m: f32,
    lateral_offset_zones: Vec<KapLateralOffsetZone>,
    heading_mode: HeadingMode,
    speed_limit: f32,
    speed_zones: Vec<(f32, f32)>,
    gap_transfer: bool,
    abort_above_speed: f32,
    abort_below_speed: f32,
    abort_below_speed_after_ms: u64,
    longitudinal_cooldown_ms: u64,
    fixed_keys: Vec<Key>,
    fixed_add_keys: Vec<Key>,
    fixed_key_sequence: Vec<KeySequenceStep>,
    fixed_add_key_sequence: Vec<KeySequenceStep>,
    safe_cross_track: f32,
    panic_cross_track: f32,
    stuck_timeout_ms: u64,
    cross_track_timeout_ms: u64,
}

impl Default for KapFollowRequest {
    fn default() -> Self {
        Self {
            kap_dir: None,
            track_id: None,
            slot: None,
            lookahead: 12.0,
            max_cross_track: 35.0,
            target_progress: 0.0,
            start_path_s: None,
            start_window_m: 80.0,
            lateral_offset: 0.0,
            offset_start_s: None,
            offset_end_s: None,
            offset_ramp_m: 20.0,
            lateral_offset_zones: Vec::new(),
            heading_mode: HeadingMode::Auto,
            speed_limit: 8.0,
            speed_zones: Vec::new(),
            gap_transfer: false,
            abort_above_speed: 0.0,
            abort_below_speed: 0.0,
            abort_below_speed_after_ms: 0,
            longitudinal_cooldown_ms: 100,
            fixed_keys: Vec::new(),
            fixed_add_keys: Vec::new(),
            fixed_key_sequence: Vec::new(),
            fixed_add_key_sequence: Vec::new(),
            safe_cross_track: 6.0,
            panic_cross_track: 12.0,
            stuck_timeout_ms: 1500,
            cross_track_timeout_ms: 15_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct KapLateralOffsetZone {
    start_s: f32,
    end_s: f32,
    offset: f32,
    ramp_m: f32,
}

fn lateral_offset_in_window(
    path: &KapFollowPath,
    path_s: f32,
    start: f32,
    end: f32,
    offset: f32,
    ramp_m: f32,
) -> f32 {
    if offset.abs() <= f32::EPSILON {
        return 0.0;
    }
    let span = path.forward_distance(start, end);
    let position = path.forward_distance(start, path_s.rem_euclid(path.length));
    if !span.is_finite() || !position.is_finite() || position > span {
        return 0.0;
    }
    let ramp = ramp_m.max(0.0).min(span * 0.5);
    let gain = if ramp <= f32::EPSILON {
        1.0
    } else {
        (position / ramp)
            .min((span - position) / ramp)
            .clamp(0.0, 1.0)
    };
    offset * gain
}

fn lateral_offset_at(path: &KapFollowPath, path_s: f32, request: &KapFollowRequest) -> f32 {
    if !request.lateral_offset_zones.is_empty() {
        return request
            .lateral_offset_zones
            .iter()
            .map(|zone| {
                lateral_offset_in_window(
                    path,
                    path_s,
                    zone.start_s,
                    zone.end_s,
                    zone.offset,
                    zone.ramp_m,
                )
            })
            .sum();
    }
    if request.lateral_offset.abs() <= f32::EPSILON {
        return 0.0;
    }
    let (Some(start), Some(end)) = (request.offset_start_s, request.offset_end_s) else {
        return request.lateral_offset;
    };
    lateral_offset_in_window(
        path,
        path_s,
        start,
        end,
        request.lateral_offset,
        request.offset_ramp_m,
    )
}

fn kap_control_cross_track(
    path: &KapFollowPath,
    pose: P3,
    path_s: f32,
    centerline_cross_track: f32,
    request: &KapFollowRequest,
) -> f32 {
    let offset = lateral_offset_at(path, path_s, request);
    if offset.abs() <= f32::EPSILON {
        return centerline_cross_track;
    }
    path.point_at_lateral_offset(path_s, offset)
        .map(|point| point_distance_weighted(pose, point))
        .unwrap_or(centerline_cross_track)
}

fn adaptive_kap_lookahead(path: &KapFollowPath, path_s: f32, requested: f32) -> f32 {
    let requested = requested.max(2.0);
    let Some(origin) = path.point_at(path_s) else {
        return requested;
    };
    let Some(origin_heading) = path.heading_at(path_s) else {
        return requested;
    };
    let mut max_heading_delta = 0.0f32;
    let mut max_elevation_delta = 0.0f32;
    // Target distance and hazard-detection horizon are separate contracts.
    // Using a short requested lookahead as the scan horizon made lookahead=4
    // blind to factory_I02 hairpins until the kart was already inside them;
    // the old 4 m output clamp then also disabled the adaptive speed guard.
    // Keep the steering target local while looking far enough ahead to brake
    // before authored curvature/elevation transitions.
    let detection_horizon = requested.max(24.0);
    for fraction in [0.25f32, 0.50, 0.75, 1.0] {
        let sample_s = path_s + detection_horizon * fraction;
        if let Some(heading) = path.heading_at(sample_s) {
            max_heading_delta =
                max_heading_delta.max(normalize_angle(heading - origin_heading).abs());
        }
        if let Some(point) = path.point_at(sample_s) {
            max_elevation_delta = max_elevation_delta.max((point[2] - origin[2]).abs());
        }
    }
    let scale = if max_heading_delta >= 0.75 || max_elevation_delta >= 2.0 {
        0.30
    } else if max_heading_delta >= 0.40 || max_elevation_delta >= 1.0 {
        0.50
    } else if max_heading_delta >= 0.20 || max_elevation_delta >= 0.5 {
        0.72
    } else {
        1.0
    };
    (requested * scale).clamp(2.0, requested)
}

#[derive(Clone, Debug, Default)]
struct KapTopologyProfile {
    gaps: Vec<KapGapWindow>,
}

impl KapTopologyProfile {
    fn from_path(path: &KapFollowPath, nav_mesh: &NavMesh) -> Self {
        const SUPPORT_DISTANCE_M: f32 = 2.5;
        Self {
            gaps: kap_gap_windows(&path.points, |point| {
                nav_mesh
                    .support_distance(point)
                    .map(|distance| distance <= SUPPORT_DISTANCE_M)
                    .unwrap_or(false)
            }),
        }
    }

    fn active_gap(&self, path_s: f32) -> Option<&KapGapWindow> {
        self.gaps
            .iter()
            .find(|gap| path_s >= gap.approach_start_s && path_s <= gap.exit_s + 5.0)
    }
}

#[derive(Clone, Debug)]
struct KapFollowPath {
    path_id: String,
    track_id: String,
    slot: Option<usize>,
    points: Vec<P3>,
    cumulative: Vec<f32>,
    length: f32,
    closed: bool,
}

#[derive(Clone, Debug)]
struct KapRouteMineOptions {
    window_m: f32,
    step_m: f32,
    max_heading_change_rad: f32,
    min_score: f32,
    limit: usize,
}

impl Default for KapRouteMineOptions {
    fn default() -> Self {
        Self {
            window_m: 35.0,
            step_m: 5.0,
            max_heading_change_rad: 0.45,
            min_score: 0.55,
            limit: 50,
        }
    }
}

#[derive(Clone, Debug)]
struct KapRouteWindow {
    track_id: String,
    slot: Option<usize>,
    path_id: String,
    start_s: f32,
    end_s: f32,
    center_s: f32,
    length_m: f32,
    progress_start: f32,
    progress_end: f32,
    heading_change_rad: f32,
    max_segment_heading_delta_rad: f32,
    straightness: f32,
    grade_change: f32,
    grade_abs_max: f32,
    score: f32,
    kind: &'static str,
    verdict: &'static str,
    start: P3,
    end: P3,
}

#[derive(Clone, Copy, Debug)]
struct KapProjection {
    path_s: f32,
    cross_track: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputBackend {
    SendInput,
    PostMessage,
    InternalSetters,
}

impl InputBackend {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.to_ascii_lowercase().as_str() {
            "sendinput" | "send-input" | "foreground" => Ok(Self::SendInput),
            "postmessage" | "post-message" | "window-message" | "wm" => Ok(Self::PostMessage),
            "internal" | "internal-setters" | "engine-setters" | "goplaykart-setters" => {
                Ok(Self::InternalSetters)
            }
            _ => Err(format!("unknown input backend: {raw}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::SendInput => "sendinput",
            Self::PostMessage => "postmessage",
            Self::InternalSetters => "internal-setters",
        }
    }

    fn requires_foreground(self) -> bool {
        matches!(self, Self::SendInput)
    }
}

fn parse_input_backend_param(params: &BTreeMap<String, String>) -> Result<InputBackend, String> {
    match params
        .get("backend")
        .or_else(|| params.get("input_backend"))
        .map(String::as_str)
    {
        Some(raw) => InputBackend::parse(raw),
        None => Ok(InputBackend::SendInput),
    }
}

impl Default for DriveOptions {
    fn default() -> Self {
        Self {
            model_path: None,
            physics_baseline_path: None,
            log_path: None,
            collision_path: None,
            wall_hard_path: None,
            cancel_path: None,
            heading_calibration_path: None,
            control_mode: ControlMode::Legacy,
            input_backend: InputBackend::SendInput,
            heading_mode: HeadingMode::Auto,
            stop_on_surface: false,
            stop_surface_id: None,
            stop_state_kind: None,
            timeout_ok: false,
            max_ms: 15_000,
            step_ms: 160,
            stop_distance: 3.0,
            stuck_timeout_ms: 1500,
            unstuck_reverse_ms: 800,
            max_unstuck_attempts: 4,
            terminal_brake_ms: 15_000,
            terminal_brake_step_ms: 80,
            terminal_stop_speed: 0.75,
            terminal_settle_ms: 3000,
        }
    }
}

fn parse_drive_options(args: &[String], command: &str) -> Result<DriveOptions, String> {
    let mut options = DriveOptions::default();
    let mut cursor = FlagCursor::new(args, 0);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--log" => {
                options.log_path = Some(PathBuf::from(cursor.take_value("--log requires a path")?))
            }
            "--model" => {
                options.model_path =
                    Some(PathBuf::from(cursor.take_value("--model requires a path")?))
            }
            "--physics-baseline" => {
                options.physics_baseline_path = Some(PathBuf::from(
                    cursor.take_value("--physics-baseline requires a path")?,
                ))
            }
            "--control-mode" => {
                options.control_mode =
                    ControlMode::parse(cursor.take_value("--control-mode requires a value")?)?
            }
            "--collision" => {
                options.collision_path = Some(PathBuf::from(
                    cursor.take_value("--collision requires a path")?,
                ))
            }
            "--wall-hard" => {
                options.wall_hard_path = Some(PathBuf::from(
                    cursor.take_value("--wall-hard requires a path")?,
                ))
            }
            "--cancel-file" => {
                options.cancel_path = Some(PathBuf::from(
                    cursor.take_value("--cancel-file requires a path")?,
                ))
            }
            "--heading-calibration" => {
                options.heading_calibration_path = Some(PathBuf::from(
                    cursor.take_value("--heading-calibration requires a path")?,
                ))
            }
            "--input-backend" => {
                options.input_backend =
                    InputBackend::parse(cursor.take_value("--input-backend requires a value")?)?
            }
            "--heading-mode" => {
                options.heading_mode =
                    HeadingMode::parse(cursor.take_value("--heading-mode requires a value")?)?
            }
            "--stop-on-surface" => {
                options.stop_on_surface = true;
            }
            "--timeout-ok" => {
                options.timeout_ok = true;
            }
            "--stop-on-surface-id" => {
                let value = cursor.take_value("--stop-on-surface-id requires a value")?;
                options.stop_surface_id = Some(parse_u32_arg(value, "surface_id")?);
            }
            "--stop-on-state-kind" => {
                let value = cursor.take_value("--stop-on-state-kind requires a value")?;
                options.stop_state_kind = Some(parse_u32_arg(value, "state kind")?);
            }
            "--max-ms" => {
                options.max_ms = cursor
                    .take_value("--max-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--max-ms must be an integer".to_string())?;
            }
            "--step-ms" => {
                options.step_ms = cursor
                    .take_value("--step-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--step-ms must be an integer".to_string())?;
            }
            "--stop-distance" => {
                options.stop_distance = cursor
                    .take_value("--stop-distance requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--stop-distance must be a number".to_string())?;
            }
            "--stuck-timeout-ms" => {
                options.stuck_timeout_ms = cursor
                    .take_value("--stuck-timeout-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--stuck-timeout-ms must be an integer".to_string())?;
            }
            "--unstuck-reverse-ms" => {
                options.unstuck_reverse_ms = cursor
                    .take_value("--unstuck-reverse-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--unstuck-reverse-ms must be an integer".to_string())?;
            }
            "--max-unstuck-attempts" => {
                options.max_unstuck_attempts = cursor
                    .take_value("--max-unstuck-attempts requires a value")?
                    .parse::<u32>()
                    .map_err(|_| "--max-unstuck-attempts must be an integer".to_string())?;
            }
            "--terminal-brake-ms" => {
                options.terminal_brake_ms = cursor
                    .take_value("--terminal-brake-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--terminal-brake-ms must be an integer".to_string())?;
            }
            "--terminal-brake-step-ms" => {
                options.terminal_brake_step_ms = cursor
                    .take_value("--terminal-brake-step-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--terminal-brake-step-ms must be an integer".to_string())?;
                if options.terminal_brake_step_ms == 0 {
                    return Err("--terminal-brake-step-ms must be positive".to_string());
                }
            }
            "--terminal-stop-speed" => {
                options.terminal_stop_speed = cursor
                    .take_value("--terminal-stop-speed requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--terminal-stop-speed must be a number".to_string())?;
            }
            "--terminal-settle-ms" => {
                options.terminal_settle_ms = cursor
                    .take_value("--terminal-settle-ms requires a value")?
                    .parse::<u64>()
                    .map_err(|_| "--terminal-settle-ms must be an integer".to_string())?;
            }
            other => return Err(format!("unknown {command} option: {other}")),
        }
        cursor.advance();
    }
    Ok(options)
}

fn choose_target<'a>(
    targets: &'a [KapTarget],
    pose: Pose,
    nav_mesh: Option<&NavMesh>,
) -> Option<&'a KapTarget> {
    let pose_component = nav_mesh.and_then(|mesh| mesh.nearest_component(pose.pos));
    let reachable = |target: &&KapTarget| -> bool {
        let Some(mesh) = nav_mesh else { return true };
        let Some(pose_component) = pose_component else {
            return true;
        };
        mesh.nearest_component(target.center) == Some(pose_component)
    };

    targets
        .iter()
        .filter(reachable)
        .min_by(|a, b| {
            target_score(a, pose, nav_mesh)
                .partial_cmp(&target_score(b, pose, nav_mesh))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .or_else(|| {
            targets.iter().min_by(|a, b| {
                target_score(a, pose, nav_mesh)
                    .partial_cmp(&target_score(b, pose, nav_mesh))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
}

fn is_non_drivable_surface(surface: Option<&str>) -> bool {
    matches!(
        surface,
        Some("리셋" | "reset" | "pit" | "transparent" | "transparency")
    )
}

fn fit_vehicle_model(poses: &[PoseSample], controls: &[ControlSample]) -> VehicleModel {
    let base = VehicleModel::low_speed_default();
    let forward_accel = fit_accel_for_key(poses, controls, Key::Up).unwrap_or(base.forward_accel);
    let brake_decel = fit_decel_for_key(poses, controls, Key::Down).unwrap_or(base.brake_decel);
    let coast_decel = fit_coast_decel(poses, controls).unwrap_or(base.coast_decel);
    let turn_left = fit_turn_rate_for_keys(poses, controls, &[Key::Up, Key::Left]);
    let turn_right = fit_turn_rate_for_keys(poses, controls, &[Key::Up, Key::Right]);
    let turn_rate_mid = mean_present(&[turn_left, turn_right]).unwrap_or(base.turn_rate_mid);
    let turn_rate_low = (turn_rate_mid * 1.35).max(base.turn_rate_low * 0.5);
    let turn_rate_low_mid = turn_rate_mid;
    let turn_rate_high = turn_rate_mid;
    let observed_max = poses
        .iter()
        .map(|sample| sample.pose.speed)
        .fold(0.0_f32, f32::max);
    let max_control_speed = if observed_max > 1.0 {
        (observed_max * 0.7).clamp(8.0, 35.0)
    } else {
        base.max_control_speed
    };

    VehicleModel {
        max_control_speed,
        hard_turn_rad: base.hard_turn_rad,
        soft_turn_rad: base.soft_turn_rad,
        forward_accel,
        forward_accel_stationary: forward_accel,
        forward_accel_low: forward_accel,
        forward_accel_low_mid: forward_accel,
        coast_decel,
        brake_decel,
        down_brake_min_speed: base.down_brake_min_speed,
        down_low_speed_response_accel: 0.0,
        brake_decel_low_mid: brake_decel,
        brake_decel_mid: brake_decel,
        turn_rate_low,
        turn_rate_low_mid,
        turn_rate_mid,
        turn_rate_high,
        combo_left_accel_low: forward_accel,
        combo_left_accel_low_mid: forward_accel,
        combo_right_accel_low: forward_accel,
        combo_right_accel_low_mid: forward_accel,
        combo_left_turn_rate_low: turn_rate_low,
        combo_left_turn_rate_low_mid: turn_rate_low_mid,
        combo_right_turn_rate_low: turn_rate_low,
        combo_right_turn_rate_low_mid: turn_rate_low_mid,
    }
}

fn fit_accel_for_key(poses: &[PoseSample], controls: &[ControlSample], key: Key) -> Option<f32> {
    let windows = control_windows(controls, |sample| same_keys(&sample.keys, &[key]));
    mean_present(
        &windows
            .iter()
            .filter_map(|(start, end)| speed_slope(poses, *start, *end))
            .filter(|slope| *slope > 0.0)
            .collect::<Vec<_>>()
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>(),
    )
}

fn fit_decel_for_key(poses: &[PoseSample], controls: &[ControlSample], key: Key) -> Option<f32> {
    let windows = control_windows(controls, |sample| same_keys(&sample.keys, &[key]));
    mean_present(
        &windows
            .iter()
            .filter_map(|(start, end)| speed_slope(poses, *start, *end))
            .filter(|slope| *slope < 0.0)
            .map(|slope| Some(-slope))
            .collect::<Vec<_>>(),
    )
}

fn fit_coast_decel(poses: &[PoseSample], controls: &[ControlSample]) -> Option<f32> {
    let windows = controls
        .windows(2)
        .filter_map(|pair| {
            let end = &pair[0];
            let next = &pair[1];
            if end.event == "hold_end" && next.event == "sleep" {
                Some((end.tick_ms, next.tick_ms.saturating_add(250)))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    mean_present(
        &windows
            .iter()
            .filter_map(|(start, end)| speed_slope(poses, *start, *end))
            .filter(|slope| *slope < 0.0)
            .map(|slope| Some(-slope))
            .collect::<Vec<_>>(),
    )
}

fn fit_turn_rate_for_keys(
    poses: &[PoseSample],
    controls: &[ControlSample],
    keys: &[Key],
) -> Option<f32> {
    let windows = control_windows(controls, |sample| same_keys(&sample.keys, keys));
    mean_present(
        &windows
            .iter()
            .filter_map(|(start, end)| heading_rate(poses, *start, *end))
            .map(|rate| Some(rate.abs()))
            .collect::<Vec<_>>(),
    )
}

fn control_windows<F>(controls: &[ControlSample], pred: F) -> Vec<(u128, u128)>
where
    F: Fn(&ControlSample) -> bool,
{
    let mut out = Vec::new();
    for (idx, sample) in controls.iter().enumerate() {
        if sample.event != "hold_begin" || !pred(sample) {
            continue;
        }
        let end = controls[idx + 1..]
            .iter()
            .find(|candidate| {
                candidate.event == "hold_end" && same_keys(&candidate.keys, &sample.keys)
            })
            .map(|candidate| candidate.tick_ms)
            .unwrap_or(sample.tick_ms);
        if end > sample.tick_ms {
            out.push((sample.tick_ms, end));
        }
    }
    out
}

fn speed_slope(poses: &[PoseSample], start: u128, end: u128) -> Option<f32> {
    let a = nearest_pose(poses, start)?;
    let b = nearest_pose(poses, end)?;
    let dt = (b.tick_ms.saturating_sub(a.tick_ms) as f32) / 1000.0;
    if dt > 0.05 {
        Some((b.pose.speed - a.pose.speed) / dt)
    } else {
        None
    }
}

fn heading_rate(poses: &[PoseSample], start: u128, end: u128) -> Option<f32> {
    let a = nearest_pose(poses, start)?;
    let b = nearest_pose(poses, end)?;
    let dt = (b.tick_ms.saturating_sub(a.tick_ms) as f32) / 1000.0;
    if dt > 0.05 {
        let (a_heading, _) = navigation_heading(a.pose);
        let (b_heading, _) = navigation_heading(b.pose);
        Some(normalize_angle(b_heading - a_heading) / dt)
    } else {
        None
    }
}

fn nearest_pose(poses: &[PoseSample], tick_ms: u128) -> Option<PoseSample> {
    poses
        .iter()
        .min_by_key(|sample| sample.tick_ms.abs_diff(tick_ms))
        .copied()
}

fn same_keys(a: &[Key], b: &[Key]) -> bool {
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort_by_key(|key| key.name());
    bb.sort_by_key(|key| key.name());
    aa == bb
}

fn keys_label(keys: &[Key]) -> String {
    if keys.is_empty() {
        return "Coast".to_string();
    }
    let mut ordered = keys.to_vec();
    ordered.sort_by_key(|key| key.name());
    ordered
        .iter()
        .map(|key| key.name())
        .collect::<Vec<_>>()
        .join("+")
}

fn mean_present(values: &[Option<f32>]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0;
    for value in values.iter().flatten() {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

fn target_score(target: &KapTarget, pose: Pose, nav_mesh: Option<&NavMesh>) -> f32 {
    if let Some(score) = nav_mesh.and_then(|mesh| mesh.ring_score(pose, target.center)) {
        score
    } else {
        point_score(target.center, pose.pos) + 10_000.0
    }
}

fn load_kap_follow_paths(request: &KapFollowRequest) -> Result<Vec<KapFollowPath>, String> {
    let kap_dir = request
        .kap_dir
        .as_ref()
        .ok_or("KAP follow requires --kap-dir")?;
    let track_id = request
        .track_id
        .as_deref()
        .ok_or("KAP follow requires --kap-track or an inferable track path")?;
    if let Some(slot) = request.slot {
        let path = kap_dir.join(format!("{track_id}_{slot}.kap"));
        let bytes =
            fs::read(&path).map_err(|err| format!("read exact KAP {}: {err}", path.display()))?;
        let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
            .map_err(|err| format!("decode exact KAP {}: {err}", path.display()))?;
        let kap = kart_fmt::kap::parse(&clear)
            .ok_or_else(|| format!("parse exact KAP {}: invalid payload", path.display()))?;
        if kap.points.len() < 2 {
            return Err(format!(
                "exact KAP {} has fewer than two points",
                path.display()
            ));
        }
        return Ok(vec![KapFollowPath::new(
            path.display().to_string(),
            track_id.to_string(),
            Some(slot),
            kap.points,
        )]);
    }
    let mut paths = Vec::new();
    visit_files(kap_dir, &mut |path| {
        if path.extension().and_then(|s| s.to_str()) != Some("kap") {
            return;
        }
        let Some(file_track_id) = kap_track_id(path) else {
            return;
        };
        if file_track_id != track_id {
            return;
        }
        let slot = kap_slot(path);
        let Ok(bytes) = fs::read(path) else {
            return;
        };
        let Ok(clear) = kart_rho::encoded_block::decode_len_prefixed(&bytes) else {
            return;
        };
        let Some(kap) = kart_fmt::kap::parse(&clear) else {
            return;
        };
        if kap.points.len() < 2 {
            return;
        }
        paths.push(KapFollowPath::new(
            path.display().to_string(),
            file_track_id,
            slot,
            kap.points,
        ));
    })?;
    paths.sort_by(|a, b| a.slot.cmp(&b.slot).then_with(|| a.path_id.cmp(&b.path_id)));
    if paths.is_empty() {
        Err(format!(
            "no KAP paths found in {} for track_id={} slot={:?}",
            kap_dir.display(),
            track_id,
            request.slot
        ))
    } else {
        Ok(paths)
    }
}

fn choose_kap_follow_path<'a>(
    paths: &'a [KapFollowPath],
    pose: Pose,
    request: &KapFollowRequest,
) -> Option<&'a KapFollowPath> {
    if let Some(start_path_s) = request.start_path_s {
        let window = request.start_window_m.max(request.max_cross_track).max(1.0);
        let selected = paths
            .iter()
            .filter(|path| {
                path.project_near_strict(pose.pos, start_path_s, window)
                    .map(|projection| projection.cross_track <= request.max_cross_track)
                    .unwrap_or(false)
            })
            .min_by(|a, b| {
                let da = a
                    .project_near_strict(pose.pos, start_path_s, window)
                    .map(|p| p.cross_track)
                    .unwrap_or(f32::INFINITY);
                let db = b
                    .project_near_strict(pose.pos, start_path_s, window)
                    .map(|p| p.cross_track)
                    .unwrap_or(f32::INFINITY);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        if selected.is_some() {
            return selected;
        }
        return None;
    }
    let prefer_closed = request.target_progress > 0.0 && request.slot.is_none();
    if prefer_closed {
        let closed = paths
            .iter()
            .filter(|path| path.closed)
            .filter(|path| {
                path.project(pose.pos)
                    .map(|projection| projection.cross_track <= request.max_cross_track)
                    .unwrap_or(false)
            })
            .min_by(|a, b| {
                let da = a
                    .project(pose.pos)
                    .map(|p| p.cross_track)
                    .unwrap_or(f32::INFINITY);
                let db = b
                    .project(pose.pos)
                    .map(|p| p.cross_track)
                    .unwrap_or(f32::INFINITY);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        if closed.is_some() {
            return closed;
        }
    }
    paths.iter().min_by(|a, b| {
        let da = a
            .project(pose.pos)
            .map(|p| p.cross_track)
            .unwrap_or(f32::INFINITY);
        let db = b
            .project(pose.pos)
            .map(|p| p.cross_track)
            .unwrap_or(f32::INFINITY);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn initial_kap_projection(
    path: &KapFollowPath,
    pose: Pose,
    request: &KapFollowRequest,
) -> Option<KapProjection> {
    if let Some(start_path_s) = request.start_path_s {
        let window = request.start_window_m.max(request.max_cross_track).max(1.0);
        if let Some(projection) = path.project_near_strict(pose.pos, start_path_s, window) {
            if projection.cross_track <= request.max_cross_track {
                return Some(projection);
            }
        }
        return None;
    }
    path.project(pose.pos)
}

fn kap_projection_acceptance(request: &KapFollowRequest) -> f32 {
    request
        .panic_cross_track
        .max(request.safe_cross_track * 2.0)
        .max(20.0)
}

fn kap_projection_is_remote(
    projection: KapProjection,
    vertical_gap: f32,
    request: &KapFollowRequest,
) -> bool {
    projection.cross_track > request.max_cross_track
        || (vertical_gap > 12.0 && projection.cross_track > kap_projection_acceptance(request))
}

fn kap_recovery_projection_is_better(
    path: &KapFollowPath,
    candidate: KapProjection,
    directional: KapProjection,
    cursor_s: f32,
    projection_window: f32,
    improvement_margin: f32,
) -> bool {
    path.path_s_distance(candidate.path_s, cursor_s) <= projection_window
        && candidate.cross_track + improvement_margin < directional.cross_track
}

fn kap_recovery_projection_respects_topology_floor(
    path: &KapFollowPath,
    candidate: KapProjection,
    topology_floor_s: Option<f32>,
    projection_window: f32,
) -> bool {
    let Some(floor_s) = topology_floor_s else {
        return true;
    };
    path.path_s_distance(floor_s, candidate.path_s) <= 0.001
        || path.forward_delta_bounded(floor_s, candidate.path_s, projection_window) > 0.0
}

fn kap_frontier_progress_credit(
    path: &KapFollowPath,
    credited_cursor_s: f32,
    cursor_s: f32,
    max_progress_step_s: f32,
    recovery_cursor_regression: bool,
) -> f32 {
    if recovery_cursor_regression {
        return 0.0;
    }
    path.forward_delta_bounded(credited_cursor_s, cursor_s, max_progress_step_s)
        .max(0.0)
}

fn kap_downward_topology_transfer(
    forward_delta_s: f32,
    cursor_vertical_drop: f32,
    raw_vertical_gap: f32,
    max_progress_step_s: f32,
) -> bool {
    forward_delta_s > 0.0
        && forward_delta_s <= max_progress_step_s
        && cursor_vertical_drop > 12.0
        && raw_vertical_gap <= 3.0
}

fn kap_position_replacement_threshold(request: &KapFollowRequest) -> f32 {
    request.max_cross_track.max(40.0)
}

fn kap_position_replacement_detected(activity_delta: f32, request: &KapFollowRequest) -> bool {
    activity_delta.is_finite() && activity_delta >= kap_position_replacement_threshold(request)
}

impl KapFollowPath {
    fn point_at_lateral_offset(&self, path_s: f32, offset: f32) -> Option<P3> {
        let point = self.point_at(path_s)?;
        if offset.abs() <= f32::EPSILON {
            return Some(point);
        }
        let heading = self.heading_at(path_s)?;
        Some([
            point[0] - heading.sin() * offset,
            point[1] + heading.cos() * offset,
            point[2],
        ])
    }

    fn new(path_id: String, track_id: String, slot: Option<usize>, points: Vec<P3>) -> Self {
        let closed = point_distance_2d(points[0], *points.last().unwrap_or(&points[0])) <= 35.0;
        let mut cumulative = Vec::with_capacity(points.len());
        cumulative.push(0.0);
        let mut length = 0.0;
        for pair in points.windows(2) {
            length += point_distance_2d(pair[0], pair[1]);
            cumulative.push(length);
        }
        if closed {
            length += point_distance_2d(*points.last().unwrap(), points[0]);
        }
        Self {
            path_id,
            track_id,
            slot,
            points,
            cumulative,
            length,
            closed,
        }
    }

    fn project(&self, point: P3) -> Option<KapProjection> {
        self.project_with_filter(point, None, 0.0)
    }

    fn project_near(&self, point: P3, center_s: f32, window: f32) -> Option<KapProjection> {
        self.project_with_filter(point, Some(center_s), window)
            .or_else(|| self.project(point))
    }

    fn project_near_strict(&self, point: P3, center_s: f32, window: f32) -> Option<KapProjection> {
        self.project_with_filter(point, Some(center_s), window)
    }

    fn project_near_forward(
        &self,
        point: P3,
        center_s: f32,
        forward_window: f32,
        backward_tolerance: f32,
    ) -> Option<KapProjection> {
        self.project_with_directional_filter(
            point,
            center_s,
            forward_window.max(0.0),
            backward_tolerance.max(0.0),
        )
        .or_else(|| self.project_near(point, center_s, forward_window))
    }

    fn project_with_filter(
        &self,
        point: P3,
        center_s: Option<f32>,
        window: f32,
    ) -> Option<KapProjection> {
        if self.points.len() < 2 || self.length <= f32::EPSILON {
            return None;
        }
        let mut best = KapProjection {
            path_s: 0.0,
            cross_track: f32::INFINITY,
        };
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        };
        for i in 0..segment_count {
            let a = self.points[i];
            let j = if i + 1 < self.points.len() { i + 1 } else { 0 };
            let b = self.points[j];
            let vx = b[0] - a[0];
            let vy = b[1] - a[1];
            let seg2 = vx * vx + vy * vy;
            if seg2 <= f32::EPSILON {
                continue;
            }
            let wx = point[0] - a[0];
            let wy = point[1] - a[1];
            let t = ((wx * vx + wy * vy) / seg2).clamp(0.0, 1.0);
            let proj = [a[0] + vx * t, a[1] + vy * t, a[2] + (b[2] - a[2]) * t];
            let cross_track = point_distance_weighted(point, proj);
            if cross_track < best.cross_track {
                let base_s = if i < self.cumulative.len() {
                    self.cumulative[i]
                } else {
                    self.cumulative.last().copied().unwrap_or(0.0)
                };
                let path_s = base_s + point_distance_2d(a, b) * t;
                if let Some(center_s) = center_s {
                    if self.path_s_distance(path_s, center_s) > window {
                        continue;
                    }
                }
                best = KapProjection {
                    path_s,
                    cross_track,
                };
            }
        }
        best.cross_track.is_finite().then_some(best)
    }

    fn project_with_directional_filter(
        &self,
        point: P3,
        center_s: f32,
        forward_window: f32,
        backward_tolerance: f32,
    ) -> Option<KapProjection> {
        if self.points.len() < 2 || self.length <= f32::EPSILON {
            return None;
        }
        let mut best = KapProjection {
            path_s: 0.0,
            cross_track: f32::INFINITY,
        };
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        };
        for i in 0..segment_count {
            let a = self.points[i];
            let j = if i + 1 < self.points.len() { i + 1 } else { 0 };
            let b = self.points[j];
            let vx = b[0] - a[0];
            let vy = b[1] - a[1];
            let seg2 = vx * vx + vy * vy;
            if seg2 <= f32::EPSILON {
                continue;
            }
            let wx = point[0] - a[0];
            let wy = point[1] - a[1];
            let t = ((wx * vx + wy * vy) / seg2).clamp(0.0, 1.0);
            let proj = [a[0] + vx * t, a[1] + vy * t, a[2] + (b[2] - a[2]) * t];
            let cross_track = point_distance_weighted(point, proj);
            if cross_track >= best.cross_track {
                continue;
            }
            let base_s = if i < self.cumulative.len() {
                self.cumulative[i]
            } else {
                self.cumulative.last().copied().unwrap_or(0.0)
            };
            let path_s = base_s + point_distance_2d(a, b) * t;
            let forward = self.forward_distance(center_s, path_s);
            let backward = self.forward_distance(path_s, center_s);
            if forward > forward_window && backward > backward_tolerance {
                continue;
            }
            best = KapProjection {
                path_s,
                cross_track,
            };
        }
        best.cross_track.is_finite().then_some(best)
    }

    fn path_s_distance(&self, a: f32, b: f32) -> f32 {
        let direct = (a - b).abs();
        if self.closed && self.length > f32::EPSILON {
            direct.min((self.length - direct).abs())
        } else {
            direct
        }
    }

    fn forward_distance(&self, from_s: f32, to_s: f32) -> f32 {
        let delta = to_s - from_s;
        if delta >= 0.0 {
            delta
        } else if self.closed && self.length > f32::EPSILON {
            delta + self.length
        } else {
            f32::INFINITY
        }
    }

    fn forward_delta_bounded(&self, from_s: f32, to_s: f32, max_step_s: f32) -> f32 {
        let delta = to_s - from_s;
        if delta >= 0.0 {
            if delta <= max_step_s {
                delta
            } else {
                0.0
            }
        } else if self.closed && self.length > f32::EPSILON {
            let wrapped = delta + self.length;
            if wrapped >= 0.0 && wrapped <= max_step_s {
                wrapped
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn seam_completion_delta(&self, path_s: f32, tolerance_s: f32) -> f32 {
        if !self.closed || self.length <= f32::EPSILON || tolerance_s <= 0.0 {
            return 0.0;
        }
        let remaining = self.length - path_s.clamp(0.0, self.length);
        if remaining > 0.0 && remaining <= tolerance_s {
            remaining
        } else {
            0.0
        }
    }

    fn point_at(&self, mut path_s: f32) -> Option<P3> {
        if self.points.len() < 2 || self.length <= f32::EPSILON {
            return None;
        }
        if self.closed {
            path_s %= self.length;
            if path_s < 0.0 {
                path_s += self.length;
            }
        } else {
            path_s = path_s.clamp(0.0, self.length);
        }
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        };
        for i in 0..segment_count {
            let a = self.points[i];
            let j = if i + 1 < self.points.len() { i + 1 } else { 0 };
            let b = self.points[j];
            let base_s = if i < self.cumulative.len() {
                self.cumulative[i]
            } else {
                self.cumulative.last().copied().unwrap_or(0.0)
            };
            let seg_len = point_distance_2d(a, b);
            if seg_len <= f32::EPSILON {
                continue;
            }
            if path_s <= base_s + seg_len || i + 1 == segment_count {
                let t = ((path_s - base_s) / seg_len).clamp(0.0, 1.0);
                return Some([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ]);
            }
        }
        self.points.last().copied()
    }

    fn heading_at(&self, mut path_s: f32) -> Option<f32> {
        if self.points.len() < 2 || self.length <= f32::EPSILON {
            return None;
        }
        if self.closed {
            path_s %= self.length;
            if path_s < 0.0 {
                path_s += self.length;
            }
        } else {
            path_s = path_s.clamp(0.0, self.length);
        }
        let segment_count = if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        };
        for i in 0..segment_count {
            let a = self.points[i];
            let j = if i + 1 < self.points.len() { i + 1 } else { 0 };
            let b = self.points[j];
            let base_s = if i < self.cumulative.len() {
                self.cumulative[i]
            } else {
                self.cumulative.last().copied().unwrap_or(0.0)
            };
            let seg_len = point_distance_2d(a, b);
            if seg_len <= f32::EPSILON {
                continue;
            }
            if path_s <= base_s + seg_len || i + 1 == segment_count {
                return Some((b[1] - a[1]).atan2(b[0] - a[0]));
            }
        }
        None
    }
}

fn mine_kap_route_windows(
    path: &KapFollowPath,
    options: &KapRouteMineOptions,
) -> Vec<KapRouteWindow> {
    if path.length <= options.window_m.max(1.0) || options.step_m <= 0.0 || options.window_m <= 0.0
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    let max_start = if path.closed {
        path.length
    } else {
        (path.length - options.window_m).max(0.0)
    };
    let mut start_s = 0.0;
    while start_s < max_start {
        if let Some(window) = score_kap_route_window(path, start_s, options.window_m, options) {
            if window.score >= options.min_score {
                out.push(window);
            }
        }
        start_s += options.step_m;
    }
    out
}

fn score_kap_route_window(
    path: &KapFollowPath,
    start_s: f32,
    length_m: f32,
    options: &KapRouteMineOptions,
) -> Option<KapRouteWindow> {
    let end_s = start_s + length_m;
    let start = path.point_at(start_s)?;
    let end = path.point_at(end_s)?;
    let direct = point_distance_2d(start, end);
    let straightness = if length_m > f32::EPSILON {
        (direct / length_m).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let start_heading = path.heading_at(start_s)?;
    let end_heading = path.heading_at(end_s)?;
    let heading_change_rad = normalize_angle(end_heading - start_heading).abs();

    let samples = 8usize;
    let mut prev_heading = start_heading;
    let mut max_segment_heading_delta_rad = 0.0f32;
    let mut grade_abs_max = 0.0f32;
    for sample_i in 1..=samples {
        let s = start_s + length_m * sample_i as f32 / samples as f32;
        let heading = path.heading_at(s)?;
        max_segment_heading_delta_rad =
            max_segment_heading_delta_rad.max(normalize_angle(heading - prev_heading).abs());
        prev_heading = heading;
        let a = path.point_at(s - length_m / samples as f32)?;
        let b = path.point_at(s)?;
        let horizontal = point_distance_2d(a, b);
        if horizontal > f32::EPSILON {
            grade_abs_max = grade_abs_max.max(((b[2] - a[2]) / horizontal).abs());
        }
    }
    let grade_change = (end[2] - start[2]).abs();
    let heading_score = if options.max_heading_change_rad > f32::EPSILON {
        1.0 - (heading_change_rad / options.max_heading_change_rad).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let segment_score = 1.0 - (max_segment_heading_delta_rad / 0.35).clamp(0.0, 1.0);
    let grade_score = 1.0 - (grade_abs_max / 0.15).clamp(0.0, 1.0);
    let score =
        straightness * 0.45 + heading_score * 0.30 + segment_score * 0.20 + grade_score * 0.05;
    let kind = if heading_change_rad <= 0.12 && straightness >= 0.96 {
        "straight"
    } else if heading_change_rad <= options.max_heading_change_rad {
        "gentle_curve"
    } else {
        "curve"
    };
    let verdict = if score >= 0.82 {
        "good_sampling_window"
    } else if score >= options.min_score {
        "usable_with_margin"
    } else {
        "reject"
    };

    Some(KapRouteWindow {
        track_id: path.track_id.clone(),
        slot: path.slot,
        path_id: path.path_id.clone(),
        start_s,
        end_s: if path.closed && end_s >= path.length {
            end_s - path.length
        } else {
            end_s.min(path.length)
        },
        center_s: if path.closed {
            (start_s + length_m * 0.5) % path.length
        } else {
            (start_s + length_m * 0.5).min(path.length)
        },
        length_m,
        progress_start: start_s / path.length,
        progress_end: if path.closed {
            (end_s % path.length) / path.length
        } else {
            end_s.min(path.length) / path.length
        },
        heading_change_rad,
        max_segment_heading_delta_rad,
        straightness,
        grade_change,
        grade_abs_max,
        score,
        kind,
        verdict,
        start,
        end,
    })
}

fn visit_files(dir: &Path, f: &mut impl FnMut(&Path)) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|err| format!("read_dir {}: {err}", dir.display()))? {
        let entry = entry.map_err(|err| format!("dir entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            visit_files(&path, f)?;
        } else {
            f(&path);
        }
    }
    Ok(())
}

fn point_score(a: P3, b: P3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let horizontal = (dx * dx + dy * dy).sqrt();
    horizontal + dz.abs() * 4.0
}

fn kap_targets(
    track_path: &Path,
    surface_filter: Option<&str>,
    rail_filter: Option<&str>,
) -> Result<Vec<KapTarget>, String> {
    let bytes =
        fs::read(track_path).map_err(|err| format!("read {}: {err}", track_path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|err| format!("parse {}: {err}", track_path.display()))?;
    let track_id = track_id(track_path);
    let mut out = Vec::new();

    for (node_idx, world) in track.flatten() {
        let node = &track.nodes[node_idx];
        let road_surface = tag_attr(node.extra_tag.as_ref(), "road", "surface");
        let road_rail = tag_attr(node.extra_tag.as_ref(), "road", "rail").unwrap_or_default();
        if surface_filter.is_some() && road_surface.as_deref() != surface_filter {
            continue;
        }
        if rail_filter.is_some() && road_rail != rail_filter.unwrap_or_default() {
            continue;
        }
        if surface_filter.is_none()
            && rail_filter.is_none()
            && road_surface.is_none()
            && road_rail.is_empty()
        {
            continue;
        }

        let Some(geom) = &node.geom else { continue };
        for (tri_i, tri_idx) in geom_triangles(geom).into_iter().enumerate() {
            if tri_idx.iter().any(|&i| i as usize >= geom.vertices.len()) {
                continue;
            }
            let tri = [
                xform(&world, geom.vertices[tri_idx[0] as usize]),
                xform(&world, geom.vertices[tri_idx[1] as usize]),
                xform(&world, geom.vertices[tri_idx[2] as usize]),
            ];
            let center = tri_center(tri);
            let surface = road_surface.clone().unwrap_or_default();
            out.push(KapTarget {
                target_id: format!("{}:{}:{}", track_id, node.name, tri_i),
                track_id: track_id.clone(),
                node: node.name.clone(),
                road_surface: surface.clone(),
                surface_id: if surface.is_empty() {
                    0
                } else {
                    surface_id(&surface)
                },
                road_rail: road_rail.clone(),
                tri_index: tri_i,
                center,
                normal: tri_normal(tri),
            });
        }
    }

    out.sort_by(|a, b| {
        a.road_surface
            .cmp(&b.road_surface)
            .then_with(|| a.node.cmp(&b.node))
            .then_with(|| a.tri_index.cmp(&b.tri_index))
    });
    Ok(out)
}

fn read_latest_pose(path: &Path) -> Result<Pose, String> {
    let headers =
        read_tsv_header(path).ok_or_else(|| format!("pose {} is empty", path.display()))?;
    let header_cols: Vec<_> = headers.iter().map(String::as_str).collect();
    let pose_cols = PoseColumns::from_header(&header_cols)?;

    let mut latest_primary = None;
    let mut latest_any = None;
    let mut latest_scored: Option<(u128, f32, String)> = None;
    for line in read_tail_lines(path, 256 * 1024) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<_> = line.split('\t').collect();
        let Ok(parsed_pose) = pose_cols.parse_pose(&cols) else {
            continue;
        };
        if cols == header_cols || !pose_cols.row_is_plausible(&cols, parsed_pose) {
            continue;
        }
        if let Some(score) = pose_cols.candidate_score(&cols) {
            let tick = pose_cols.tick_ms_or_zero(&cols);
            let replace = latest_scored
                .as_ref()
                .map(|(best_tick, best_score, _)| {
                    tick > *best_tick || (tick == *best_tick && score > *best_score)
                })
                .unwrap_or(true);
            if replace {
                latest_scored = Some((tick, score, line));
            }
            continue;
        }
        if pose_cols.is_primary_candidate(&cols) {
            latest_primary = Some(line);
        } else {
            latest_any = Some(line);
        }
    }
    if latest_scored.is_none() && latest_primary.is_none() && latest_any.is_none() {
        let file =
            File::open(path).map_err(|err| format!("open pose {}: {err}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|err| format!("stat pose {}: {err}", path.display()))?
            .len();
        let mut lines = BufReader::new(file.take(len)).lines();
        let _ = lines.next();
        for line in lines {
            let line = line.map_err(|err| format!("read pose row: {err}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let cols: Vec<_> = line.split('\t').collect();
            let Ok(parsed_pose) = pose_cols.parse_pose(&cols) else {
                continue;
            };
            if cols == header_cols || !pose_cols.row_is_plausible(&cols, parsed_pose) {
                continue;
            }
            if let Some(score) = pose_cols.candidate_score(&cols) {
                let tick = pose_cols.tick_ms_or_zero(&cols);
                let replace = latest_scored
                    .as_ref()
                    .map(|(best_tick, best_score, _)| {
                        tick > *best_tick || (tick == *best_tick && score > *best_score)
                    })
                    .unwrap_or(true);
                if replace {
                    latest_scored = Some((tick, score, line));
                }
            } else if pose_cols.is_primary_candidate(&cols) {
                latest_primary = Some(line);
            } else {
                latest_any = Some(line);
            }
        }
    }
    let latest = latest_scored
        .map(|(_, _, line)| line)
        .or(latest_primary)
        .or(latest_any)
        .ok_or_else(|| format!("pose {} has no rows", path.display()))?;
    let cols: Vec<_> = latest.split('\t').collect();
    pose_cols.parse_pose(&cols)
}

fn prefer_owner_pose(pose_age_ms: Option<u128>, owner_age_ms: Option<u128>) -> bool {
    pose_age_ms.is_none_or(|age| age > 500) && owner_age_ms.is_some_and(|age| age <= 2_000)
}

fn read_latest_drive_pose(pose_path: &Path) -> Result<Pose, String> {
    let owner_path = pose_path
        .parent()
        .map(|dir| dir.join("probe_owner.tsv"))
        .unwrap_or_else(|| PathBuf::from("probe_owner.tsv"));
    if prefer_owner_pose(
        file_modified_age_ms(pose_path),
        file_modified_age_ms(&owner_path),
    ) {
        if let Ok(owner_pose) = read_latest_pose(&owner_path) {
            return Ok(owner_pose);
        }
    }
    read_latest_pose(pose_path)
}

fn read_latest_ground_drive_feedback(path: &Path, pose: Pose) -> Option<GroundDriveFeedback> {
    // Full-hook captures can produce probe_ground_drive.tsv faster than
    // 10 MiB/s. On Windows, repeatedly opening a several-hundred-megabyte
    // file under that write pressure can block the real-time controller
    // before its first input step even though only a small tail is needed.
    // Kind40 feedback is an optional online correction; the versioned
    // physics baseline remains the bounded fallback. Refuse an unbounded
    // telemetry transport instead of surrendering input authority to it.
    const REALTIME_FEEDBACK_FILE_BUDGET: u64 = 32 * 1024 * 1024;
    if fs::metadata(path).ok()?.len() > REALTIME_FEEDBACK_FILE_BUDGET {
        return None;
    }
    let headers = read_tsv_header(path)?;
    let header_cols: Vec<_> = headers.iter().map(String::as_str).collect();
    let cols = GroundDriveColumns::from_header(&header_cols).ok()?;
    let mut rows = Vec::new();
    for line in read_tail_lines(path, 256 * 1024) {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<_> = line.split('\t').collect();
        if values == header_cols {
            continue;
        }
        if let Some(row) = cols.parse_kind40_row(&values) {
            rows.push(row);
        }
    }
    let latest_tick = rows.last()?.tick_ms;
    let recent: Vec<_> = rows
        .into_iter()
        .filter(|row| latest_tick - row.tick_ms <= 120.0)
        .collect();
    GroundDriveFeedback::from_kind40_rows(&recent, pose)
}

fn file_modified_age_ms(path: &Path) -> Option<u128> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .elapsed()
        .ok()
        .map(|elapsed| elapsed.as_millis())
}

fn wall_response_kind_label(kind: u32) -> &'static str {
    match kind {
        4 => "hard_velocity_post",
        5 => "hard_angular_post",
        8 => "soft_velocity_add",
        9 => "soft_angular_add0",
        10 => "soft_angular_add1",
        _ => "unknown",
    }
}

fn wall_response_fresh_for_pose(wall_tick_ms: Option<f32>, pose_tick_ms: Option<f32>) -> bool {
    match (wall_tick_ms, pose_tick_ms) {
        (Some(wall_tick), Some(pose_tick)) => {
            let age_ms = pose_tick - wall_tick;
            (-50.0..=500.0).contains(&age_ms)
        }
        _ => true,
    }
}

fn read_latest_wall_contact_feedback(
    path: &Path,
    pose_tick_ms: Option<f32>,
) -> Option<WallContactFeedback> {
    if file_modified_age_ms(path)? > 2_000 {
        return None;
    }
    let headers = read_tsv_header(path)?;
    let header_cols: Vec<_> = headers.iter().map(String::as_str).collect();
    let tick_i = col_index(&header_cols, &["tick_ms", "tick"]).ok();
    let kind_i = col_index(&header_cols, &["kind"]).ok()?;
    let site_i = col_index(&header_cols, &["site_rva"]).ok();
    let hit_n_x_i = col_index(&header_cols, &["hit_n_x"]).ok()?;
    let hit_n_y_i = col_index(&header_cols, &["hit_n_y"]).ok()?;
    let hit_n_z_i = col_index(&header_cols, &["hit_n_z"]).ok()?;
    let hit_dot_vel_i = col_index(&header_cols, &["hit_dot_vel"]).ok();
    for line in read_tail_lines(path, 256 * 1024).into_iter().rev() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<_> = line.split('\t').collect();
        if cols == header_cols {
            continue;
        }
        let kind = cols.get(kind_i)?.parse::<u32>().ok()?;
        if !matches!(kind, 4 | 5 | 8 | 9 | 10) {
            continue;
        }
        let hit_normal = [
            parse_optional_col(&cols, hit_n_x_i)?,
            parse_optional_col(&cols, hit_n_y_i)?,
            parse_optional_col(&cols, hit_n_z_i)?,
        ];
        let tick_ms = tick_i.and_then(|index| parse_optional_col(&cols, index));
        if !wall_response_fresh_for_pose(tick_ms, pose_tick_ms) {
            continue;
        }
        return Some(WallContactFeedback {
            kind,
            kind_label: wall_response_kind_label(kind),
            site_rva: site_i
                .and_then(|index| cols.get(index).map(|value| (*value).to_string()))
                .unwrap_or_default(),
            tick_ms,
            hit_normal,
            hit_dot_vel: hit_dot_vel_i.and_then(|index| parse_optional_col(&cols, index)),
        });
    }
    None
}

struct GroundDriveColumns {
    tick_i: usize,
    kind_i: usize,
    src_x_i: usize,
    src_y_i: usize,
    src_z_i: usize,
    pre_x_i: usize,
    pre_y_i: usize,
    pre_z_i: usize,
    post_x_i: usize,
    post_y_i: usize,
    post_z_i: usize,
}

impl GroundDriveColumns {
    fn from_header(header: &[&str]) -> Result<Self, String> {
        Ok(Self {
            tick_i: col_index(header, &["tick_ms", "tick"])?,
            kind_i: col_index(header, &["kind"])?,
            src_x_i: col_index(header, &["src_x"])?,
            src_y_i: col_index(header, &["src_y"])?,
            src_z_i: col_index(header, &["src_z"])?,
            pre_x_i: col_index(header, &["pre_x"])?,
            pre_y_i: col_index(header, &["pre_y"])?,
            pre_z_i: col_index(header, &["pre_z"])?,
            post_x_i: col_index(header, &["post_x"])?,
            post_y_i: col_index(header, &["post_y"])?,
            post_z_i: col_index(header, &["post_z"])?,
        })
    }

    fn parse_kind40_row(&self, cols: &[&str]) -> Option<GroundDriveKind40Row> {
        if cols.get(self.kind_i)? != &"40" {
            return None;
        }
        let tick_ms = parse_optional_col(cols, self.tick_i)?;
        let src = [
            parse_optional_col(cols, self.src_x_i)?,
            parse_optional_col(cols, self.src_y_i)?,
            parse_optional_col(cols, self.src_z_i)?,
        ];
        let pre = [
            parse_optional_col(cols, self.pre_x_i)?,
            parse_optional_col(cols, self.pre_y_i)?,
            parse_optional_col(cols, self.pre_z_i)?,
        ];
        let post = [
            parse_optional_col(cols, self.post_x_i)?,
            parse_optional_col(cols, self.post_y_i)?,
            parse_optional_col(cols, self.post_z_i)?,
        ];
        let observed = [post[0] - pre[0], post[1] - pre[1], post[2] - pre[2]];
        let residual = [
            observed[0] - src[0],
            observed[1] - src[1],
            observed[2] - src[2],
        ];
        let residual_mag =
            (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2])
                .sqrt();
        let src_mag = (src[0] * src[0] + src[1] * src[1] + src[2] * src[2]).sqrt();
        let delta = if src_mag > 1e-9 && residual_mag / src_mag > 0.01 {
            src
        } else {
            observed
        };
        Some(GroundDriveKind40Row { tick_ms, delta })
    }
}

fn read_pose_samples(path: &Path) -> Result<Vec<PoseSample>, String> {
    let file = File::open(path).map_err(|err| format!("open pose {}: {err}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|err| format!("stat pose {}: {err}", path.display()))?
        .len();
    let mut lines = BufReader::new(file.take(len)).lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("pose {} is empty", path.display()))?
        .map_err(|err| format!("read pose header: {err}"))?;
    let header_cols: Vec<_> = header.split('\t').collect();
    let pose_cols = PoseColumns::from_header(&header_cols)?;

    let mut samples = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line.map_err(|err| format!("read pose row: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<_> = line.split('\t').collect();
        if !pose_cols.is_primary_candidate(&cols) {
            continue;
        }
        let tick_ms = pose_cols.parse_tick_ms(&cols, path, line_no + 2)?;
        let pose = pose_cols.parse_pose(&cols)?;
        samples.push(PoseSample { tick_ms, pose });
    }
    samples.sort_by_key(|sample| sample.tick_ms);
    Ok(samples)
}

struct PoseColumns {
    tick_i: Option<usize>,
    x_i: usize,
    y_i: usize,
    z_i: usize,
    m01_i: Option<usize>,
    m11_i: Option<usize>,
    yaw_i: Option<usize>,
    vx_i: Option<usize>,
    vy_i: Option<usize>,
    speed_i: Option<usize>,
    candidate_i: Option<usize>,
    candidate_score_i: Option<usize>,
    matrix_i: Vec<usize>,
}

impl PoseColumns {
    fn from_header(header: &[&str]) -> Result<Self, String> {
        let matrix_i: Vec<usize> = (0..3)
            .flat_map(|row| (0..3).map(move |col| (row, col)))
            .filter_map(|(row, col)| {
                let names = [
                    format!("post_m{row}{col}"),
                    format!("m{row}{col}"),
                    format!("pre_m{row}{col}"),
                ];
                header
                    .iter()
                    .position(|value| names.iter().any(|name| value == name))
            })
            .collect();
        Ok(Self {
            tick_i: col_index(header, &["tick_ms", "tick"]).ok(),
            x_i: col_index(header, &["post_pos_x", "x", "pos_x"])?,
            y_i: col_index(header, &["post_pos_y", "y", "pos_y"])?,
            z_i: col_index(header, &["post_pos_z", "z", "pos_z"])?,
            m01_i: col_index(header, &["post_m01", "m01", "pre_m01"]).ok(),
            m11_i: col_index(header, &["post_m11", "m11", "pre_m11"]).ok(),
            yaw_i: col_index(header, &["post_ang_z", "yaw", "ang_z"]).ok(),
            vx_i: col_index(header, &["vel_x", "vx"]).ok(),
            vy_i: col_index(header, &["vel_y", "vy"]).ok(),
            speed_i: col_index(header, &["speed"]).ok(),
            candidate_i: col_index(header, &["candidate"]).ok(),
            candidate_score_i: col_index(header, &["candidate_score"]).ok(),
            matrix_i: if matrix_i.len() == 9 {
                matrix_i
            } else {
                Vec::new()
            },
        })
    }

    fn row_is_plausible(&self, cols: &[&str], pose: Pose) -> bool {
        if !pose
            .pos
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1_000_000.0)
            || !pose.speed.is_finite()
            || pose.speed >= 500.0
            || !pose.yaw.is_finite()
        {
            return false;
        }
        if self.matrix_i.len() != 9 {
            return true;
        }
        let matrix: Option<Vec<f32>> = self
            .matrix_i
            .iter()
            .map(|index| parse_optional_col(cols, *index))
            .collect();
        let Some(matrix) = matrix else {
            return false;
        };
        if matrix
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 2.0)
        {
            return false;
        }
        let rows = [&matrix[0..3], &matrix[3..6], &matrix[6..9]];
        let norm = |row: &[f32]| row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        rows.iter().all(|row| (norm(row) - 1.0).abs() <= 0.2)
            && dot(rows[0], rows[1]).abs() <= 0.2
            && dot(rows[0], rows[2]).abs() <= 0.2
            && dot(rows[1], rows[2]).abs() <= 0.2
    }

    fn candidate_score(&self, cols: &[&str]) -> Option<f32> {
        self.candidate_score_i
            .and_then(|index| parse_optional_col(cols, index))
            .filter(|score| score.is_finite())
    }

    fn tick_ms_or_zero(&self, cols: &[&str]) -> u128 {
        let Some(index) = self.tick_i else {
            return 0;
        };
        let Some(value) = cols.get(index) else {
            return 0;
        };
        value
            .parse::<u128>()
            .ok()
            .or_else(|| {
                value
                    .parse::<f64>()
                    .ok()
                    .map(|tick| tick.max(0.0).round() as u128)
            })
            .unwrap_or(0)
    }

    fn is_primary_candidate(&self, cols: &[&str]) -> bool {
        self.candidate_i
            .and_then(|i| cols.get(i))
            .map(|value| value.is_empty() || *value == "0")
            .unwrap_or(true)
    }

    fn parse_tick_ms(&self, cols: &[&str], path: &Path, line_no: usize) -> Result<u128, String> {
        let Some(tick_i) = self.tick_i else {
            return Err(format!("{}:{} missing tick_ms", path.display(), line_no));
        };
        let value = cols
            .get(tick_i)
            .ok_or_else(|| format!("{}:{} missing tick_ms", path.display(), line_no))?;
        if let Ok(tick) = value.parse::<u128>() {
            return Ok(tick);
        }
        let tick = value
            .parse::<f64>()
            .map_err(|_| format!("{}:{} invalid tick_ms", path.display(), line_no))?;
        Ok(tick.max(0.0).round() as u128)
    }

    fn parse_pose(&self, cols: &[&str]) -> Result<Pose, String> {
        let pos = [
            parse_col(cols, self.x_i, "x")?,
            parse_col(cols, self.y_i, "y")?,
            parse_col(cols, self.z_i, "z")?,
        ];
        let vx = match self.vx_i {
            Some(i) => Some(parse_col(cols, i, "vel_x")?),
            None => None,
        };
        let vy = match self.vy_i {
            Some(i) => Some(parse_col(cols, i, "vel_y")?),
            None => None,
        };
        let speed = if let Some(i) = self.speed_i {
            parse_col(cols, i, "speed")?
        } else if let (Some(vx), Some(vy)) = (vx, vy) {
            (vx * vx + vy * vy).sqrt()
        } else {
            0.0
        };
        let matrix_yaw = if let (Some(m01_i), Some(m11_i)) = (self.m01_i, self.m11_i) {
            let m01 = parse_col(cols, m01_i, "m01")?;
            let m11 = parse_col(cols, m11_i, "m11")?;
            if m01.is_finite() && m11.is_finite() && (m01.abs() + m11.abs()) > 0.001 {
                Some((-m11).atan2(-m01))
            } else {
                None
            }
        } else {
            None
        };
        let yaw = if let Some(yaw) = matrix_yaw {
            yaw
        } else if let Some(i) = self.yaw_i {
            parse_col(cols, i, "yaw")?
        } else if let (Some(vx), Some(vy)) = (vx, vy) {
            vy.atan2(vx)
        } else {
            0.0
        };
        let velocity_heading = if let (Some(vx), Some(vy)) = (vx, vy) {
            if (vx * vx + vy * vy).sqrt() >= 0.05 {
                Some(vy.atan2(vx))
            } else {
                None
            }
        } else {
            None
        };
        if !pos.iter().all(|value| value.is_finite())
            || !yaw.is_finite()
            || !speed.is_finite()
            || velocity_heading.is_some_and(|value| !value.is_finite())
        {
            return Err("pose row contains non-finite position, heading, or speed".to_string());
        }
        Ok(Pose {
            pos,
            yaw,
            speed,
            velocity_heading,
            tick_ms: self
                .tick_i
                .and_then(|index| parse_optional_col(cols, index)),
        })
    }
}

fn read_control_samples(path: &Path) -> Result<Vec<ControlSample>, String> {
    let file = File::open(path).map_err(|err| format!("open control {}: {err}", path.display()))?;
    let mut samples = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("read control row: {err}"))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<_> = line.split('\t').collect();
        if cols.len() < 4 {
            continue;
        }
        if line_no == 0 && cols[0] == "time_ms" && cols[1] == "event" {
            continue;
        }
        let tick_ms = cols[0]
            .parse::<u128>()
            .map_err(|_| format!("{}:{} invalid control tick", path.display(), line_no + 1))?;
        if !control_event_uses_keys(cols[1]) {
            continue;
        }
        let keys = if cols[2].is_empty() {
            Vec::new()
        } else {
            parse_keys(cols[2]).map_err(|err| {
                format!(
                    "{}:{} invalid control keys for event {}: {err}",
                    path.display(),
                    line_no + 1,
                    cols[1]
                )
            })?
        };
        samples.push(ControlSample {
            tick_ms,
            event: cols[1].to_string(),
            keys,
        });
    }
    samples.sort_by_key(|sample| sample.tick_ms);
    Ok(samples)
}

fn control_event_uses_keys(event: &str) -> bool {
    // target_select / kap_select put target_id / path_id in the keys column.
    !matches!(event, "target_select" | "kap_select")
}

fn read_latest_surface_id(path: &Path) -> Result<Option<u32>, String> {
    let file =
        File::open(path).map_err(|err| format!("open collision {}: {err}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("collision {} is empty", path.display()))?
        .map_err(|err| format!("read collision header: {err}"))?;
    let header_cols: Vec<_> = header.split('\t').collect();
    let surface_i = col_index(&header_cols, &["surface_id"])?;

    let mut latest = None;
    for line in lines {
        let line = line.map_err(|err| format!("read collision row: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        latest = Some(line);
    }
    let Some(latest) = latest else {
        return Ok(None);
    };
    let cols: Vec<_> = latest.split('\t').collect();
    let value = cols
        .get(surface_i)
        .ok_or_else(|| format!("missing collision surface_id at index {surface_i}"))?;
    if value.is_empty() || *value == "0" {
        Ok(None)
    } else {
        Ok(Some(parse_u32_arg(value, "surface_id")?))
    }
}

fn read_latest_state_kind_seq(path: &Path, wanted_kind: u32) -> Result<Option<u64>, String> {
    let file = File::open(path).map_err(|err| format!("open state {}: {err}", path.display()))?;
    let header = BufReader::new(file)
        .lines()
        .next()
        .ok_or_else(|| format!("state {} is empty", path.display()))?
        .map_err(|err| format!("read state header {}: {err}", path.display()))?;
    let header_fields: Vec<&str> = header.split('\t').collect();
    let seq_i = col_index(&header_fields, &["seq"])?;
    let kind_i = col_index(&header_fields, &["kind"])?;
    let mut latest = None;
    for line in read_tail_lines(path, 256 * 1024) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (Some(seq_text), Some(kind_text)) = (fields.get(seq_i), fields.get(kind_i)) else {
            continue;
        };
        let Ok(kind) = kind_text.parse::<u32>() else {
            continue;
        };
        if kind != wanted_kind {
            continue;
        }
        let Ok(seq) = seq_text.parse::<u64>() else {
            continue;
        };
        latest = Some(latest.map_or(seq, |previous: u64| previous.max(seq)));
    }
    Ok(latest)
}

fn plan_step(model: &VehicleModel, pose: Pose, target: P3, stop_distance: f32) -> DriveDecision {
    plan_step_with_heading_mode(model, pose, target, stop_distance, HeadingMode::Auto)
}

fn plan_step_with_heading_mode(
    model: &VehicleModel,
    pose: Pose,
    target: P3,
    stop_distance: f32,
    heading_mode: HeadingMode,
) -> DriveDecision {
    plan_step_with_physics_feedback(model, pose, target, stop_distance, heading_mode, None, None)
}

fn plan_step_with_physics_feedback(
    model: &VehicleModel,
    pose: Pose,
    target: P3,
    stop_distance: f32,
    heading_mode: HeadingMode,
    ground_feedback: Option<&GroundDriveFeedback>,
    physics_baseline: Option<&PhysicsBaseline>,
) -> DriveDecision {
    let dx = target[0] - pose.pos[0];
    let dy = target[1] - pose.pos[1];
    let distance = (dx * dx + dy * dy).sqrt();
    let target_yaw = dy.atan2(dx);
    let (nav_heading, heading_source) = navigation_heading_with_mode(pose, heading_mode);
    let heading_error = normalize_angle(target_yaw - nav_heading);
    let reverse_velocity = pose_has_reverse_velocity(pose);

    if distance <= stop_distance {
        return DriveDecision {
            keys: Vec::new(),
            done: true,
            heading_error,
            reason: format!("within_stop_distance:{distance:.3}"),
        };
    }

    let turn_key = if heading_error > 0.0 {
        Key::Left
    } else {
        Key::Right
    };
    let abs_err = heading_error.abs();
    let down_effect = model.down_effect_for_speed(pose.speed);
    let brake_distance = match down_effect {
        DownEffect::Brake { decel } if decel > 0.1 => pose.speed * pose.speed / (2.0 * decel),
        _ => 0.0,
    };
    let turn_rate = model.turn_rate_for_speed(pose.speed);
    let min_turn_speed = model.max_control_speed.min(6.0);
    let steer_speed_limit = (turn_rate * 18.0).clamp(min_turn_speed, model.max_control_speed);
    let combo_turn_rate = model.combo_turn_rate_for_speed(turn_key, pose.speed);
    let combo_steer_speed_limit =
        (combo_turn_rate * 18.0).clamp(min_turn_speed, model.max_control_speed);
    let throttle_horizon_s = 0.30_f32;
    let model_predicted_up_speed =
        model.predicted_speed_after_throttle(pose.speed, throttle_horizon_s);
    let model_predicted_combo_speed =
        model.predicted_speed_after_combo_throttle(turn_key, pose.speed, throttle_horizon_s);
    let model_predicted_coast_speed =
        model.predicted_speed_after_coast(pose.speed, throttle_horizon_s, physics_baseline);
    let predicted_up_speed = ground_feedback
        .copied()
        .map(|feedback| {
            feedback.predicted_speed(model_predicted_up_speed, pose.speed, throttle_horizon_s)
        })
        .unwrap_or(model_predicted_up_speed);
    let predicted_combo_speed = ground_feedback
        .copied()
        .map(|feedback| {
            feedback.predicted_speed(model_predicted_combo_speed, pose.speed, throttle_horizon_s)
        })
        .unwrap_or(model_predicted_combo_speed);
    let mut keys = Vec::new();
    let mut reason;
    if distance <= stop_distance + brake_distance
        && matches!(down_effect, DownEffect::Brake { .. })
        && !reverse_velocity
    {
        keys.push(Key::Down);
        reason = format!(
            "target_brake:{brake_distance:.2}:effect={down_effect:?}:heading={heading_source}"
        );
    } else if abs_err > 2.35 {
        // Down is reverse, not a direction-independent brake.  A rear-cone
        // bearing therefore must not select it: once reverse motion starts,
        // unsigned speed plus velocity heading can sustain an accelerating
        // oscillation.  Coast while moving, or use minimum forward traction
        // to rotate from rest.
        keys.push(turn_key);
        if pose.speed < min_turn_speed {
            keys.push(Key::Up);
        }
        reason =
            format!("rear_turn:{turn_rate:.2}:rear_error={abs_err:.3}:heading={heading_source}");
    } else if abs_err > model.hard_turn_rad
        && pose.speed > steer_speed_limit
        && matches!(down_effect, DownEffect::Brake { .. })
        && !reverse_velocity
    {
        keys.push(Key::Down);
        keys.push(turn_key);
        reason = format!("hard_turn_brake:{steer_speed_limit:.2}:heading={heading_source}");
    } else if abs_err > model.hard_turn_rad {
        keys.push(turn_key);
        // Pose-hook rows stop advancing when the kart is physically stuck, so
        // their last speed can remain around 4 m/s even after owner velocity
        // has fallen to zero. Steering-only cannot rotate a stationary kart.
        // Keep minimum throttle through the whole low-speed steering band.
        // KAP's runtime speed-abort is the hard safety guard; suppressing the
        // only traction key here deadlocks on a stale non-zero pose speed.
        if pose.speed < min_turn_speed {
            keys.push(Key::Up);
        }
        reason = format!(
            "hard_turn:{turn_rate:.2}:combo={combo_turn_rate:.2}:pred_combo={predicted_combo_speed:.2}:limit={combo_steer_speed_limit:.2}:heading={heading_source}"
        );
    } else if abs_err > model.soft_turn_rad {
        keys.push(turn_key);
        if predicted_combo_speed <= combo_steer_speed_limit {
            keys.push(Key::Up);
            reason = format!(
                "forward_turn:{turn_rate:.2}:combo={combo_turn_rate:.2}:pred_combo={predicted_combo_speed:.2}:limit={combo_steer_speed_limit:.2}:heading={heading_source}"
            );
        } else {
            reason = format!(
                "turn_hold_speed:{turn_rate:.2}:combo={combo_turn_rate:.2}:pred_combo={predicted_combo_speed:.2}:pred_coast={model_predicted_coast_speed:.2}:limit={combo_steer_speed_limit:.2}:heading={heading_source}"
            );
        }
    } else if pose.speed > model.max_control_speed
        && matches!(down_effect, DownEffect::Brake { .. })
        && !reverse_velocity
    {
        keys.push(Key::Down);
        reason = format!("overspeed_brake:heading={heading_source}");
    } else if pose.speed > model.max_control_speed && !reverse_velocity {
        reason = format!("overspeed_coast:down_effect={down_effect:?}:heading={heading_source}");
    } else if pose.speed > model.max_control_speed {
        reason = format!("reverse_velocity_coast:heading={heading_source}");
    } else {
        keys.push(Key::Up);
        reason = format!("forward:pred_up={predicted_up_speed:.2}:heading={heading_source}");
    }

    DriveDecision {
        keys,
        done: false,
        heading_error,
        reason: if let Some(feedback) = ground_feedback {
            reason.push(':');
            reason.push_str(&feedback.reason_fragment());
            if let Some(baseline) = physics_baseline {
                reason.push(':');
                reason.push_str(&baseline.reason_fragment());
            }
            reason
        } else if let Some(baseline) = physics_baseline {
            reason.push(':');
            reason.push_str(&baseline.reason_fragment());
            reason
        } else {
            reason
        },
    }
}

fn plan_step_mpc(
    model: &VehicleModel,
    physics_baseline: Option<&PhysicsBaseline>,
    path: &KapFollowPath,
    pose: Pose,
    target: P3,
    projection: KapProjection,
    request: &KapFollowRequest,
    ground_feedback: Option<&GroundDriveFeedback>,
    wall_feedback: Option<&WallContactFeedback>,
) -> DriveDecision {
    let dx = target[0] - pose.pos[0];
    let dy = target[1] - pose.pos[1];
    let target_yaw = dy.atan2(dx);
    let (nav_heading, heading_source) = navigation_heading_with_mode(pose, request.heading_mode);
    let heading_error = normalize_angle(target_yaw - nav_heading);
    let turn_key = if heading_error > 0.0 {
        Key::Left
    } else {
        Key::Right
    };
    let horizon_s = 0.35_f32;
    let mut candidates = vec![
        Vec::<Key>::new(),
        vec![Key::Up],
        vec![turn_key],
        vec![Key::Up, turn_key],
    ];
    if matches!(
        model.down_effect_for_speed(pose.speed),
        DownEffect::Brake { .. }
    ) {
        candidates.push(vec![Key::Down]);
        candidates.push(vec![Key::Down, turn_key]);
    }
    let cruise_speed = mpc_cruise_speed(model, request);
    let mut best: Option<(Vec<Key>, f32, f32, f32, f32, String)> = None;
    for first in candidates.iter() {
        for second in candidates.iter() {
            for third in candidates.iter() {
                let sequence = [&first[..], &second[..], &third[..]];
                let mut sim_speed = pose.speed;
                let mut sim_heading_error = heading_error;
                let mut sim_path_s = projection.path_s;
                let mut score = 0.0_f32;
                let mut rollout_progress_s = 0.0_f32;
                let mut first_predicted_speed = pose.speed;
                let mut first_predicted_heading_error = heading_error;
                let mut first_yaw_delta = 0.0_f32;
                let mut previous: &[Key] = &[];
                for (i, keys) in sequence.iter().enumerate() {
                    let step = score_mpc_candidate(
                        model,
                        physics_baseline,
                        wall_feedback,
                        sim_speed,
                        sim_heading_error,
                        projection,
                        request,
                        cruise_speed,
                        keys,
                        horizon_s,
                    );
                    if i == 0 {
                        first_predicted_speed = step.predicted_speed;
                        first_predicted_heading_error = step.predicted_heading_error;
                        first_yaw_delta = step.yaw_delta;
                    }
                    let discount = match i {
                        0 => 1.0,
                        1 => 0.70,
                        _ => 0.45,
                    };
                    let switch_penalty = if i > 0 && !same_keys(previous, keys) {
                        0.04
                    } else {
                        0.0
                    };
                    let step_progress_s =
                        (step.predicted_speed * horizon_s * sim_heading_error.cos().max(0.0))
                            .min(request.lookahead.max(1.0));
                    let next_path_s = sim_path_s + step_progress_s;
                    let future_heading_error = path
                        .heading_at(next_path_s + request.lookahead)
                        .map(|future_heading| normalize_angle(future_heading - nav_heading))
                        .unwrap_or(step.predicted_heading_error);
                    score += step.score * discount + future_heading_error.abs() * 0.25 * discount
                        - step_progress_s * 0.08 * discount
                        + switch_penalty;
                    sim_speed = step.predicted_speed;
                    sim_heading_error = step.predicted_heading_error;
                    sim_path_s = next_path_s;
                    rollout_progress_s += step_progress_s;
                    previous = keys;
                }
                score += sim_heading_error.abs() * 0.35;
                match best {
                    Some((_, best_score, _, _, _, _)) if best_score <= score => {}
                    _ => {
                        best = Some((
                            first.to_vec(),
                            score,
                            first_predicted_speed,
                            first_predicted_heading_error,
                            first_yaw_delta,
                            format!(
                                "{}>{}>{}:s={:.2}",
                                keys_label(first),
                                keys_label(second),
                                keys_label(third),
                                rollout_progress_s
                            ),
                        ))
                    }
                }
            }
        }
    }
    let (keys, score, predicted_speed, predicted_heading_error, yaw_delta, sequence_label) = best
        .unwrap_or_else(|| {
            (
                vec![Key::Up],
                0.0,
                pose.speed,
                heading_error,
                0.0,
                "Up".to_string(),
            )
        });
    let mut reason = format!(
        "mpc:horizon={horizon_s:.2}:rollout=3:score={score:.3}:seq={sequence_label}:pred_speed={predicted_speed:.2}:pred_heading={predicted_heading_error:.3}:yaw_delta={yaw_delta:.3}:heading={heading_source}:cruise={cruise_speed:.2}:mode={}",
        ControlMode::Mpc.name()
    );
    if let Some(feedback) = ground_feedback {
        reason.push(':');
        reason.push_str(&feedback.reason_fragment());
    }
    if let Some(baseline) = physics_baseline {
        reason.push(':');
        reason.push_str(&baseline.reason_fragment());
    }
    DriveDecision {
        keys,
        done: false,
        heading_error,
        reason,
    }
}

#[derive(Clone, Copy, Debug)]
struct MpcCandidateScore {
    score: f32,
    predicted_speed: f32,
    predicted_heading_error: f32,
    yaw_delta: f32,
}

fn mpc_cruise_speed(model: &VehicleModel, request: &KapFollowRequest) -> f32 {
    if request.speed_limit > 0.0 {
        (request.speed_limit * 0.90).max(4.0)
    } else {
        model.max_control_speed * 0.75
    }
}

fn score_mpc_candidate(
    model: &VehicleModel,
    physics_baseline: Option<&PhysicsBaseline>,
    wall_feedback: Option<&WallContactFeedback>,
    speed: f32,
    heading_error: f32,
    projection: KapProjection,
    request: &KapFollowRequest,
    cruise_speed: f32,
    keys: &[Key],
    horizon_s: f32,
) -> MpcCandidateScore {
    let predicted_speed = predict_candidate_speed(model, physics_baseline, speed, horizon_s, keys);
    let turn_rate = if keys.contains(&Key::Left) {
        model.combo_turn_rate_for_speed(Key::Left, speed)
    } else if keys.contains(&Key::Right) {
        model.combo_turn_rate_for_speed(Key::Right, speed)
    } else {
        0.0
    };
    let yaw_delta = if keys.contains(&Key::Left) {
        turn_rate * horizon_s
    } else if keys.contains(&Key::Right) {
        -turn_rate * horizon_s
    } else {
        0.0
    };
    let predicted_heading_error = normalize_angle(heading_error - yaw_delta);
    let overspeed = (predicted_speed - request.speed_limit).max(0.0);
    let underspeed = if predicted_speed < 0.2 {
        0.2 - predicted_speed
    } else {
        0.0
    };
    let cruise_deficit = (cruise_speed - predicted_speed).max(0.0);
    let cross_track_pressure = if request.safe_cross_track > 0.0 {
        (projection.cross_track.abs() / request.safe_cross_track).max(0.0)
    } else {
        0.0
    };
    let steering = keys.contains(&Key::Left) || keys.contains(&Key::Right);
    let high_cross_track_pressure = (cross_track_pressure - 0.55).max(0.0);
    let lateral_recovery_pressure = if high_cross_track_pressure > 0.0 && !steering {
        0.45 + high_cross_track_pressure * 2.0
    } else if high_cross_track_pressure > 0.0 {
        -0.08 - high_cross_track_pressure * 0.20
    } else {
        0.0
    };
    let throttle_bonus = if keys.contains(&Key::Up) && predicted_speed <= request.speed_limit {
        -0.16
    } else {
        0.0
    };
    let wall_risk_throttle_penalty = physics_baseline
        .filter(|baseline| baseline.wall_risk_active(projection, request, speed, heading_error))
        .filter(|_| keys.contains(&Key::Up))
        .map(|baseline| baseline.mpc_wall_risk_throttle_penalty)
        .unwrap_or(0.0);
    let live_wall_throttle_penalty = wall_feedback
        .filter(|feedback| feedback.is_response())
        .filter(|_| keys.contains(&Key::Up))
        .and_then(|_| physics_baseline.map(|baseline| baseline.mpc_wall_risk_throttle_penalty))
        .unwrap_or(0.0);
    let coast_penalty = if keys.is_empty() && speed < cruise_speed {
        0.12
    } else {
        0.0
    };
    let brake_penalty = if keys.contains(&Key::Down) && speed < 2.0 {
        0.35
    } else {
        0.0
    };
    let progress_reward = predicted_speed.min(request.speed_limit.max(0.0)) * 0.04;
    let score = predicted_heading_error.abs() * 2.0
        + overspeed * 0.75
        + underspeed * 1.5
        + cruise_deficit * 0.20
        + cross_track_pressure * 0.15
        + lateral_recovery_pressure
        + brake_penalty
        + coast_penalty
        + throttle_bonus
        + wall_risk_throttle_penalty
        + live_wall_throttle_penalty
        - progress_reward;
    MpcCandidateScore {
        score,
        predicted_speed,
        predicted_heading_error,
        yaw_delta,
    }
}

fn predict_candidate_speed(
    model: &VehicleModel,
    physics_baseline: Option<&PhysicsBaseline>,
    speed: f32,
    horizon_s: f32,
    keys: &[Key],
) -> f32 {
    if keys.contains(&Key::Down) {
        return match model.down_effect_for_speed(speed) {
            DownEffect::Brake { decel } => (speed - decel * horizon_s).max(0.0),
            DownEffect::ReverseOrSettle => {
                (speed + model.down_low_speed_response_accel * horizon_s).max(0.0)
            }
        };
    }
    if keys.contains(&Key::Up) {
        let turn_key = if keys.contains(&Key::Left) {
            Some(Key::Left)
        } else if keys.contains(&Key::Right) {
            Some(Key::Right)
        } else {
            None
        };
        return if let Some(turn_key) = turn_key {
            model.predicted_speed_after_combo_throttle(turn_key, speed, horizon_s)
        } else {
            model.predicted_speed_after_throttle(speed, horizon_s)
        };
    }
    model.predicted_speed_after_coast(speed, horizon_s, physics_baseline)
}

fn apply_kap_safety(
    decision: &mut DriveDecision,
    pose: Pose,
    projection: KapProjection,
    request: &KapFollowRequest,
    physics_baseline: Option<&PhysicsBaseline>,
    wall_feedback: Option<&WallContactFeedback>,
) {
    apply_kap_safety_with_model(
        decision,
        pose,
        projection,
        request,
        &VehicleModel::low_speed_default(),
        physics_baseline,
        wall_feedback,
    );
}

fn apply_kap_safety_with_model(
    decision: &mut DriveDecision,
    pose: Pose,
    projection: KapProjection,
    request: &KapFollowRequest,
    model: &VehicleModel,
    physics_baseline: Option<&PhysicsBaseline>,
    wall_feedback: Option<&WallContactFeedback>,
) {
    let abs_cross_track = projection.cross_track.abs();
    let reverse_velocity = pose_has_reverse_velocity(pose);
    if reverse_velocity {
        decision.keys.retain(|key| *key != Key::Down);
        decision.reason.push_str(":reverse_velocity_down_guard");
        // Directional steering is not passive in every runtime branch.  In
        // particular, the factory route can enter a high-energy reverse-yaw
        // response where a steering-only command grows |v| while the pose
        // heading and velocity heading disagree.  Remove steering and apply
        // forward input as an opposing brake until reverse speed returns to
        // the route's controllable band.
        let reverse_coast_threshold = request.speed_limit.max(4.0);
        if pose.speed > reverse_coast_threshold {
            decision.keys.clear();
            decision.keys.push(Key::Up);
            decision.reason.push_str(&format!(
                ":reverse_velocity_overspeed_forward_brake:{:.3}>{:.3}",
                pose.speed, reverse_coast_threshold
            ));
            return;
        }
    }
    let baseline_wall_risk = physics_baseline
        .map(|baseline| {
            baseline.wall_risk_active(projection, request, pose.speed, decision.heading_error)
        })
        .unwrap_or(false);
    let live_wall_risk = wall_feedback
        .map(|feedback| feedback.is_response())
        .unwrap_or(false);
    let wall_risk = baseline_wall_risk || live_wall_risk;
    if request.panic_cross_track > 0.0 && abs_cross_track > request.panic_cross_track {
        let turn_key = if decision.heading_error > 0.0 {
            Key::Left
        } else {
            Key::Right
        };
        let recovery_speed = request.speed_limit.max(4.0);
        // Panic recovery needs a materially smaller turn radius than normal
        // route control. Waiting until the full route limit (10 m/s on
        // factory_I02) left the kart circling outside the corridor at 9 m/s.
        let brake_speed = (recovery_speed * 0.5).max(4.0);
        decision.keys.clear();
        if pose.speed > brake_speed
            && matches!(
                model.down_effect_for_speed(pose.speed),
                DownEffect::Brake { .. }
            )
            && !reverse_velocity
        {
            decision.keys.push(Key::Down);
        } else if pose.speed > recovery_speed * 0.35 {
            // Once outside the panic corridor, lateral recovery owns the
            // control budget. factory_I02 showed that adding throttle at
            // 7 m/s merely widened the miss from 24 m to 47 m.
        } else if wall_risk {
            // Treat wall risk as a contact-recovery boundary until live wall normals are wired in.
        } else {
            decision.keys.push(Key::Up);
        }
        decision.keys.push(turn_key);
        decision.reason = format!(
            "{}:kap_panic_corridor:{:.3}>{:.3}:recovery_speed={:.3}:brake_speed={:.3}:wall_risk={}:{}",
            decision.reason,
            abs_cross_track,
            request.panic_cross_track,
            recovery_speed,
            brake_speed,
            wall_risk,
            wall_feedback
                .map(|feedback| feedback.reason_tag())
                .unwrap_or_else(|| {
                    if baseline_wall_risk {
                        "wall_feedback=proxy".to_string()
                    } else {
                        "wall_feedback=none".to_string()
                    }
                })
        );
        return;
    }

    if wall_risk {
        let turn_key = if decision.heading_error > 0.0 {
            Key::Left
        } else {
            Key::Right
        };
        decision.keys.retain(|key| *key != Key::Up);
        if !decision.keys.contains(&Key::Down)
            && pose.speed
                < physics_baseline
                    .map(|baseline| baseline.wall_risk_speed_floor)
                    .unwrap_or(2.0)
        {
            decision.keys.insert(0, Key::Up);
        }
        if !decision.keys.contains(&turn_key) {
            decision.keys.push(turn_key);
        }
        decision.reason = format!(
            "{}:kap_wall_risk:{:.3}>={:.3}:heading={:.3}:wall_risk=true:{}",
            decision.reason,
            abs_cross_track,
            request.safe_cross_track
                * physics_baseline
                    .map(|baseline| baseline.wall_risk_cross_track_scale)
                    .unwrap_or(0.85),
            decision.heading_error,
            wall_feedback
                .map(|feedback| feedback.reason_tag())
                .unwrap_or_else(|| "wall_feedback=proxy".to_string())
        );
        return;
    }

    if request.safe_cross_track > 0.0 && abs_cross_track > request.safe_cross_track {
        let turn_key = if decision.heading_error > 0.0 {
            Key::Left
        } else {
            Key::Right
        };
        let recovery_speed = request.speed_limit.max(4.0);
        let brake_speed = recovery_speed * 1.25;
        if pose.speed > brake_speed && !reverse_velocity && !decision.keys.contains(&Key::Down) {
            decision.keys.retain(|key| *key != Key::Up);
            decision.keys.insert(0, Key::Down);
        } else if pose.speed > recovery_speed * 0.35 {
            decision.keys.retain(|key| *key != Key::Up);
        } else if wall_risk {
            decision.keys.retain(|key| *key != Key::Up);
        } else if pose.speed > recovery_speed {
            decision.keys.retain(|key| *key != Key::Up);
        } else if !decision.keys.contains(&Key::Down) && !decision.keys.contains(&Key::Up) {
            decision.keys.insert(0, Key::Up);
        }
        if !decision.keys.contains(&turn_key) {
            decision.keys.push(turn_key);
        }
        decision.reason = format!(
            "{}:kap_safe_corridor:{:.3}>{:.3}:recovery_speed={:.3}:brake_speed={:.3}:wall_risk={}:{}",
            decision.reason,
            abs_cross_track,
            request.safe_cross_track,
            recovery_speed,
            brake_speed,
            wall_risk,
            wall_feedback
                .map(|feedback| feedback.reason_tag())
                .unwrap_or_else(|| {
                    if baseline_wall_risk {
                        "wall_feedback=proxy".to_string()
                    } else {
                        "wall_feedback=none".to_string()
                    }
                })
        );
    }
}

fn pose_has_reverse_velocity(pose: Pose) -> bool {
    pose.speed >= 3.0
        && pose
            .velocity_heading
            .map(|heading| normalize_angle(heading - pose.yaw).abs() > PI / 2.0)
            .unwrap_or(false)
}

fn apply_adaptive_lookahead_speed_guard(
    decision: &mut DriveDecision,
    pose: Pose,
    requested_lookahead: f32,
    effective_lookahead: f32,
    configured_speed_cap: f32,
) {
    apply_adaptive_lookahead_speed_guard_with_model(
        decision,
        pose,
        requested_lookahead,
        effective_lookahead,
        configured_speed_cap,
        &VehicleModel::low_speed_default(),
    );
}

fn apply_adaptive_lookahead_speed_guard_with_model(
    decision: &mut DriveDecision,
    pose: Pose,
    requested_lookahead: f32,
    effective_lookahead: f32,
    configured_speed_cap: f32,
    model: &VehicleModel,
) {
    if decision.done || requested_lookahead <= f32::EPSILON {
        return;
    }
    let ratio = effective_lookahead / requested_lookahead;
    let geometry_speed_cap: f32 = if effective_lookahead >= requested_lookahead - 1.0e-3 {
        // Full lookahead has no extra geometry restriction, but the authored
        // route-zone limit is still a hard control constraint.
        configured_speed_cap.max(0.0)
    } else if ratio <= 0.35 {
        7.0
    } else if ratio <= 0.55 {
        10.0
    } else {
        14.0
    };
    // A geometry guard may tighten a route/profile cap, never relax it.
    // Relaxing a 5 m/s authored zone back to 7 m/s caused factory_I02's
    // descending transition near KAP s=2670 to be entered at 10--12 m/s.
    let speed_cap = geometry_speed_cap.min(configured_speed_cap.max(0.0));
    if pose.speed <= speed_cap {
        return;
    }
    let reverse_velocity = pose_has_reverse_velocity(pose);
    if matches!(
        model.down_effect_for_speed(pose.speed),
        DownEffect::Brake { .. }
    ) && !reverse_velocity
    {
        decision.keys.retain(|key| *key != Key::Up);
        if !decision.keys.contains(&Key::Down) {
            decision.keys.push(Key::Down);
        }
        decision.reason = format!(
            "{}:kap_adaptive_forward_brake:speed={:.3}>cap={:.3}:lookahead={:.3}/{:.3}",
            decision.reason, pose.speed, speed_cap, effective_lookahead, requested_lookahead
        );
    } else if pose.speed >= 5.0 && reverse_velocity {
        decision.keys.retain(|key| *key != Key::Down);
        if !decision.keys.contains(&Key::Up) {
            decision.keys.push(Key::Up);
        }
        decision.reason = format!(
            "{}:kap_adaptive_reverse_brake:speed={:.3}>cap={:.3}:lookahead={:.3}/{:.3}",
            decision.reason, pose.speed, speed_cap, effective_lookahead, requested_lookahead
        );
    } else {
        decision
            .keys
            .retain(|key| *key != Key::Up && *key != Key::Down);
        decision.reason = format!(
            "{}:kap_adaptive_coast_guard:speed={:.3}>cap={:.3}:lookahead={:.3}/{:.3}",
            decision.reason, pose.speed, speed_cap, effective_lookahead, requested_lookahead
        );
    }
}

fn apply_kap_direction_aware_overspeed(decision: &mut DriveDecision, pose: Pose) {
    apply_kap_direction_aware_overspeed_with_model(
        decision,
        pose,
        &VehicleModel::low_speed_default(),
    );
}

fn apply_kap_direction_aware_overspeed_with_model(
    decision: &mut DriveDecision,
    pose: Pose,
    model: &VehicleModel,
) {
    if decision.done || pose.speed <= 3.0 || !decision.reason.contains("overspeed_brake") {
        return;
    }
    let reverse_velocity = pose_has_reverse_velocity(pose);
    if matches!(
        model.down_effect_for_speed(pose.speed),
        DownEffect::Brake { .. }
    ) && !reverse_velocity
    {
        decision.keys.retain(|key| *key != Key::Up);
        if !decision.keys.contains(&Key::Down) {
            decision.keys.push(Key::Down);
        }
        decision.reason = format!(
            "{}:kap_overspeed_forward_brake:speed={:.3}",
            decision.reason, pose.speed
        );
    } else if pose.speed >= 5.0 && reverse_velocity {
        // Longitudinal commands are directional, not signed acceleration
        // requests.  Down brakes established forward motion; Up brakes
        // established reverse motion.  Coasting the latter allowed a
        // factory_I02 reverse-yaw response to grow beyond 20 m/s and leave
        // the elevated route.
        decision.keys.retain(|key| *key != Key::Down);
        if !decision.keys.contains(&Key::Up) {
            decision.keys.push(Key::Up);
        }
        decision.reason = format!(
            "{}:kap_overspeed_reverse_brake:speed={:.3}",
            decision.reason, pose.speed
        );
    } else {
        decision
            .keys
            .retain(|key| *key != Key::Up && *key != Key::Down);
        decision.reason = format!(
            "{}:kap_overspeed_coast:low_speed_guard:speed={:.3}",
            decision.reason, pose.speed
        );
    }
}

fn apply_kap_gap_transfer(
    decision: &mut DriveDecision,
    pose: Pose,
    projection: KapProjection,
    gap: &KapGapWindow,
    request: &KapFollowRequest,
) {
    if decision.done
        || projection.cross_track
            > request
                .panic_cross_track
                .max(request.safe_cross_track)
                .max(3.0)
        || pose_has_reverse_velocity(pose)
    {
        return;
    }

    // An authored KAP arc over a static support hole is an AIR-ENTRY
    // boundary, not an elevation hazard. Braking for the rising trajectory
    // destroys the launch momentum and makes the kart fall through the gap.
    // Preserve longitudinal authority from the approach until the supported
    // landing is reacquired; steering remains owned by the normal path
    // controller.
    decision.keys.retain(|key| *key != Key::Down);
    if pose.speed < gap.required_speed && !decision.keys.contains(&Key::Up) {
        decision.keys.insert(0, Key::Up);
    }
    let phase = if projection.path_s < gap.entry_s {
        "approach"
    } else if projection.path_s <= gap.exit_s {
        "air_entry"
    } else {
        "landing"
    };
    decision.reason = format!(
        "{}:kap_gap_transfer:{phase}:window={:.3}-{:.3}:required_speed={:.3}:speed={:.3}",
        decision.reason, gap.entry_s, gap.exit_s, gap.required_speed, pose.speed
    );
}

fn apply_kap_low_speed_recovery(
    decision: &mut DriveDecision,
    pose: Pose,
    projection: KapProjection,
    request: &KapFollowRequest,
    no_progress_ms: u128,
    physics_baseline: Option<&PhysicsBaseline>,
) {
    let near_zero_threshold = physics_baseline
        .map(|baseline| baseline.low_speed_near_zero_threshold)
        .unwrap_or(0.05);
    let max_recovery_speed = physics_baseline
        .map(|baseline| baseline.low_speed_recovery_max_speed)
        .unwrap_or(0.7);
    if decision.done || request.stuck_timeout_ms == 0 || pose.speed > max_recovery_speed {
        return;
    }
    if request.safe_cross_track > 0.0 && projection.cross_track > request.safe_cross_track {
        return;
    }
    if pose.speed <= near_zero_threshold
        && decision.keys.iter().any(|key| key.is_steering())
        && !decision.keys.iter().any(|key| key.is_longitudinal())
    {
        decision.keys.retain(|key| *key != Key::Down);
        decision.keys.push(Key::Up);
        decision.reason = format!(
            "{}:kap_low_speed_steering_throttle_pulse:speed={:.3}:cross_track={:.3}",
            decision.reason, pose.speed, projection.cross_track
        );
        return;
    }
    let threshold_ms = physics_baseline
        .map(|baseline| baseline.timeout_threshold_ms(request.stuck_timeout_ms))
        .unwrap_or_else(|| (u128::from(request.stuck_timeout_ms) / 2).clamp(500, 1200));
    if no_progress_ms < threshold_ms {
        return;
    }

    decision.keys.retain(|key| *key != Key::Down);
    if !decision.keys.contains(&Key::Up) {
        decision.keys.push(Key::Up);
    }
    decision.reason = format!(
        "{}:kap_low_speed_steering_recover:no_progress={}ms:threshold={}ms:speed={:.3}:cross_track={:.3}",
        decision.reason, no_progress_ms, threshold_ms, pose.speed, projection.cross_track
    );
}

fn apply_kap_wall_stall_recovery(
    decision: &mut DriveDecision,
    pose: Pose,
    no_progress_ms: u128,
    route_forward_recent: bool,
    recovery_latched: &mut bool,
) {
    // A kart can remain inside the nominal KAP corridor while its nose is
    // pinned against a side wall.  Normal corridor recovery then repeatedly
    // applies forward steering into the same contact, and the generic stuck
    // watchdog eventually aborts despite a valid road surface behind it.
    //
    // Use a bounded three-phase manoeuvre only after live/proxy wall safety
    // has already classified the step and forward progress has been absent
    // long enough to exclude an ordinary corner:
    //   1. reverse with opposite lock to pull the nose away from the wall;
    //   2. release longitudinal input so the brake/reverse selector settles;
    //   3. drive forward using the route controller's original steering.
    //
    // The cycle repeats while the progress watchdog remains stale. Any
    // accepted forward KAP progress resets no_progress_ms and immediately
    // returns authority to the normal controller.
    let wall_stall =
        decision.reason.contains("kap_wall_risk") && no_progress_ms >= 1_500 && pose.speed <= 2.5;
    // Contact telemetry is not guaranteed for every authored curb, seam, or
    // low obstacle. If low-speed recovery has made no forward progress for
    // six seconds, use the same bounded manoeuvre without requiring a wall
    // label, but only after corridor safety also reports a violation. Merely
    // slowing inside the corridor is not evidence of an obstacle: reversing
    // there pulled factory_I02 from s=88 back to s=75 for the entire watchdog
    // window. The later/slower/corridor gate preserves normal path recovery.
    let generic_stall = decision.reason.contains("kap_low_speed_steering_recover")
        && decision.reason.contains("kap_safe_corridor")
        && no_progress_ms >= 6_000
        && pose.speed <= 0.7;
    let was_latched = *recovery_latched;
    if route_forward_recent || no_progress_ms < 1_500 {
        *recovery_latched = false;
    } else if wall_stall || generic_stall {
        *recovery_latched = true;
    }
    if decision.done || !*recovery_latched {
        return;
    }

    // Once the escape pulse has accelerated the kart, steering while applying
    // the opposing longitudinal input can sustain a high-speed circle instead
    // of settling the selector.  This was observed on factory_I02 with stale
    // route progress: the latched recovery alternated route/opposite lock at
    // 17.6 m/s for the full watchdog window.  Counter-brake in a straight line
    // first; resume the bounded low-speed manoeuvre only after speed falls.
    if pose.speed >= 4.0 {
        let reverse_velocity = pose_has_reverse_velocity(pose);
        decision.keys.clear();
        decision
            .keys
            .push(if reverse_velocity { Key::Up } else { Key::Down });
        decision.reason = format!(
            "{}:kap_wall_stall_recovery:latched:high_speed_straight_brake:{}:no_progress={}ms:speed={:.3}",
            decision.reason,
            if reverse_velocity { "reverse" } else { "forward" },
            no_progress_ms,
            pose.speed
        );
        return;
    }

    let route_turn = if decision.keys.contains(&Key::Left) {
        Some(Key::Left)
    } else if decision.keys.contains(&Key::Right) {
        Some(Key::Right)
    } else {
        None
    };
    let opposite_turn = match route_turn {
        Some(Key::Left) => Some(Key::Right),
        Some(Key::Right) => Some(Key::Left),
        _ => None,
    };
    let recovery_start_ms = if was_latched || wall_stall {
        1_500
    } else {
        6_000
    };
    let phase_ms = (no_progress_ms - recovery_start_ms) % 3_000;
    decision.keys.clear();
    let phase = if phase_ms < 700 {
        // Keep ownership for the entire manoeuvre.  The old implementation
        // only ran while speed was below the stall threshold, so the first
        // reverse pulse immediately handed a fast backwards kart back to the
        // ordinary forward planner.  That produced alternating 20-30 m/s
        // impacts at factory_I02's moving gear instead of a bounded escape.
        if pose_has_reverse_velocity(pose) && pose.speed >= 4.0 {
            decision.keys.push(Key::Up);
        } else {
            decision.keys.push(Key::Down);
        }
        if let Some(turn) = opposite_turn {
            decision.keys.push(turn);
        }
        if pose_has_reverse_velocity(pose) && pose.speed >= 4.0 {
            "reverse_speed_brake"
        } else {
            "reverse_opposite_lock"
        }
    } else if phase_ms < 1_000 {
        if let Some(turn) = opposite_turn {
            decision.keys.push(turn);
        }
        "settle_opposite_lock"
    } else {
        decision.keys.push(Key::Up);
        if let Some(turn) = route_turn {
            decision.keys.push(turn);
        }
        "forward_route_lock"
    };
    decision.reason = format!(
        "{}:kap_wall_stall_recovery:{}:{}:no_progress={}ms:speed={:.3}",
        decision.reason,
        if wall_stall {
            "wall"
        } else if generic_stall {
            "generic_geometry"
        } else {
            "latched"
        },
        phase,
        no_progress_ms,
        pose.speed
    );
}

impl Key {
    fn is_longitudinal(self) -> bool {
        matches!(self, Key::Up | Key::Down)
    }

    fn is_steering(self) -> bool {
        matches!(self, Key::Left | Key::Right)
    }
}

fn navigation_heading(pose: Pose) -> (f32, &'static str) {
    navigation_heading_with_mode(pose, HeadingMode::Auto)
}

fn pose_with_learned_forward_yaw(
    mut pose: Pose,
    previous_step_was_forward: bool,
    learned_offset: &mut Option<f32>,
) -> Pose {
    // Matrix axes are candidates until correlated with a known forward input.
    // Learn only after an Up (not reverse/brake) step at usable speed.
    if previous_step_was_forward && pose.speed >= 2.0 {
        if let Some(velocity_heading) = pose.velocity_heading {
            let candidate = normalize_angle(velocity_heading - pose.yaw);
            let stale = learned_offset
                .map(|current| normalize_angle(candidate - current).abs() > 0.5)
                .unwrap_or(true);
            if stale {
                *learned_offset = Some(candidate);
            }
        }
    }
    if let Some(offset) = *learned_offset {
        pose.yaw = normalize_angle(pose.yaw + offset);
    }
    pose
}

fn option_angle_changed(before: Option<f32>, after: Option<f32>) -> bool {
    match (before, after) {
        (None, Some(_)) => true,
        (Some(a), Some(b)) => normalize_angle(b - a).abs() > 0.001,
        _ => false,
    }
}

fn read_heading_calibration(path: &Path) -> Result<Option<f32>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)
        .map_err(|err| format!("read heading calibration {}: {err}", path.display()))?;
    for line in text.lines() {
        let mut fields = line.split('\t');
        if fields.next() == Some("offset_rad") {
            let value = fields
                .next()
                .ok_or_else(|| format!("{} missing offset value", path.display()))?
                .parse::<f32>()
                .map_err(|_| format!("{} invalid offset value", path.display()))?;
            return Ok(Some(value));
        }
    }
    Err(format!("{} missing offset_rad row", path.display()))
}

fn write_heading_calibration(path: &Path, offset: f32, pose: Pose) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|err| format!("create heading calibration dir {}: {err}", parent.display()))?;
    let text = format!(
        "schema\tkr_heading_calibration_v1\n\
offset_rad\t{offset:.9}\n\
source\tforward_input_velocity\n\
speed\t{:.9}\n\
tick_ms\t{:.3}\n",
        pose.speed,
        pose.tick_ms.unwrap_or(0.0),
    );
    fs::write(path, text)
        .map_err(|err| format!("write heading calibration {}: {err}", path.display()))
}

fn navigation_heading_with_mode(pose: Pose, mode: HeadingMode) -> (f32, &'static str) {
    // Below a few metres per second the velocity vector is dominated by reset
    // settling, gravity on ramps, collision correction, and quantisation.  It
    // is not a stable proxy for the kart's intended forward direction.  Using
    // it at the old 0.2 m/s threshold made auto mode flip steering (and even
    // select reverse) immediately after a reset or near a rail contact.  Keep
    // owner/pose orientation authoritative until translation is unambiguous.
    const AUTO_VELOCITY_HEADING_MIN_SPEED: f32 = 3.0;
    match mode {
        HeadingMode::PoseYaw => return (pose.yaw, "pose_yaw_forced"),
        HeadingMode::Velocity => {
            if let Some(heading) = pose.velocity_heading {
                return (heading, "velocity_forced");
            }
            return (pose.yaw, "pose_yaw_velocity_missing");
        }
        HeadingMode::Auto => {}
    }
    if pose.speed >= AUTO_VELOCITY_HEADING_MIN_SPEED {
        if let Some(heading) = pose.velocity_heading {
            // Velocity is a direction of travel, not necessarily the kart's
            // forward direction.  In reverse it differs from the calibrated
            // pose yaw by roughly PI; treating that as the navigation heading
            // makes Down look like forward motion and feeds an accelerating
            // reverse loop.  Only use velocity when it agrees with the
            // forward-facing pose hemisphere.
            if normalize_angle(heading - pose.yaw).abs() <= PI / 2.0 {
                return (heading, "velocity");
            }
            return (pose.yaw, "pose_yaw_reverse_velocity");
        }
    }
    (pose.yaw, "pose_yaw")
}

fn col_index(header: &[&str], names: &[&str]) -> Result<usize, String> {
    names
        .iter()
        .find_map(|name| header.iter().position(|col| col == name))
        .ok_or_else(|| format!("missing pose column; expected one of {}", names.join(",")))
}

fn parse_col(cols: &[&str], index: usize, name: &str) -> Result<f32, String> {
    cols.get(index)
        .ok_or_else(|| format!("missing column {name} at index {index}"))?
        .parse::<f32>()
        .map_err(|_| format!("invalid float in column {name}"))
}

fn parse_optional_col(cols: &[&str], index: usize) -> Option<f32> {
    let value = cols.get(index)?.parse::<f32>().ok()?;
    value.is_finite().then_some(value)
}

fn parse_f32_arg(value: &str, name: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("{name} must be a number"))
}

fn parse_bool_arg(value: &str, name: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enabled" => Ok(true),
        "0" | "false" | "no" | "off" | "disabled" => Ok(false),
        _ => Err(format!("{name} must be true or false")),
    }
}

fn parse_u32_arg(value: &str, name: &str) -> Result<u32, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u32::from_str_radix(hex, 16).map_err(|_| format!("{name} must be a u32"))
    } else {
        value
            .parse::<u32>()
            .map_err(|_| format!("{name} must be a u32"))
    }
}

fn normalize_angle(angle: f32) -> f32 {
    if !angle.is_finite() {
        return 0.0;
    }
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::{json_extract_pid_after, self_speed_value};

    fn write_test_kap(path: &Path, points: Vec<P3>) {
        let clear = kart_fmt::kap::Kap {
            points,
            ..Default::default()
        }
        .to_bytes();
        let mut encoded = Vec::with_capacity(4 + clear.len());
        encoded.extend_from_slice(&(clear.len() as i32).to_le_bytes());
        encoded.extend_from_slice(&clear);
        std::fs::write(path, encoded).unwrap();
    }

    #[test]
    fn exact_kap_slot_loads_direct_file_and_ignores_siblings() {
        let dir = std::env::temp_dir().join(format!(
            "kart_exact_kap_slot_{}_{}",
            std::process::id(),
            tick_millis()
        ));
        std::fs::create_dir_all(dir.join("irrelevant/nested")).unwrap();
        write_test_kap(
            &dir.join("factory_I02_6.kap"),
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        );
        std::fs::write(dir.join("irrelevant/nested/broken.kap"), b"broken").unwrap();
        let request = KapFollowRequest {
            kap_dir: Some(dir.clone()),
            track_id: Some("factory_I02".to_string()),
            slot: Some(6),
            ..Default::default()
        };

        let paths = load_kap_follow_paths(&request).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].track_id, "factory_I02");
        assert_eq!(paths[0].slot, Some(6));
        assert_eq!(paths[0].points.len(), 2);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exact_kap_slot_reports_missing_requested_file() {
        let dir = std::env::temp_dir().join(format!(
            "kart_missing_exact_kap_slot_{}_{}",
            std::process::id(),
            tick_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let request = KapFollowRequest {
            kap_dir: Some(dir.clone()),
            track_id: Some("factory_I02".to_string()),
            slot: Some(6),
            ..Default::default()
        };

        let error = load_kap_follow_paths(&request).unwrap_err();
        assert!(error.contains("factory_I02_6.kap"));
        assert!(error.contains("read exact KAP"));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn dynamic_state_angular_velocity_uses_world_shortest_arc() {
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let half = std::f32::consts::FRAC_PI_4;
        let q1 = [half.cos(), 0.0, 0.0, half.sin()];
        let omega = angular_velocity_between(q0, q1, 0.5);
        assert!(omega[0].abs() < 1e-6);
        assert!(omega[1].abs() < 1e-6);
        assert!((omega[2] - std::f32::consts::PI).abs() < 1e-5);

        let negated = [-q1[0], -q1[1], -q1[2], -q1[3]];
        let same_omega = angular_velocity_between(q0, negated, 0.5);
        assert!((same_omega[2] - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn dynamic_state_axes_drop_authored_scale() {
        let matrix = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 6.0, 7.0, 1.0,
        ];
        assert_eq!(
            normalized_matrix_axes(matrix).unwrap(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn china_i04_jm_faces_reconstruct_collision_face_44() {
        let expected = [-0.999964178, 0.00846313965, 5.80824371e-06];
        let faces = [
            [
                [355.261566, 89.0733185, 20.0189095],
                [355.266144, 114.170685, 20.0189095],
                [348.037323, 89.1344604, 20.0189514],
            ],
            [
                [355.266144, 114.170685, 20.0189095],
                [348.041901, 114.231827, 20.0189514],
                [348.037323, 89.1344604, 20.0189514],
            ],
        ];
        for face in faces {
            let actual = shortest_directed_triangle_edge(face);
            for axis in 0..3 {
                assert!((actual[axis] - expected[axis]).abs() < 2e-7);
            }
        }
    }

    #[test]
    fn segment_triangle_intersection_reports_fraction_and_point() {
        let tri = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let (fraction, point) =
            segment_triangle_intersection([0.0, 0.0, 2.0], [0.0, 0.0, -2.0], tri).unwrap();
        assert!((fraction - 0.5).abs() < 1.0e-6);
        assert!(point.iter().all(|value| value.abs() < 1.0e-6));
    }

    #[test]
    fn segment_triangle_intersection_rejects_miss_and_outside_segment() {
        let tri = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(segment_triangle_intersection([2.0, 2.0, 1.0], [2.0, 2.0, -1.0], tri).is_none());
        assert!(segment_triangle_intersection([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], tri).is_none());
    }

    #[test]
    fn swept_sphere_triangle_hits_face_at_radius_offset() {
        let tri = [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]];
        let hit = swept_sphere_triangle_intersection([0.0, 0.0, 2.0], [0.0, 0.0, -2.0], 0.5, tri)
            .unwrap();
        assert!((hit.fraction - 0.375).abs() < 1.0e-6);
        assert!((hit.center[2] - 0.5).abs() < 1.0e-6);
        assert!((hit.point[2]).abs() < 1.0e-6);
        assert!(hit.normal[2] > 0.999);
    }

    #[test]
    fn swept_sphere_triangle_hits_edge_and_vertex() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let edge = swept_sphere_triangle_intersection([1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 0.25, tri)
            .unwrap();
        assert!((edge.fraction - 0.375).abs() < 1.0e-6);
        assert!((edge.point[0] - 1.0).abs() < 1.0e-6 && edge.point[1].abs() < 1.0e-6);
        let vertex =
            swept_sphere_triangle_intersection([-1.0, -1.0, 0.0], [0.2, 0.2, 0.0], 0.2, tri)
                .unwrap();
        assert!(vertex.fraction > 0.70 && vertex.fraction < 0.75);
        assert!(vertex.point.iter().all(|value| value.abs() < 1.0e-6));
    }

    #[test]
    fn swept_sphere_triangle_reports_initial_overlap_and_miss() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let overlap =
            swept_sphere_triangle_intersection([0.5, 0.5, 0.1], [0.5, 0.5, 2.0], 0.2, tri).unwrap();
        assert_eq!(overlap.fraction, 0.0);
        assert!(
            swept_sphere_triangle_intersection([3.0, 3.0, 1.0], [3.0, 3.0, -1.0], 0.25, tri)
                .is_none()
        );
    }

    #[test]
    fn swept_obb_triangle_hits_face_at_half_extent_offset() {
        let tri = [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]];
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let hit = swept_obb_triangle_intersection(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            axes,
            [0.25, 0.25, 0.5],
            tri,
        )
        .unwrap();
        assert!((hit.fraction - 0.375).abs() < 1.0e-6);
        assert!((hit.center[2] - 0.5).abs() < 1.0e-6);
        assert!(hit.normal[2] > 0.999);
        assert!(hit.support_point[2].abs() < 1.0e-6);
    }

    #[test]
    fn swept_obb_triangle_uses_cross_axes_for_edge_separation() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let s = 0.5f32.sqrt();
        let axes = [[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let hit = swept_obb_triangle_intersection(
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            axes,
            [0.2, 0.2, 0.2],
            tri,
        )
        .unwrap();
        assert!(hit.fraction > 0.35 && hit.fraction < 0.37);
        assert!(hit.normal[1] < -0.99);
        assert!(swept_obb_triangle_intersection(
            [3.0, 3.0, 1.0],
            [3.0, 3.0, -1.0],
            axes,
            [0.2, 0.2, 0.2],
            tri
        )
        .is_none());
    }

    #[test]
    fn swept_obb_triangle_reports_initial_overlap_and_rejects_invalid_basis() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let hit = swept_obb_triangle_intersection(
            [0.5, 0.5, 0.1],
            [0.5, 0.5, 2.0],
            axes,
            [0.2, 0.2, 0.2],
            tri,
        )
        .unwrap();
        assert_eq!(hit.fraction, 0.0);
        assert!(hit.normal[2] > 0.999);
        let reversed = swept_obb_triangle_intersection(
            [0.5, 0.5, 0.1],
            [0.5, 0.5, 0.1],
            axes,
            [0.2, 0.2, 0.2],
            [tri[0], tri[2], tri[1]],
        )
        .unwrap();
        assert!(reversed.normal[2] < -0.999);
        assert!(validate_obb([[1.0, 0.0, 0.0]; 3], [1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn swept_rotating_obb_detects_rotation_only_contact() {
        let tri = [[0.8, -2.0, -2.0], [0.8, 2.0, -2.0], [0.8, 0.0, 2.0]];
        let start_axes = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let end_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let hit = swept_rotating_obb_triangle_intersection(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            start_axes,
            end_axes,
            [1.0, 0.1, 0.1],
            tri,
        )
        .unwrap();
        assert!(
            hit.fraction > 0.4 && hit.fraction < 0.8,
            "fraction={}",
            hit.fraction
        );
        assert!(hit.normal[0] < -0.99);
        assert!(
            (hit.support_point[0] - 0.8).abs() < 2.0e-5,
            "fraction={} normal={:?} support={:?}",
            hit.fraction,
            hit.normal,
            hit.support_point
        );
    }

    #[test]
    fn swept_rotating_obb_preserves_fixed_orientation_result_and_miss() {
        let tri = [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]];
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fixed = swept_obb_triangle_intersection(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            axes,
            [0.25, 0.25, 0.5],
            tri,
        )
        .unwrap();
        let rotating = swept_rotating_obb_triangle_intersection(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            axes,
            axes,
            [0.25, 0.25, 0.5],
            tri,
        )
        .unwrap();
        assert!((fixed.fraction - rotating.fraction).abs() < 1.0e-7);
        let far_tri = [[3.0, -2.0, 0.0], [3.0, 2.0, 0.0], [3.0, 0.0, 2.0]];
        assert!(swept_rotating_obb_triangle_intersection(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            axes,
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.1, 0.1],
            far_tri,
        )
        .is_none());
    }

    fn test_pose(yaw: f32, speed: f32, velocity_heading: Option<f32>) -> Pose {
        Pose {
            pos: [0.0, 0.0, 0.0],
            yaw,
            speed,
            velocity_heading,
            tick_ms: None,
        }
    }

    fn test_path() -> KapFollowPath {
        KapFollowPath::new(
            "test.kap".to_string(),
            "test_track".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [250.0, 0.0, 0.0], [500.0, 0.0, 0.0]],
        )
    }

    #[test]
    fn local_lateral_offset_ramps_only_inside_path_window() {
        let path = test_path();
        let request = KapFollowRequest {
            lateral_offset: -16.0,
            offset_start_s: Some(100.0),
            offset_end_s: Some(200.0),
            offset_ramp_m: 20.0,
            ..KapFollowRequest::default()
        };
        assert_eq!(lateral_offset_at(&path, 90.0, &request), 0.0);
        assert!((lateral_offset_at(&path, 110.0, &request) + 8.0).abs() < 0.001);
        assert!((lateral_offset_at(&path, 150.0, &request) + 16.0).abs() < 0.001);
        assert!((lateral_offset_at(&path, 190.0, &request) + 8.0).abs() < 0.001);
        assert_eq!(lateral_offset_at(&path, 210.0, &request), 0.0);
    }

    #[test]
    fn multiple_lateral_offset_zones_own_separate_route_hazards() {
        let path = test_path();
        let request = KapFollowRequest {
            lateral_offset_zones: parse_kap_lateral_offset_zones("100:180:-9:20,300:400:-16:25")
                .unwrap(),
            ..KapFollowRequest::default()
        };

        assert!((lateral_offset_at(&path, 140.0, &request) + 9.0).abs() < 0.001);
        assert_eq!(lateral_offset_at(&path, 240.0, &request), 0.0);
        assert!((lateral_offset_at(&path, 350.0, &request) + 16.0).abs() < 0.001);
    }

    #[test]
    fn lateral_offset_zones_reject_overlap_and_invalid_ranges() {
        assert!(parse_kap_lateral_offset_zones("100:200:-9:20,190:250:-16:20").is_err());
        assert!(parse_kap_lateral_offset_zones("200:100:-9:20").is_err());
        assert!(parse_kap_lateral_offset_zones("100:200:-9").is_err());
    }

    #[test]
    fn negative_lateral_offset_moves_right_of_positive_x_path() {
        let point = test_path().point_at_lateral_offset(150.0, -16.0).unwrap();
        assert!((point[0] - 150.0).abs() < 0.001);
        assert!((point[1] + 16.0).abs() < 0.001);
    }

    #[test]
    fn kap_control_cross_track_uses_actual_offset_path() {
        let path = test_path();
        let request = KapFollowRequest {
            lateral_offset: -16.0,
            offset_start_s: Some(100.0),
            offset_end_s: Some(200.0),
            offset_ramp_m: 20.0,
            ..KapFollowRequest::default()
        };

        let cross_track =
            kap_control_cross_track(&path, [150.0, -14.0, 0.0], 150.0, 14.0, &request);

        assert!((cross_track - 2.0).abs() < 0.001);
    }

    #[test]
    fn kap_control_cross_track_preserves_centerline_outside_offset_window() {
        let path = test_path();
        let request = KapFollowRequest {
            lateral_offset: -16.0,
            offset_start_s: Some(100.0),
            offset_end_s: Some(200.0),
            offset_ramp_m: 20.0,
            ..KapFollowRequest::default()
        };

        assert_eq!(
            kap_control_cross_track(&path, [90.0, -14.0, 0.0], 90.0, 14.0, &request),
            14.0
        );
    }

    #[test]
    fn planner_prefers_velocity_heading_when_moving() {
        let pose = test_pose(0.0, 5.0, Some(PI / 2.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Up]);
        assert!(decision.heading_error.abs() < 0.001);
        assert!(decision.reason.contains("heading=velocity"));
    }

    #[test]
    fn planner_can_force_pose_yaw_heading_when_moving() {
        let pose = test_pose(0.0, 5.0, Some(PI / 2.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step_with_heading_mode(
            &VehicleModel::low_speed_default(),
            pose,
            target,
            1.0,
            HeadingMode::PoseYaw,
        );

        assert!(decision.heading_error > 1.0);
        assert!(decision.reason.contains("heading=pose_yaw_forced"));
    }

    #[test]
    fn planner_falls_back_to_pose_yaw_at_low_speed() {
        let pose = test_pose(0.0, 0.05, Some(PI / 2.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(decision.heading_error > 1.0);
        assert!(decision.reason.contains("heading=pose_yaw"));
    }

    #[test]
    fn planner_ignores_reset_settling_velocity_below_three_mps() {
        let pose = test_pose(0.0, 2.99, Some(PI / 2.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(decision.heading_error > 1.0);
        assert!(decision.reason.contains("heading=pose_yaw"));
    }

    #[test]
    fn planner_uses_velocity_heading_at_three_mps() {
        let pose = test_pose(0.0, 3.0, Some(PI / 2.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Up]);
        assert!(decision.heading_error.abs() < 0.001);
        assert!(decision.reason.contains("heading=velocity"));
    }

    #[test]
    fn planner_rejects_reverse_velocity_as_forward_heading() {
        let pose = test_pose(PI, 16.0, Some(0.0));
        let target = [-100.0, 0.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert!(decision.heading_error.abs() < 0.001);
        assert!(decision
            .reason
            .contains("heading=pose_yaw_reverse_velocity"));
        assert!(!decision.reason.contains("rear_turn_brake"));
    }

    #[test]
    fn planner_hard_turn_keeps_throttle_with_stale_four_mps_pose_speed() {
        let pose = test_pose(0.0, 4.01, Some(0.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&VehicleModel::low_speed_default(), pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(decision.reason.contains("hard_turn"));
    }

    #[test]
    fn planner_turns_forward_when_target_is_directly_behind() {
        let pose = test_pose(PI, 0.1, None);
        let target = [100.0, 0.0, 0.0];
        let decision = plan_step_with_heading_mode(
            &VehicleModel::low_speed_default(),
            pose,
            target,
            1.0,
            HeadingMode::PoseYaw,
        );
        assert!(decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Left) || decision.keys.contains(&Key::Right));
        assert!(!decision.keys.contains(&Key::Down));
        assert!(decision.reason.contains("rear_turn"));

        let reversing_fast = test_pose(PI, 3.0, None);
        let brake = plan_step_with_heading_mode(
            &VehicleModel::low_speed_default(),
            reversing_fast,
            target,
            1.0,
            HeadingMode::PoseYaw,
        );
        assert!(!brake.keys.contains(&Key::Down));
        assert!(brake.keys.contains(&Key::Left) || brake.keys.contains(&Key::Right));
        assert!(brake.reason.contains("rear_turn"));
    }

    #[test]
    fn drive_options_parse_bounded_unstuck_policy() {
        let args = vec![
            "--stuck-timeout-ms".to_string(),
            "1700".to_string(),
            "--unstuck-reverse-ms".to_string(),
            "900".to_string(),
            "--max-unstuck-attempts".to_string(),
            "3".to_string(),
            "--heading-mode".to_string(),
            "pose-yaw".to_string(),
            "--terminal-brake-ms".to_string(),
            "2400".to_string(),
            "--terminal-brake-step-ms".to_string(),
            "60".to_string(),
            "--terminal-stop-speed".to_string(),
            "0.4".to_string(),
            "--terminal-settle-ms".to_string(),
            "1200".to_string(),
            "--cancel-file".to_string(),
            "D:\\kr\\dynout\\experiment_cancel.flag".to_string(),
        ];
        let options = parse_drive_options(&args, "drive-to").unwrap();
        assert_eq!(options.stuck_timeout_ms, 1700);
        assert_eq!(options.unstuck_reverse_ms, 900);
        assert_eq!(options.max_unstuck_attempts, 3);
        assert_eq!(options.heading_mode, HeadingMode::PoseYaw);
        assert_eq!(options.terminal_brake_ms, 2400);
        assert_eq!(options.terminal_brake_step_ms, 60);
        assert!((options.terminal_stop_speed - 0.4).abs() < f32::EPSILON);
        assert_eq!(options.terminal_settle_ms, 1200);
        assert_eq!(
            options.cancel_path,
            Some(PathBuf::from(r"D:\kr\dynout\experiment_cancel.flag"))
        );
    }

    #[test]
    fn terminal_timeout_distinguishes_grade_roll_from_brake_timeout() {
        assert_eq!(terminal_timeout_outcome(0), "brake_timeout");
        assert_eq!(
            terminal_timeout_outcome(1),
            "grade_roll_no_static_equilibrium"
        );
        assert_eq!(
            terminal_timeout_outcome(3),
            "grade_roll_no_static_equilibrium"
        );
    }

    #[test]
    fn terminal_zero_cross_detection_releases_instead_of_reversing() {
        assert!(terminal_crossed_zero(0.766, 2.626));
        assert!(!terminal_crossed_zero(1.01, 2.626));
        assert!(!terminal_crossed_zero(0.766, 0.85));
        assert!(!terminal_crossed_zero(f32::INFINITY, 2.626));
    }

    #[test]
    fn terminal_brake_does_not_flip_on_high_speed_momentum() {
        assert!(!terminal_brake_should_flip(14.0, 18.0));
        assert!(!terminal_brake_should_flip(4.01, 5.0));
        assert!(terminal_brake_should_flip(3.0, 3.09));
        assert!(!terminal_brake_should_flip(3.0, 3.08));
    }

    #[test]
    fn low_speed_kap_uses_short_sparse_longitudinal_pulses() {
        assert_eq!(kap_control_step_ms(160, 5.0), 40);
        assert_eq!(kap_longitudinal_cooldown_ms(100, 5.0), 200);
        assert_eq!(kap_control_step_ms(160, 12.0), 160);
        assert_eq!(kap_longitudinal_cooldown_ms(100, 12.0), 0);
    }

    #[test]
    fn pose_yaw_learns_forward_axis_only_after_forward_motion() {
        let mut offset = None;
        let moving = test_pose(0.0, 5.0, Some(PI / 2.0));
        let calibrated = pose_with_learned_forward_yaw(moving, true, &mut offset);
        assert!((offset.unwrap() - PI / 2.0).abs() < 0.001);
        assert!((calibrated.yaw - PI / 2.0).abs() < 0.001);

        let reverse_like = test_pose(0.0, 5.0, Some(-PI / 2.0));
        let preserved = pose_with_learned_forward_yaw(reverse_like, false, &mut offset);
        assert!((offset.unwrap() - PI / 2.0).abs() < 0.001);
        assert!((preserved.yaw - PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn pose_yaw_replaces_stale_persisted_offset_after_forward_evidence() {
        let mut offset = Some(PI);
        let moving = test_pose(0.0, 5.0, Some(PI / 2.0));
        let calibrated = pose_with_learned_forward_yaw(moving, true, &mut offset);
        assert!((offset.unwrap() - PI / 2.0).abs() < 0.001);
        assert!((calibrated.yaw - PI / 2.0).abs() < 0.001);
        assert!(option_angle_changed(Some(PI), offset));
    }

    #[test]
    fn heading_calibration_round_trips_and_preloaded_value_is_preserved() {
        let path = std::env::temp_dir().join(format!(
            "kart-autoprobe-heading-calibration-{}.tsv",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let pose = test_pose(0.0, 5.0, Some(PI / 2.0));
        write_heading_calibration(&path, PI / 2.0, pose).unwrap();
        let mut offset = read_heading_calibration(&path).unwrap();
        assert!((offset.unwrap() - PI / 2.0).abs() < 0.001);

        let reverse_like = test_pose(0.0, 5.0, Some(-PI / 2.0));
        let calibrated = pose_with_learned_forward_yaw(reverse_like, false, &mut offset);
        assert!((offset.unwrap() - PI / 2.0).abs() < 0.001);
        assert!((calibrated.yaw - PI / 2.0).abs() < 0.001);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn control_samples_keep_calibration_events_and_skip_select_rows() {
        let path =
            std::env::temp_dir().join(format!("kart_control_samples_{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            concat!(
                "time_ms\tevent\tkeys\tduration_ms\n",
                "100\thold_begin\tup\t500\n",
                "600\thold_end\tup\t500\n",
                "600\tsleep\t\t750\n",
                "1350\trelease_all\t\t0\n",
                "1400\ttarget_select\ttarget_id_123\ttrack\tnode\tsurface\t1\trail\tok\t0\t0\t0\t0\t0\t0\n",
                "1450\tkap_select\tD:/kap/path.kap\ttrack\t0\t10\t5.0\ttrue\t12\t0.5\t0\t0\n",
                "1500\tkap_step_begin\tup+left\t160\n",
            ),
        )
        .unwrap();
        let samples = read_control_samples(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let events: Vec<_> = samples.iter().map(|sample| sample.event.as_str()).collect();
        assert_eq!(
            events,
            vec![
                "hold_begin",
                "hold_end",
                "sleep",
                "release_all",
                "kap_step_begin"
            ]
        );
        assert_eq!(samples[0].keys, vec![Key::Up]);
        assert!(samples[2].keys.is_empty());
        assert_eq!(samples[4].keys, vec![Key::Up, Key::Left]);
    }

    #[test]
    fn latest_pose_prefers_latest_primary_then_latest_secondary() {
        let path =
            std::env::temp_dir().join(format!("kart_latest_pose_{}.tsv", std::process::id()));

        std::fs::write(
            &path,
            concat!(
                "tick_ms\tx\ty\tz\tcandidate\n",
                "100\t1.0\t1.0\t0.0\t1\n",
                "200\t2.0\t2.0\t0.0\t0\n",
                "300\t3.0\t3.0\t0.0\t1\n",
            ),
        )
        .unwrap();
        let pose = read_latest_pose(&path).unwrap();
        assert!((pose.pos[0] - 2.0).abs() < 0.001, "latest primary row wins");

        std::fs::write(
            &path,
            concat!(
                "tick_ms\tx\ty\tz\tcandidate\n",
                "100\t1.0\t1.0\t0.0\t1\n",
                "200\t2.0\t2.0\t0.0\t1\n",
                "300\t3.0\t3.0\t0.0\t1\n",
            ),
        )
        .unwrap();
        let pose = read_latest_pose(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            (pose.pos[0] - 3.0).abs() < 0.001,
            "without primary rows, the latest row wins (not the oldest)"
        );
    }

    #[test]
    fn drive_pose_falls_back_to_fresh_owner_when_pose_writer_stalls() {
        assert!(!prefer_owner_pose(Some(100), Some(50)));
        assert!(prefer_owner_pose(Some(501), Some(50)));
        assert!(prefer_owner_pose(None, Some(1_999)));
        assert!(!prefer_owner_pose(Some(501), Some(2_001)));
        assert!(!prefer_owner_pose(Some(501), None));
    }

    #[test]
    fn latest_pose_uses_highest_scored_candidate_at_latest_tick() {
        let path = std::env::temp_dir().join(format!(
            "kart_latest_scored_owner_pose_{}.tsv",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "tick_ms\tpos_x\tpos_y\tpos_z\tcandidate\tcandidate_score\n",
                "100\t10.0\t1.0\t0.0\t1\t9.0\n",
                "200\t20.0\t2.0\t0.0\t0\t3.00003\n",
                "200\t21.0\t2.0\t0.0\t1\t3.00771\n",
                "200\t-341426816\t0.0\t19005800400\t2\t3.15592\n",
            ),
        )
        .unwrap();
        let pose = read_latest_pose(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            (pose.pos[0] - 21.0).abs() < 0.001,
            "latest sample's highest-scored physically plausible owner candidate wins"
        );
    }

    #[test]
    fn latest_pose_rejects_higher_scored_non_orthogonal_matrix() {
        let path = std::env::temp_dir().join(format!(
            "kart_latest_matrix_owner_pose_{}.tsv",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "tick_ms\tpos_x\tpos_y\tpos_z\tcandidate\tcandidate_score\t",
                "m00\tm01\tm02\tm10\tm11\tm12\tm20\tm21\tm22\n",
                "200\t3.4\t-0.3\t8.7\t0\t3.15592\t",
                "0.95\t-1.69\t8.57\t0.61\t-1.65\t8.71\t-1.81\t-1.66\t8.03\n",
                "200\t684.99\t276.63\t32.01\t2\t3.00583\t",
                "-1.0\t-0.0006\t0.0\t0.0006\t-0.9999\t0.015\t0.0\t0.015\t0.9999\n",
            ),
        )
        .unwrap();
        let pose = read_latest_pose(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!((pose.pos[0] - 684.99).abs() < 0.001);
    }

    #[test]
    fn vehicle_model_reads_speed_banded_turn_rates() {
        let path =
            std::env::temp_dir().join(format!("kart_vehicle_model_{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            "max_control_speed\t10\nforward_accel_stationary\t18\nforward_accel_low\t9.5\nforward_accel_low_mid\t7.2\ndown_brake_min_speed\t9\ndown_low_speed_response_accel\t12.5\nbrake_decel_low_mid\t14\nbrake_decel_mid\t13\nturn_rate_low\t1.5\nturn_rate_low_mid\t0.45\nturn_rate_mid\t0.7\nturn_rate_high\t0.35\ncombo_left_accel_low\t8.1\ncombo_left_accel_low_mid\t6.2\ncombo_right_accel_low\t8.3\ncombo_right_accel_low_mid\t6.4\ncombo_left_turn_rate_low\t1.2\ncombo_left_turn_rate_low_mid\t0.55\ncombo_right_turn_rate_low\t1.3\ncombo_right_turn_rate_low_mid\t0.57\n",
        )
        .unwrap();
        let model = VehicleModel::from_tsv(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!((model.forward_accel_for_speed(0.1) - 18.0).abs() < 0.001);
        assert!((model.forward_accel_for_speed(1.0) - 9.5).abs() < 0.001);
        assert!((model.forward_accel_for_speed(3.0) - 7.2).abs() < 0.001);
        match model.down_effect_for_speed(1.0) {
            DownEffect::ReverseOrSettle => {}
            other => panic!("unexpected low-speed down effect: {other:?}"),
        }
        match model.down_effect_for_speed(9.0) {
            DownEffect::Brake { decel } => assert!((decel - 14.0).abs() < 0.001),
            other => panic!("unexpected mid-speed down effect: {other:?}"),
        }
        assert!((model.turn_rate_for_speed(1.0) - 1.5).abs() < 0.001);
        assert!((model.turn_rate_for_speed(3.0) - 0.45).abs() < 0.001);
        assert!((model.turn_rate_for_speed(7.0) - 0.7).abs() < 0.001);
        assert!((model.turn_rate_for_speed(12.0) - 0.35).abs() < 0.001);
        assert!((model.combo_accel_for_speed(Key::Left, 1.0) - 8.1).abs() < 0.001);
        assert!((model.combo_accel_for_speed(Key::Right, 3.0) - 6.4).abs() < 0.001);
        assert!((model.combo_turn_rate_for_speed(Key::Left, 3.0) - 0.55).abs() < 0.001);
        assert!(model
            .serialize_tsv()
            .contains("forward_accel_low\t9.500000"));
        assert!(model
            .serialize_tsv()
            .contains("down_low_speed_response_accel\t12.500000"));
        assert!(model
            .serialize_tsv()
            .contains("down_brake_min_speed\t9.000000"));
        assert!(model.serialize_tsv().contains("turn_rate_high\t0.350000"));
        assert!(model
            .serialize_tsv()
            .contains("combo_left_accel_low_mid\t6.200000"));
    }

    #[test]
    fn planner_uses_speed_banded_turn_rate_in_reason() {
        let mut model = VehicleModel::low_speed_default();
        model.turn_rate_low = 1.5;
        model.turn_rate_low_mid = 0.45;
        model.turn_rate_mid = 0.35;
        model.turn_rate_high = 0.35;
        model.max_control_speed = 12.0;
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let target = [0.0, 100.0, 0.0];
        let decision = plan_step(&model, pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Down, Key::Left]);
        assert!(decision.reason.contains("hard_turn_brake:6.30"));
    }

    #[test]
    fn planner_uses_forward_accel_prediction_to_hold_turn_speed() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 12.0;
        model.soft_turn_rad = 0.1;
        model.hard_turn_rad = 2.0;
        model.turn_rate_low_mid = 0.35;
        model.forward_accel_low_mid = 12.0;
        model.combo_left_turn_rate_low_mid = 0.35;
        model.combo_left_accel_low_mid = 12.0;
        let pose = test_pose(0.0, 3.5, Some(0.0));
        let target = [100.0, 20.0, 0.0];
        let decision = plan_step(&model, pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Left]);
        assert!(decision.reason.contains("turn_hold_speed"));
        assert!(decision.reason.contains("pred_combo=7.10"));
    }

    #[test]
    fn planner_uses_kind40_feedback_for_throttle_prediction() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 12.0;
        model.soft_turn_rad = 0.1;
        model.hard_turn_rad = 2.0;
        model.turn_rate_low_mid = 0.35;
        model.forward_accel_low_mid = 12.0;
        model.combo_left_turn_rate_low_mid = 0.35;
        model.combo_left_accel_low_mid = 12.0;
        let pose = test_pose(0.0, 3.5, Some(0.0));
        let target = [100.0, 20.0, 0.0];
        let feedback = GroundDriveFeedback::from_kind40_rows(
            &[
                GroundDriveKind40Row {
                    tick_ms: 100.0,
                    delta: [-0.05, 0.0, 0.0],
                },
                GroundDriveKind40Row {
                    tick_ms: 116.0,
                    delta: [-0.05, 0.0, 0.0],
                },
            ],
            pose,
        )
        .unwrap();

        let without_feedback = plan_step(&model, pose, target, 1.0);
        let with_feedback = plan_step_with_physics_feedback(
            &model,
            pose,
            target,
            1.0,
            HeadingMode::Auto,
            Some(&feedback),
            None,
        );

        assert_eq!(without_feedback.keys, vec![Key::Left]);
        assert_eq!(with_feedback.keys, vec![Key::Left, Key::Up]);
        assert!(with_feedback.reason.contains("phys=kind40"));
    }

    #[test]
    fn physics_baseline_tsv_drives_piecewise_coast_and_thresholds() {
        let path = std::env::temp_dir().join(format!(
            "physics_baseline_{}_{}.tsv",
            std::process::id(),
            tick_millis()
        ));
        std::fs::write(
            &path,
            "version\ttest_baseline\n\
low_speed_near_zero_threshold\t0.040000\n\
low_speed_recovery_max_speed\t0.600000\n\
low_speed_timeout_min_ms\t400\n\
low_speed_timeout_max_ms\t900\n\
coast_lowproj_k_lt2\t0.100000\n\
coast_lowproj_k_2_5\t0.200000\n\
coast_highproj_k_5_10\t0.300000\n\
coast_highproj_k_10_20\t0.400000\n\
coast_highproj_k_20p\t0.500000\n",
        )
        .unwrap();
        let baseline = PhysicsBaseline::from_tsv(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(baseline.version, "test_baseline");
        assert!((baseline.coast_decel_for_speed(1.0) - 0.1).abs() < 0.001);
        assert!((baseline.coast_decel_for_speed(3.0) - 0.6).abs() < 0.001);
        assert!((baseline.coast_decel_for_speed(7.0) - 2.1).abs() < 0.001);
        assert!((baseline.coast_decel_for_speed(15.0) - 6.0).abs() < 0.001);
        assert!((baseline.coast_decel_for_speed(25.0) - 12.5).abs() < 0.001);
        assert_eq!(baseline.timeout_threshold_ms(3000), 900);
    }

    #[test]
    fn mpc_planner_uses_physics_baseline_and_marks_reason() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 8.0;
        model.turn_rate_low_mid = 0.60;
        model.combo_left_turn_rate_low_mid = 0.60;
        model.combo_left_accel_low_mid = 3.0;
        let baseline = PhysicsBaseline::default_closed_clean();
        let request = KapFollowRequest {
            speed_limit: 8.0,
            safe_cross_track: 6.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 3.0, Some(0.0));
        let decision = plan_step_mpc(
            &model,
            Some(&baseline),
            &test_path(),
            pose,
            [100.0, 20.0, 0.0],
            KapProjection {
                path_s: 10.0,
                cross_track: 1.0,
            },
            &request,
            None,
            None,
        );

        assert!(!decision.keys.is_empty());
        assert!(decision.reason.contains("mpc:horizon"));
        assert!(decision.reason.contains("rollout=3"));
        assert!(decision.reason.contains("seq="));
        assert!(decision.reason.contains(":s="));
        assert!(decision.reason.contains("baseline=builtin_clean_baseline"));
    }

    #[test]
    fn mpc_planner_prefers_throttle_below_cruise_speed() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 18.0;
        model.forward_accel = 10.0;
        let request = KapFollowRequest {
            speed_limit: 12.0,
            safe_cross_track: 6.0,
            heading_mode: HeadingMode::Velocity,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 8.0, Some(0.0));
        let decision = plan_step_mpc(
            &model,
            Some(&PhysicsBaseline::default_closed_clean()),
            &test_path(),
            pose,
            [100.0, 0.0, 0.0],
            KapProjection {
                path_s: 10.0,
                cross_track: 0.1,
            },
            &request,
            None,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Up]);
        assert!(decision.reason.contains("cruise=10.80"));
        assert!(decision.reason.contains("rollout=3"));
    }

    #[test]
    fn ground_drive_feedback_reads_kind40_post_pre_delta() {
        let path = std::env::temp_dir().join(format!(
            "kart_ground_drive_kind40_{}.tsv",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "tick_ms\tkind\tsrc_x\tsrc_y\tsrc_z\tpre_x\tpre_y\tpre_z\tpost_x\tpost_y\tpost_z\n\
100\t40\t0.10\t0.00\t0.00\t1.00\t2.00\t0.00\t1.10\t2.00\t0.00\n\
116\t40\t0.20\t0.00\t0.00\t1.10\t2.00\t0.00\t1.30\t2.00\t0.00\n",
        )
        .unwrap();
        let pose = test_pose(0.0, 3.5, Some(0.0));
        let feedback = read_latest_ground_drive_feedback(&path, pose).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(feedback.kind40_rows, 2);
        assert!((feedback.delta_speed - 0.3).abs() < 0.001);
        assert!(feedback.reason_fragment().contains("phys=kind40"));
    }

    #[test]
    fn oversized_ground_feedback_degrades_without_tail_io() {
        let path = std::env::temp_dir().join(format!(
            "kart_ground_drive_oversized_{}.tsv",
            std::process::id()
        ));
        let file = File::create(&path).unwrap();
        file.set_len(32 * 1024 * 1024 + 1).unwrap();
        let pose = test_pose(0.0, 3.5, Some(0.0));
        assert!(read_latest_ground_drive_feedback(&path, pose).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn planner_uses_combo_curve_for_turn_throttle_prediction() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 12.0;
        model.soft_turn_rad = 0.1;
        model.hard_turn_rad = 2.0;
        model.turn_rate_low_mid = 0.35;
        model.combo_left_turn_rate_low_mid = 0.60;
        model.forward_accel_low_mid = 18.0;
        model.combo_left_accel_low_mid = 3.0;
        let pose = test_pose(0.0, 3.5, Some(0.0));
        let target = [100.0, 20.0, 0.0];
        let decision = plan_step(&model, pose, target, 1.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(decision.reason.contains("forward_turn"));
        assert!(decision.reason.contains("combo=0.60"));
        assert!(decision.reason.contains("pred_combo=4.40"));
    }

    #[test]
    fn kap_request_speed_limit_overrides_model_control_speed() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 10.0;
        let request = KapFollowRequest {
            speed_limit: 12.0,
            ..KapFollowRequest::default()
        };

        apply_request_speed_limit(&mut model, &request);

        assert_eq!(model.max_control_speed, 12.0);
    }

    #[test]
    fn kap_speed_zones_switch_without_restarting_route_state() {
        let zones = parse_kap_speed_zones("600:8,0:20,1500:12").expect("speed zones");
        assert_eq!(zones, vec![(0.0, 20.0), (600.0, 8.0), (1500.0, 12.0)]);
        assert_eq!(kap_speed_limit_at(&zones, 0.0, 5.0), 20.0);
        assert_eq!(kap_speed_limit_at(&zones, 599.9, 5.0), 20.0);
        assert_eq!(kap_speed_limit_at(&zones, 600.0, 5.0), 8.0);
        assert_eq!(kap_speed_limit_at(&zones, 1700.0, 5.0), 12.0);
    }

    #[test]
    fn kap_speed_zones_reject_invalid_or_duplicate_boundaries() {
        assert!(parse_kap_speed_zones("600").is_err());
        assert!(parse_kap_speed_zones("-1:8").is_err());
        assert!(parse_kap_speed_zones("600:8,600:10").is_err());
    }

    #[test]
    fn planner_accepts_low_speed_limit_below_turn_floor() {
        let mut model = VehicleModel::low_speed_default();
        model.max_control_speed = 4.0;
        let pose = test_pose(0.0, 0.0, None);
        let target = [0.0, 20.0, 0.0];

        let decision = plan_step(&model, pose, target, 0.0);

        assert!(!decision.done);
        assert!(decision.reason.contains("limit=4.00"));
    }

    #[test]
    fn kap_safety_removes_throttle_without_braking_near_safe_corridor_speed() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 9.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up],
            done: false,
            heading_error: PI / 4.0,
            reason: "forward".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 6.0,
            },
            &request,
            None,
            None,
        );

        assert!(!decision.keys.contains(&Key::Up));
        assert!(!decision.keys.contains(&Key::Down));
        assert!(decision.keys.contains(&Key::Left));
        assert!(decision.reason.contains("kap_safe_corridor"));
    }

    #[test]
    fn kap_safety_uses_absolute_cross_track_for_negative_offsets() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 9.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up],
            done: false,
            heading_error: -PI / 4.0,
            reason: "forward".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: -6.0,
            },
            &request,
            None,
            None,
        );

        assert!(!decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Right));
        assert!(decision.reason.contains("kap_safe_corridor:6.000>4.000"));
    }

    #[test]
    fn kap_safety_keeps_throttle_when_slow_outside_safe_corridor() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 1.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Right],
            done: false,
            heading_error: -PI / 4.0,
            reason: "hard_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 6.0,
            },
            &request,
            None,
            None,
        );

        assert!(decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Right));
        assert!(!decision.keys.contains(&Key::Down));
        assert!(decision.reason.contains("recovery_speed="));
    }

    #[test]
    fn kap_safety_removes_throttle_under_wall_risk_before_safe_boundary() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let baseline = PhysicsBaseline {
            wall_risk_cross_track_scale: 0.80,
            wall_risk_heading_error_rad: 0.9,
            wall_risk_speed_floor: 2.0,
            ..PhysicsBaseline::default_closed_clean()
        };
        let pose = test_pose(0.0, 3.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -1.1,
            reason: "forward_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 3.4,
            },
            &request,
            Some(&baseline),
            None,
        );

        assert!(!decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Right));
        assert!(decision.reason.contains("wall_risk=true"));
    }

    #[test]
    fn kap_safety_uses_live_wall_feedback_before_proxy_threshold() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 3.0, Some(0.0));
        let feedback = WallContactFeedback {
            kind: 8,
            kind_label: "soft_velocity_add",
            site_rva: "0x0072E6CC".to_string(),
            tick_ms: Some(1200.0),
            hit_normal: [1.0, 0.0, 0.0],
            hit_dot_vel: Some(-0.5),
        };
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.8,
            },
            &request,
            Some(&PhysicsBaseline::default_closed_clean()),
            Some(&feedback),
        );

        assert!(!decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Right));
        assert!(decision.reason.contains("wall_feedback=soft_velocity_add"));
    }

    #[test]
    fn wall_feedback_requires_pose_tick_window_when_available() {
        assert!(wall_response_fresh_for_pose(Some(1_000.0), Some(1_500.0)));
        assert!(wall_response_fresh_for_pose(Some(1_010.0), Some(1_000.0)));
        assert!(!wall_response_fresh_for_pose(Some(900.0), Some(1_500.5)));
        assert!(!wall_response_fresh_for_pose(Some(1_100.0), Some(1_000.0)));
        assert!(wall_response_fresh_for_pose(Some(1_000.0), None));
        assert!(wall_response_fresh_for_pose(None, Some(1_000.0)));
    }

    #[test]
    fn fixed_key_sequence_returns_to_planner_after_last_stage() {
        let steps = vec![
            KeySequenceStep {
                keys: vec![Key::Up, Key::Right],
                duration_ms: 100,
            },
            KeySequenceStep {
                keys: Vec::new(),
                duration_ms: 50,
            },
        ];

        assert_eq!(active_key_sequence_step(&steps, 0).unwrap().0, 0);
        assert_eq!(active_key_sequence_step(&steps, 99).unwrap().0, 0);
        assert_eq!(active_key_sequence_step(&steps, 100).unwrap().0, 1);
        assert_eq!(active_key_sequence_step(&steps, 149).unwrap().0, 1);
        assert!(active_key_sequence_step(&steps, 150).is_none());
    }

    #[test]
    fn fixed_add_up_preserves_steering_and_replaces_brake() {
        let mut decision = DriveDecision {
            keys: vec![Key::Down, Key::Left],
            done: false,
            heading_error: 0.0,
            reason: "planner".to_string(),
        };

        apply_fixed_add_keys(&mut decision, &[Key::Up]);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(decision.reason.contains("fixed_add_keys=Up"));
    }

    #[test]
    fn fixed_add_sequence_can_delay_throttle_without_overriding_planner() {
        let steps = parse_key_sequence("None:1800,Up:5000").unwrap();
        assert!(active_key_sequence_step(&steps, 1799)
            .unwrap()
            .1
            .keys
            .is_empty());
        assert_eq!(
            active_key_sequence_step(&steps, 1800).unwrap().1.keys,
            vec![Key::Up]
        );
    }

    #[test]
    fn mpc_wall_risk_penalizes_throttle_candidate() {
        let model = VehicleModel::low_speed_default();
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let baseline = PhysicsBaseline {
            wall_risk_cross_track_scale: 0.80,
            wall_risk_heading_error_rad: 0.9,
            wall_risk_speed_floor: 2.0,
            mpc_wall_risk_throttle_penalty: 0.6,
            ..PhysicsBaseline::default_closed_clean()
        };
        let projection = KapProjection {
            path_s: 10.0,
            cross_track: 3.4,
        };
        let without_baseline = score_mpc_candidate(
            &model,
            None,
            None,
            3.0,
            -1.1,
            projection,
            &request,
            7.0,
            &[Key::Up, Key::Right],
            0.35,
        );
        let with_baseline = score_mpc_candidate(
            &model,
            Some(&baseline),
            None,
            3.0,
            -1.1,
            projection,
            &request,
            7.0,
            &[Key::Up, Key::Right],
            0.35,
        );
        let live_feedback = WallContactFeedback {
            kind: 8,
            kind_label: "soft_velocity_add",
            site_rva: "0x0072E6CC".to_string(),
            tick_ms: Some(1200.0),
            hit_normal: [1.0, 0.0, 0.0],
            hit_dot_vel: Some(-0.5),
        };
        let with_live_feedback = score_mpc_candidate(
            &model,
            Some(&baseline),
            Some(&live_feedback),
            3.0,
            -0.2,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.1,
            },
            &request,
            7.0,
            &[Key::Up, Key::Right],
            0.35,
        );
        let without_live_feedback = score_mpc_candidate(
            &model,
            Some(&baseline),
            None,
            3.0,
            -0.2,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.1,
            },
            &request,
            7.0,
            &[Key::Up, Key::Right],
            0.35,
        );

        assert!(with_baseline.score > without_baseline.score + 0.5);
        assert!(with_live_feedback.score > without_live_feedback.score + 0.5);
    }

    #[test]
    fn kap_low_speed_recovery_uses_forward_throttle_inside_corridor() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            stuck_timeout_ms: 1500,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(PI, 0.0, None);
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Up],
            done: false,
            heading_error: -PI / 2.0,
            reason: "hard_turn:heading=pose_yaw".to_string(),
        };

        apply_kap_low_speed_recovery(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.8,
            },
            &request,
            900,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(decision.reason.contains("kap_low_speed_steering_recover"));
        assert!(
            !kap_throttle_cooldown_applies(&decision),
            "sustained recovery throttle must not be removed by sparse-pulse cooldown"
        );
    }

    #[test]
    fn kap_material_progress_ignores_wall_creep_but_accepts_real_gain() {
        assert!(!kap_material_progress_advanced(101.99, 100.0));
        assert!(kap_material_progress_advanced(102.0, 100.0));
        assert!(!kap_material_progress_advanced(f32::NAN, 100.0));
    }

    #[test]
    fn kap_wall_stall_recovery_reverses_with_opposite_lock() {
        let pose = test_pose(0.0, 0.95, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward:kap_wall_risk:wall_feedback=soft_velocity_add".to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 1_800, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Down, Key::Left]);
        assert!(decision.reason.contains("reverse_opposite_lock"));
        assert!(latched);
    }

    #[test]
    fn kap_wall_stall_recovery_returns_to_route_lock_after_settle() {
        let pose = test_pose(0.0, 0.4, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Left],
            done: false,
            heading_error: 0.4,
            reason: "forward:kap_wall_risk:wall_feedback=soft_angular_add0".to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 2_700, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Up, Key::Left]);
        assert!(decision.reason.contains("forward_route_lock"));
    }

    #[test]
    fn kap_wall_stall_recovery_retains_authority_after_reverse_accelerates() {
        let mut latched = false;
        let mut initial = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward:kap_wall_risk:wall_feedback=soft_velocity_add".to_string(),
        };
        apply_kap_wall_stall_recovery(
            &mut initial,
            test_pose(0.0, 0.5, Some(0.0)),
            1_800,
            false,
            &mut latched,
        );
        assert!(latched);

        let mut accelerated = DriveDecision {
            keys: vec![Key::Up, Key::Left],
            done: false,
            heading_error: 0.3,
            reason: "ordinary_route_controller".to_string(),
        };
        let reverse_pose = test_pose(0.0, 12.0, Some(PI));
        apply_kap_wall_stall_recovery(&mut accelerated, reverse_pose, 7_700, false, &mut latched);

        assert_eq!(accelerated.keys, vec![Key::Up]);
        assert!(accelerated
            .reason
            .contains("latched:high_speed_straight_brake:reverse"));
    }

    #[test]
    fn kap_wall_stall_recovery_brakes_forward_speed_without_steering() {
        let mut latched = true;
        let mut accelerated = DriveDecision {
            keys: vec![Key::Up, Key::Left],
            done: false,
            heading_error: 0.3,
            reason: "ordinary_route_controller".to_string(),
        };
        let forward_pose = test_pose(0.0, 12.0, Some(0.0));
        apply_kap_wall_stall_recovery(&mut accelerated, forward_pose, 7_700, false, &mut latched);

        assert_eq!(accelerated.keys, vec![Key::Down]);
        assert!(accelerated
            .reason
            .contains("latched:high_speed_straight_brake:forward"));
    }

    #[test]
    fn kap_wall_stall_recovery_latch_clears_on_material_progress() {
        let mut latched = true;
        let mut decision = DriveDecision {
            keys: vec![Key::Up],
            done: false,
            heading_error: 0.0,
            reason: "forward".to_string(),
        };

        apply_kap_wall_stall_recovery(
            &mut decision,
            test_pose(0.0, 4.0, Some(0.0)),
            0,
            false,
            &mut latched,
        );

        assert!(!latched);
        assert_eq!(decision.keys, vec![Key::Up]);
    }

    #[test]
    fn kap_wall_stall_recovery_latch_clears_on_recent_local_forward_progress() {
        let mut latched = true;
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Left],
            done: false,
            heading_error: 0.2,
            reason: "ordinary_route_controller".to_string(),
        };

        apply_kap_wall_stall_recovery(
            &mut decision,
            test_pose(0.0, 1.5, Some(0.0)),
            20_000,
            true,
            &mut latched,
        );

        assert!(!latched);
        assert_eq!(decision.keys, vec![Key::Up, Key::Left]);
        assert!(!decision.reason.contains("kap_wall_stall_recovery"));
    }

    #[test]
    fn kap_wall_stall_recovery_does_not_override_normal_cornering() {
        let pose = test_pose(0.0, 0.95, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward:kap_wall_risk:wall_feedback=soft_velocity_add".to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 1_200, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Up, Key::Right]);
        assert!(!decision.reason.contains("kap_wall_stall_recovery"));
    }

    #[test]
    fn kap_generic_geometry_stall_reverses_without_wall_telemetry() {
        let pose = test_pose(0.0, 0.25, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason:
                "forward_turn:kap_safe_corridor:kap_low_speed_steering_recover:no_progress=6200ms"
                    .to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 6_200, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Down, Key::Left]);
        assert!(decision
            .reason
            .contains("generic_geometry:reverse_opposite_lock"));
    }

    #[test]
    fn kap_generic_geometry_stall_does_not_reverse_inside_safe_corridor() {
        let pose = test_pose(0.0, 0.25, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward_turn:kap_low_speed_steering_recover:no_progress=6200ms".to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 6_200, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Up, Key::Right]);
        assert!(!latched);
        assert!(!decision.reason.contains("kap_wall_stall_recovery"));
    }

    #[test]
    fn kap_generic_geometry_stall_keeps_normal_recovery_before_six_seconds() {
        let pose = test_pose(0.0, 0.25, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -0.4,
            reason: "forward_turn:kap_low_speed_steering_recover:no_progress=5900ms".to_string(),
        };

        let mut latched = false;
        apply_kap_wall_stall_recovery(&mut decision, pose, 5_900, false, &mut latched);

        assert_eq!(decision.keys, vec![Key::Up, Key::Right]);
        assert!(!decision.reason.contains("kap_wall_stall_recovery"));
    }

    #[test]
    fn kap_adaptive_speed_guard_brakes_before_sharp_geometry() {
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Left, Key::Up],
            done: false,
            heading_error: 0.4,
            reason: "forward_turn".to_string(),
        };

        apply_adaptive_lookahead_speed_guard(&mut decision, pose, 24.0, 7.2, 20.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Down]);
        assert!(decision.reason.contains("kap_adaptive_forward_brake"));
    }

    #[test]
    fn kap_adaptive_speed_guard_keeps_throttle_below_cap() {
        let pose = test_pose(0.0, 6.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Left, Key::Up],
            done: false,
            heading_error: 0.4,
            reason: "forward_turn".to_string(),
        };

        apply_adaptive_lookahead_speed_guard(&mut decision, pose, 24.0, 7.2, 20.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Up]);
        assert!(!decision.reason.contains("kap_adaptive_coast_guard"));
    }

    #[test]
    fn kap_adaptive_speed_guard_never_relaxes_route_zone_cap() {
        let pose = test_pose(0.0, 6.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Left, Key::Up],
            done: false,
            heading_error: 0.4,
            reason: "forward_turn".to_string(),
        };

        apply_adaptive_lookahead_speed_guard(&mut decision, pose, 24.0, 7.2, 5.0);

        assert_eq!(decision.keys, vec![Key::Left]);
        assert!(decision.reason.contains("kap_adaptive_coast_guard"));
        assert!(decision.reason.contains("cap=5.000"));
    }

    #[test]
    fn kap_speed_guard_enforces_route_zone_cap_at_full_lookahead() {
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Left, Key::Up],
            done: false,
            heading_error: 0.4,
            reason: "mpc:cruise=4.00".to_string(),
        };

        apply_adaptive_lookahead_speed_guard(&mut decision, pose, 12.0, 12.0, 4.0);

        assert_eq!(decision.keys, vec![Key::Left, Key::Down]);
        assert!(decision.reason.contains("kap_adaptive_forward_brake"));
        assert!(decision.reason.contains("cap=4.000"));
        assert!(decision.reason.contains("lookahead=12.000/12.000"));
    }

    #[test]
    fn kap_adaptive_speed_guard_uses_up_to_brake_reverse_velocity() {
        let pose = test_pose(0.0, 15.0, Some(PI));
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Down],
            done: false,
            heading_error: 0.2,
            reason: "forward_turn".to_string(),
        };

        apply_adaptive_lookahead_speed_guard(&mut decision, pose, 24.0, 7.2, 10.0);

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(decision.reason.contains("kap_adaptive_reverse_brake"));
    }

    #[test]
    fn kap_overspeed_uses_down_for_established_forward_motion() {
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Down],
            done: false,
            heading_error: 0.2,
            reason: "overspeed_brake:heading=velocity_forced".to_string(),
        };

        apply_kap_direction_aware_overspeed(&mut decision, pose);

        assert_eq!(decision.keys, vec![Key::Right, Key::Down]);
        assert!(decision.reason.contains("kap_overspeed_forward_brake"));
    }

    #[test]
    fn kap_overspeed_uses_up_to_brake_reverse_velocity() {
        let pose = test_pose(0.0, 15.0, Some(PI));
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Down],
            done: false,
            heading_error: 0.2,
            reason: "overspeed_brake:heading=pose_yaw_reverse_velocity".to_string(),
        };

        apply_kap_direction_aware_overspeed(&mut decision, pose);

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(decision.reason.contains("kap_overspeed_reverse_brake"));
    }

    #[test]
    fn kap_forward_brake_crosses_observed_actuator_latency() {
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let decision = DriveDecision {
            keys: vec![Key::Right, Key::Down],
            done: false,
            heading_error: 0.2,
            reason: "kap_overspeed_forward_brake".to_string(),
        };

        assert_eq!(kap_applied_step_ms(160, &decision, pose), 320);

        let zone_pose = test_pose(0.0, 7.0, Some(0.0));
        assert_eq!(kap_applied_step_ms(20, &decision, zone_pose), 180);
    }

    #[test]
    fn kap_reverse_down_does_not_extend_brake_pulse() {
        let pose = test_pose(0.0, 15.0, Some(PI));
        let decision = DriveDecision {
            keys: vec![Key::Right, Key::Down],
            done: false,
            heading_error: 0.2,
            reason: "reverse recovery".to_string(),
        };

        assert_eq!(kap_applied_step_ms(160, &decision, pose), 160);
    }

    #[test]
    fn kap_low_speed_recovery_replaces_near_zero_steering_only() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            stuck_timeout_ms: 1500,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(PI, 0.03, None);
        let mut decision = DriveDecision {
            keys: vec![Key::Right],
            done: false,
            heading_error: -PI / 2.0,
            reason: "hard_turn:heading=pose_yaw".to_string(),
        };

        apply_kap_low_speed_recovery(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.8,
            },
            &request,
            0,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(decision
            .reason
            .contains("kap_low_speed_steering_throttle_pulse"));
    }

    #[test]
    fn kap_low_speed_recovery_preserves_near_zero_combo_before_timeout() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            stuck_timeout_ms: 1500,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(PI, 0.03, None);
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Up],
            done: false,
            heading_error: -PI / 2.0,
            reason: "hard_turn:heading=pose_yaw".to_string(),
        };

        apply_kap_low_speed_recovery(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.8,
            },
            &request,
            0,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(!decision.reason.contains("kap_low_speed_longitudinal_pulse"));
    }

    #[test]
    fn kap_low_speed_recovery_leaves_corridor_recovery_to_safety() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            stuck_timeout_ms: 1500,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(PI, 0.0, None);
        let mut decision = DriveDecision {
            keys: vec![Key::Right, Key::Up],
            done: false,
            heading_error: -PI / 2.0,
            reason: "hard_turn:heading=pose_yaw".to_string(),
        };

        apply_kap_low_speed_recovery(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 6.0,
            },
            &request,
            900,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Right, Key::Up]);
        assert!(!decision.reason.contains("kap_low_speed_recover"));
    }

    #[test]
    fn kap_safety_coasts_and_turns_when_down_would_select_reverse() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 8.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 7.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -PI / 3.0,
            reason: "forward_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 12.0,
            },
            &request,
            None,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Right]);
        assert!(decision.reason.contains("kap_panic_corridor"));
        assert!(decision.reason.contains("brake_speed=4.000"));
    }

    #[test]
    fn kap_safety_brakes_in_panic_corridor_above_brake_speed() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 8.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 15.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Up, Key::Right],
            done: false,
            heading_error: -PI / 3.0,
            reason: "forward_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 12.0,
            },
            &request,
            None,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Down, Key::Right]);
        assert!(decision.reason.contains("kap_panic_corridor"));
    }

    #[test]
    fn kap_safety_applies_up_to_brake_reverse_velocity() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 8.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(PI, 16.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Right],
            done: false,
            heading_error: -2.5,
            reason: "rear_turn:heading=pose_yaw_reverse_velocity".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 12.0,
            },
            &request,
            None,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Up]);
        assert!(!decision.keys.contains(&Key::Down));
        assert!(decision
            .reason
            .contains("reverse_velocity_overspeed_forward_brake"));
        assert!(!decision.reason.contains("kap_panic_corridor"));
    }

    #[test]
    fn accepted_forward_kap_progress_overrides_ambiguous_reverse_axis_for_safety() {
        let ambiguous = test_pose(PI, 16.0, Some(0.0));
        assert!(pose_has_reverse_velocity(ambiguous));

        let authoritative = kap_direction_authority_pose(ambiguous, true, false);
        assert!(
            !pose_has_reverse_velocity(authoritative),
            "accepted forward cursor progress must restore braking authority"
        );
        assert!(kap_direction_authority_pose(ambiguous, false, false)
            .velocity_heading
            .is_some());

        let stalled = kap_direction_authority_pose(ambiguous, false, true);
        assert_eq!(stalled.speed, 0.0);
        assert!(stalled.velocity_heading.is_none());
    }

    #[test]
    fn kap_safety_does_not_combine_up_and_down_in_safe_corridor() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 12.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 9.0, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Down, Key::Right],
            done: false,
            heading_error: -PI / 8.0,
            reason: "overspeed_brake".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 6.0,
            },
            &request,
            None,
            None,
        );

        assert!(decision.keys.contains(&Key::Down));
        assert!(!decision.keys.contains(&Key::Up));
        assert!(decision.keys.contains(&Key::Right));
        assert!(decision.reason.contains("kap_safe_corridor"));
    }

    #[test]
    fn kap_safety_recovers_with_throttle_when_slow_in_panic_corridor() {
        let request = KapFollowRequest {
            safe_cross_track: 4.0,
            panic_cross_track: 8.0,
            speed_limit: 8.0,
            ..KapFollowRequest::default()
        };
        let pose = test_pose(0.0, 0.1, Some(0.0));
        let mut decision = DriveDecision {
            keys: vec![Key::Right],
            done: false,
            heading_error: -PI / 3.0,
            reason: "forward_turn".to_string(),
        };

        apply_kap_safety(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 12.0,
            },
            &request,
            None,
            None,
        );

        assert_eq!(decision.keys, vec![Key::Up, Key::Right]);
        assert!(decision.reason.contains("kap_panic_corridor"));
        assert!(decision.reason.contains("recovery_speed="));
    }

    #[test]
    fn pose_parser_prefers_orientation_matrix_yaw() {
        let header = [
            "tick_ms",
            "post_pos_x",
            "post_pos_y",
            "post_pos_z",
            "post_m01",
            "post_m11",
            "post_ang_z",
            "speed",
        ];
        let cols = ["1", "835.0", "467.0", "43.0", "1.0", "0.0", "0.0", "0.0"];
        let pose_cols = PoseColumns::from_header(&header).expect("pose columns");
        let pose = pose_cols.parse_pose(&cols).expect("pose");

        assert!(
            (pose.yaw.abs() - PI).abs() < 0.001,
            "matrix yaw should win over post_ang_z: {:?}",
            pose
        );
    }

    #[test]
    fn kap_path_projects_and_wraps_lookahead() {
        let path = KapFollowPath::new(
            "square_1.kap".to_string(),
            "square".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );

        assert!(path.closed);
        assert!((path.length - 40.0).abs() < 0.001);
        let projection = path.project([5.0, 2.0, 0.0]).expect("projection");
        assert!((projection.path_s - 5.0).abs() < 0.001);
        assert!((projection.cross_track - 2.0).abs() < 0.001);
        let target = path.point_at(projection.path_s + 38.0).expect("target");
        assert!((target[0] - 3.0).abs() < 0.001);
        assert!(target[1].abs() < 0.001);
    }

    #[test]
    fn kap_adaptive_lookahead_shortens_before_hairpin() {
        let path = KapFollowPath::new(
            "hairpin.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [12.0, 8.0, 0.0],
                [4.0, 8.0, 0.0],
            ],
        );

        let lookahead = adaptive_kap_lookahead(&path, 0.0, 24.0);
        assert!(
            (lookahead - 7.2).abs() < 0.001,
            "hairpin should select the strongest reduction: {lookahead}"
        );
    }

    #[test]
    fn kap_adaptive_lookahead_shortens_before_elevation_change() {
        let path = KapFollowPath::new(
            "ramp.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [16.0, 0.0, 3.0],
                [32.0, 0.0, 3.0],
            ],
        );

        let lookahead = adaptive_kap_lookahead(&path, 0.0, 24.0);
        assert!(
            (lookahead - 7.2).abs() < 0.001,
            "large elevation change should select the strongest reduction: {lookahead}"
        );
    }

    #[test]
    fn kap_adaptive_lookahead_preserves_straight_flat_route() {
        let path = KapFollowPath::new(
            "straight.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        );

        assert!((adaptive_kap_lookahead(&path, 10.0, 24.0) - 24.0).abs() < 0.001);
    }

    #[test]
    fn kap_short_target_lookahead_still_detects_upcoming_hairpin() {
        let path = KapFollowPath::new(
            "short-lookahead-hairpin.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [12.0, 8.0, 0.0],
                [4.0, 8.0, 0.0],
            ],
        );

        let lookahead = adaptive_kap_lookahead(&path, 0.0, 4.0);
        assert!(
            (lookahead - 2.0).abs() < 0.001,
            "a 4 m steering target must retain the independent 24 m hazard horizon: {lookahead}"
        );
    }

    #[test]
    fn kap_project_near_keeps_cursor_on_local_branch() {
        let path = KapFollowPath::new(
            "parallel.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
                [200.0, 10.0, 0.0],
                [100.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );

        let global = path.project([50.0, 8.0, 0.0]).expect("global projection");
        assert!(
            global.path_s > 300.0,
            "global projection jumped to later branch: {global:?}"
        );
        let local = path
            .project_near([50.0, 8.0, 0.0], 50.0, 80.0)
            .expect("local projection");
        assert!(
            (local.path_s - 50.0).abs() < 0.001,
            "local projection should stay near cursor: {local:?}"
        );
    }

    #[test]
    fn kap_projection_rejects_remote_stacked_road_candidate() {
        let request = KapFollowRequest {
            safe_cross_track: 6.0,
            panic_cross_track: 12.0,
            max_cross_track: 100.0,
            ..KapFollowRequest::default()
        };

        assert!(!kap_projection_is_remote(
            KapProjection {
                path_s: 2600.0,
                cross_track: 19.99,
            },
            18.0,
            &request,
        ));
        assert!(kap_projection_is_remote(
            KapProjection {
                path_s: 2660.0,
                cross_track: 20.01,
            },
            18.0,
            &request,
        ));
        assert!(!kap_projection_is_remote(
            KapProjection {
                path_s: 2724.0,
                cross_track: 47.0,
            },
            0.2,
            &request,
        ));
        assert!(!kap_projection_is_remote(
            KapProjection {
                path_s: 3889.0,
                cross_track: 50.0,
            },
            9.7,
            &request,
        ));
        assert!(kap_projection_is_remote(
            KapProjection {
                path_s: 2724.0,
                cross_track: 101.0,
            },
            0.2,
            &request,
        ));
    }

    #[test]
    fn kap_recovery_projection_allows_only_materially_better_local_regression() {
        let path = KapFollowPath::new(
            "recovery.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
        );
        let directional = KapProjection {
            path_s: 164.0,
            cross_track: 30.0,
        };
        assert!(kap_recovery_projection_is_better(
            &path,
            KapProjection {
                path_s: 136.0,
                cross_track: 2.0,
            },
            directional,
            169.0,
            60.0,
            8.0,
        ));
        assert!(!kap_recovery_projection_is_better(
            &path,
            KapProjection {
                path_s: 136.0,
                cross_track: 24.0,
            },
            directional,
            169.0,
            60.0,
            8.0,
        ));
        assert!(!kap_recovery_projection_is_better(
            &path,
            KapProjection {
                path_s: 90.0,
                cross_track: 1.0,
            },
            directional,
            169.0,
            60.0,
            8.0,
        ));
    }

    #[test]
    fn kap_recovery_projection_cannot_cross_confirmed_downward_topology_floor() {
        let path = KapFollowPath::new(
            "stacked-transfer.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 50.0],
                [100.0, 0.0, 50.0],
                [200.0, 0.0, 32.0],
                [300.0, 0.0, 32.0],
            ],
        );
        let floor_s = Some(200.0);

        assert!(!kap_recovery_projection_respects_topology_floor(
            &path,
            KapProjection {
                path_s: 150.0,
                cross_track: 0.5,
            },
            floor_s,
            60.0,
        ));
        assert!(kap_recovery_projection_respects_topology_floor(
            &path,
            KapProjection {
                path_s: 225.0,
                cross_track: 3.0,
            },
            floor_s,
            60.0,
        ));
        assert!(kap_recovery_projection_respects_topology_floor(
            &path,
            KapProjection {
                path_s: 150.0,
                cross_track: 0.5,
            },
            None,
            60.0,
        ));
    }

    #[test]
    fn kap_frontier_progress_does_not_double_count_regression_and_rewalk() {
        let path = KapFollowPath::new(
            "frontier.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
        );
        let mut frontier = 169.0;
        let mut credited = 0.0;

        let regression = kap_frontier_progress_credit(&path, frontier, 136.0, 60.0, true);
        assert_eq!(regression, 0.0);
        let rewalk = kap_frontier_progress_credit(&path, frontier, 160.0, 60.0, false);
        assert_eq!(rewalk, 0.0);
        let new_ground = kap_frontier_progress_credit(&path, frontier, 172.0, 60.0, false);
        assert!((new_ground - 3.0).abs() < 0.001);
        credited += new_ground;
        frontier = 172.0;
        let next = kap_frontier_progress_credit(&path, frontier, 175.0, 60.0, false);
        assert!((next - 3.0).abs() < 0.001);
        credited += next;
        assert!((credited - 6.0).abs() < 0.001);
    }

    #[test]
    fn kap_position_replacement_resets_watchdog_only_for_discrete_jump() {
        let request = KapFollowRequest {
            max_cross_track: 24.0,
            ..KapFollowRequest::default()
        };

        assert_eq!(kap_position_replacement_threshold(&request), 40.0);
        assert!(!kap_position_replacement_detected(39.9, &request));
        assert!(kap_position_replacement_detected(93.3, &request));
        assert!(!kap_position_replacement_detected(f32::NAN, &request));
    }

    #[test]
    fn kap_position_replacement_threshold_tracks_wide_route_scope() {
        let request = KapFollowRequest {
            max_cross_track: 80.0,
            ..KapFollowRequest::default()
        };

        assert_eq!(kap_position_replacement_threshold(&request), 80.0);
        assert!(!kap_position_replacement_detected(79.9, &request));
        assert!(kap_position_replacement_detected(80.0, &request));
    }

    #[test]
    fn kap_downward_topology_transfer_accepts_height_matched_forward_continuation() {
        assert!(kap_downward_topology_transfer(52.34, 17.86, 0.29, 60.0));
    }

    #[test]
    fn kap_downward_topology_transfer_rejects_remote_or_nonmatching_branch() {
        assert!(!kap_downward_topology_transfer(60.01, 17.86, 0.29, 60.0));
        assert!(!kap_downward_topology_transfer(52.34, 11.99, 0.29, 60.0));
        assert!(!kap_downward_topology_transfer(52.34, 17.86, 3.01, 60.0));
        assert!(!kap_downward_topology_transfer(-2.0, 17.86, 0.29, 60.0));
    }

    #[test]
    fn kap_start_path_s_sets_initial_projection_on_local_branch() {
        let path = KapFollowPath::new(
            "parallel.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
                [200.0, 10.0, 0.0],
                [100.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let pose = Pose {
            pos: [50.0, 8.0, 0.0],
            yaw: 0.0,
            speed: 0.0,
            velocity_heading: None,
            tick_ms: None,
        };
        let request = KapFollowRequest {
            start_path_s: Some(50.0),
            start_window_m: 80.0,
            ..KapFollowRequest::default()
        };

        let projection = initial_kap_projection(&path, pose, &request).expect("projection");

        assert!(
            (projection.path_s - 50.0).abs() < 0.001,
            "initial projection should honor start_path_s: {projection:?}"
        );
    }

    #[test]
    fn kap_start_path_s_does_not_fallback_to_global_projection() {
        let path = KapFollowPath::new(
            "parallel.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [200.0, 0.0, 0.0],
                [200.0, 10.0, 0.0],
                [100.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let pose = Pose {
            pos: [50.0, 8.0, 0.0],
            yaw: 0.0,
            speed: 0.0,
            velocity_heading: None,
            tick_ms: None,
        };
        let request = KapFollowRequest {
            start_path_s: Some(190.0),
            start_window_m: 5.0,
            ..KapFollowRequest::default()
        };

        let projection = initial_kap_projection(&path, pose, &request);

        assert!(
            projection.is_none(),
            "strict start_path_s must not fall back to global projection: {projection:?}"
        );
    }

    #[test]
    fn kap_project_near_allows_closed_loop_wraparound() {
        let path = KapFollowPath::new(
            "square_1.kap".to_string(),
            "square".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );

        let projection = path
            .project_near([1.0, 0.5, 0.0], 39.0, 5.0)
            .expect("wrapped projection");
        assert!(
            projection.path_s < 2.0 || projection.path_s > 38.0,
            "closed path projection should stay in wrap window: {projection:?}"
        );
    }

    #[test]
    fn kap_project_near_forward_rejects_large_reverse_seam_jump() {
        let path = KapFollowPath::new(
            "hairpin_loop.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.0, 2.0, 0.0],
                [10.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );

        let global = path.project([10.0, 1.7, 0.0]).expect("global projection");
        assert!(
            global.path_s > 30.0,
            "global projection should prefer the physically nearer return branch: {global:?}"
        );
        let local = path
            .project_near_forward([10.0, 1.7, 0.0], 10.0, 8.0, 3.0)
            .expect("directional projection");
        assert!(
            (local.path_s - 10.0).abs() < 0.001,
            "directional projection should stay on the forward branch: {local:?}"
        );
    }

    #[test]
    fn kap_project_near_forward_allows_forward_wrap_at_closed_seam() {
        let path = KapFollowPath::new(
            "square_1.kap".to_string(),
            "square".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );

        let projection = path
            .project_near_forward([1.0, 0.5, 0.0], 39.0, 5.0, 1.0)
            .expect("wrapped forward projection");
        assert!(
            projection.path_s < 2.0,
            "directional projection should wrap forward across the seam: {projection:?}"
        );
    }

    #[test]
    fn kap_forward_delta_wraps_closed_paths_only() {
        let closed = KapFollowPath::new(
            "square_1.kap".to_string(),
            "square".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );
        assert!((closed.forward_delta_bounded(39.0, 2.0, f32::INFINITY) - 3.0).abs() < 0.001);
        assert!((closed.forward_delta_bounded(5.0, 8.0, f32::INFINITY) - 3.0).abs() < 0.001);
        assert_eq!(closed.forward_delta_bounded(39.0, 2.0, 2.0), 0.0);
        assert_eq!(closed.forward_delta_bounded(5.0, 38.0, 20.0), 0.0);

        let open = KapFollowPath::new(
            "line.kap".to_string(),
            "line".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        );
        assert!(!open.closed);
        assert_eq!(open.forward_delta_bounded(20.0, 10.0, f32::INFINITY), 0.0);
    }

    #[test]
    fn kap_seam_completion_delta_accepts_closed_finish_tolerance() {
        let closed = KapFollowPath::new(
            "square_1.kap".to_string(),
            "square".to_string(),
            Some(1),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        );
        assert!((closed.seam_completion_delta(39.0, 2.0) - 1.0).abs() < 0.001);
        assert_eq!(closed.seam_completion_delta(35.0, 2.0), 0.0);

        let open = KapFollowPath::new(
            "line.kap".to_string(),
            "line".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        );
        assert_eq!(open.seam_completion_delta(99.0, 2.0), 0.0);
    }

    #[test]
    fn kap_route_miner_prefers_straight_sampling_windows() {
        let path = KapFollowPath::new(
            "straight.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [120.0, 0.0, 0.0]],
        );
        let options = KapRouteMineOptions {
            window_m: 40.0,
            step_m: 20.0,
            min_score: 0.8,
            limit: usize::MAX,
            ..KapRouteMineOptions::default()
        };

        let windows = mine_kap_route_windows(&path, &options);

        assert!(!windows.is_empty());
        assert!(windows.iter().all(|window| window.kind == "straight"));
        assert!(windows.iter().all(|window| window.score >= 0.95));
    }

    #[test]
    fn kap_route_miner_rejects_tight_turn_windows() {
        let path = KapFollowPath::new(
            "corner.kap".to_string(),
            "test".to_string(),
            Some(1),
            vec![[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0]],
        );
        let options = KapRouteMineOptions {
            window_m: 35.0,
            step_m: 5.0,
            min_score: 0.7,
            limit: usize::MAX,
            ..KapRouteMineOptions::default()
        };

        let windows = mine_kap_route_windows(&path, &options);

        assert!(
            windows.is_empty(),
            "tight corner should not be a default sampling window: {windows:?}"
        );
    }

    #[test]
    fn self_speed_prefers_current_velocity_norm_over_initial_speed() {
        let active = BTreeMap::new();
        let pose = BTreeMap::new();
        let mut owner = BTreeMap::new();
        owner.insert("initial_speed".to_string(), "0.0015233".to_string());

        let speed = self_speed_value(
            false,
            false,
            true,
            &active,
            &pose,
            &owner,
            Some("-21.8351707"),
            Some("-7.66681147"),
            Some("-0.0269393343"),
        )
        .expect("speed from velocity norm");

        let parsed = speed.parse::<f64>().expect("numeric speed");
        assert!((parsed - 23.142673).abs() < 0.001, "{speed}");
    }

    #[test]
    fn self_speed_uses_initial_speed_only_as_last_fallback() {
        let active = BTreeMap::new();
        let pose = BTreeMap::new();
        let mut owner = BTreeMap::new();
        owner.insert("initial_speed".to_string(), "0.0015233".to_string());

        let speed = self_speed_value(false, false, true, &active, &pose, &owner, None, None, None)
            .expect("fallback initial speed");

        assert_eq!(speed, "0.0015233");
    }

    #[test]
    fn process_pid_extract_requires_requested_process_marker() {
        let launcher_only = r#"{"ok":true,"source":"powershell_live","processes":[{"name":"Launcher","pid":5392,"path":"D:\\kart\\kart\\Launcher.exe"}]}"#;
        assert_eq!(
            json_extract_pid_after(launcher_only, "\"name\":\"KartRider\"", "\"pid\":"),
            None
        );

        let with_kart = r#"{"ok":true,"source":"powershell_live","processes":[{"name":"Launcher","pid":5392},{"name":"KartRider","pid":15428,"path":"D:\\kart\\kart\\KartRider.exe"}]}"#;
        assert_eq!(
            json_extract_pid_after(with_kart, "\"name\":\"KartRider\"", "\"pid\":"),
            Some("15428".to_string())
        );
    }

    #[test]
    fn latest_state_kind_seq_tracks_only_requested_fresh_rows() {
        let path =
            std::env::temp_dir().join(format!("kart_state_kind_stop_{}.tsv", std::process::id()));
        std::fs::write(
            &path,
            "seq\ttick_ms\tsite_rva\tkind\n1\t10\t0x1\t2\n2\t20\t0x2\t22\n3\t30\t0x3\t4\n",
        )
        .unwrap();
        assert_eq!(read_latest_state_kind_seq(&path, 22).unwrap(), Some(2));
        assert_eq!(read_latest_state_kind_seq(&path, 99).unwrap(), None);

        std::fs::write(
            &path,
            "seq\ttick_ms\tsite_rva\tkind\n1\t10\t0x1\t2\n2\t20\t0x2\t22\n4\t40\t0x2\t22\npartial\trow",
        )
        .unwrap();
        assert_eq!(read_latest_state_kind_seq(&path, 22).unwrap(), Some(4));

        std::fs::write(
            &path,
            concat!(
                "seq\ttick_ms\tphysics_substep_id\tlinear_write_epoch\tsite_rva\tkind\n",
                "1\t10\t7\t19\t0x1\t2\n",
                "2\t20\t7\t20\t0x2\t22\n",
                "3\t30\t8\t21\t0x3\t4\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_state_kind_seq(&path, 22).unwrap(), Some(2));

        std::fs::write(&path, "seq\ttick_ms\tsite_rva\n1\t10\t0x1\n").unwrap();
        assert!(read_latest_state_kind_seq(&path, 22)
            .unwrap_err()
            .contains("kind"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn kap_topology_profile_detects_supported_jump_arc() {
        let path = KapFollowPath::new(
            "jump".to_string(),
            "track".to_string(),
            Some(6),
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 2.0],
                [30.0, 0.0, 3.0],
                [40.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],
            ],
        );
        let gaps = kap_gap_windows(&path.points, |point| point[0] < 15.0 || point[0] > 35.0);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].entry_s, 10.0);
        assert_eq!(gaps[0].exit_s, 40.0);
        assert!(gaps[0].required_speed >= 12.0);
        assert!(gaps[0].approach_start_s <= gaps[0].entry_s);
    }

    #[test]
    fn kap_gap_transfer_replaces_brake_with_launch_throttle() {
        assert!(!KapFollowRequest::default().gap_transfer);
        assert!(parse_bool_arg("true", "kap-gap-transfer").unwrap());
        assert!(!parse_bool_arg("false", "kap-gap-transfer").unwrap());

        let mut decision = DriveDecision {
            keys: vec![Key::Down, Key::Left],
            done: false,
            heading_error: 0.1,
            reason: "overspeed_brake".to_string(),
        };
        let pose = Pose {
            tick_ms: None,
            pos: [0.0, 0.0, 0.0],
            yaw: 0.0,
            speed: 7.0,
            velocity_heading: Some(0.0),
        };
        let gap = KapGapWindow {
            approach_start_s: 0.0,
            entry_s: 20.0,
            exit_s: 40.0,
            required_speed: 14.0,
        };
        apply_kap_gap_transfer(
            &mut decision,
            pose,
            KapProjection {
                path_s: 10.0,
                cross_track: 0.5,
            },
            &gap,
            &KapFollowRequest::default(),
        );
        assert_eq!(decision.keys, vec![Key::Up, Key::Left]);
        assert!(decision.reason.contains("kap_gap_transfer:approach"));

        let mut recoverable = DriveDecision {
            keys: vec![Key::Down, Key::Right],
            done: false,
            heading_error: -0.2,
            reason: "panic".to_string(),
        };
        apply_kap_gap_transfer(
            &mut recoverable,
            pose,
            KapProjection {
                path_s: 20.0,
                cross_track: 8.6,
            },
            &gap,
            &KapFollowRequest::default(),
        );
        assert_eq!(recoverable.keys, vec![Key::Up, Key::Right]);

        let mut unrecoverable = DriveDecision {
            keys: vec![Key::Down, Key::Right],
            done: false,
            heading_error: -0.2,
            reason: "panic".to_string(),
        };
        apply_kap_gap_transfer(
            &mut unrecoverable,
            pose,
            KapProjection {
                path_s: 20.0,
                cross_track: 13.0,
            },
            &gap,
            &KapFollowRequest::default(),
        );
        assert_eq!(unrecoverable.keys, vec![Key::Down, Key::Right]);
    }
}
#[test]
fn world_collision_filter_requires_recursive_road_tag() {
    let decorative = kart_fmt::kart_object::BinaryXmlTag {
        name: "property".to_string(),
        ..Default::default()
    };
    assert!(!has_tag(Some(&decorative), "road"));
    let physical = kart_fmt::kart_object::BinaryXmlTag {
        name: "property".to_string(),
        children: vec![kart_fmt::kart_object::BinaryXmlTag {
            name: "road".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(has_tag(Some(&physical), "road"));
}
