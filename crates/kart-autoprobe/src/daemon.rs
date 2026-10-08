use super::*;

fn focus_kart_rider_retry(timeout_ms: u64) -> bool {
    let deadline = tick_millis().saturating_add(timeout_ms as u128);
    loop {
        if focus_kart_rider() {
            return true;
        }
        if tick_millis() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[derive(Clone, Debug)]
struct DaemonConfig {
    bind: String,
    dynout: PathBuf,
    probe_root: PathBuf,
    package_root: PathBuf,
}

static EXPERIMENT_ACTIVE: AtomicBool = AtomicBool::new(false);
static EXPERIMENT_CANCEL_GENERATION: AtomicU64 = AtomicU64::new(0);

// Process presence changes on the order of seconds, but the live probe spawns
// a full PowerShell run (~200-500ms). Poll loops (250ms) and status builders
// share this short-lived cache; diagnostic endpoints and capture start/stop
// verification keep calling the live path directly.
static PROCESS_STATUS_CACHE: Mutex<Option<(u128, String)>> = Mutex::new(None);
const PROCESS_STATUS_CACHE_TTL_MS: u128 = 2_000;

fn daemon_process_status_json_cached(config: &DaemonConfig) -> String {
    let now = tick_millis();
    if let Ok(cache) = PROCESS_STATUS_CACHE.lock() {
        if let Some((tick, body)) = cache.as_ref() {
            if now.saturating_sub(*tick) <= PROCESS_STATUS_CACHE_TTL_MS {
                return body.clone();
            }
        }
    }
    let body = daemon_process_status_json(config);
    if let Ok(mut cache) = PROCESS_STATUS_CACHE.lock() {
        *cache = Some((now, body.clone()));
    }
    body
}

struct ExperimentRunGuard {
    cancel_generation: u64,
}

impl Drop for ExperimentRunGuard {
    fn drop(&mut self) {
        EXPERIMENT_ACTIVE.store(false, Ordering::SeqCst);
    }
}

fn try_begin_experiment() -> Result<ExperimentRunGuard, String> {
    EXPERIMENT_ACTIVE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .map_err(|_| "experiment already active".to_string())?;
    Ok(ExperimentRunGuard {
        cancel_generation: EXPERIMENT_CANCEL_GENERATION.load(Ordering::SeqCst),
    })
}

fn experiment_cancelled(guard: &ExperimentRunGuard) -> bool {
    EXPERIMENT_CANCEL_GENERATION.load(Ordering::SeqCst) != guard.cancel_generation
}

fn check_experiment_cancelled(
    config: &DaemonConfig,
    scenario: &str,
    expected_map: &str,
    params: &BTreeMap<String, String>,
    steps: &[String],
    guard: &ExperimentRunGuard,
) -> Result<(), String> {
    if experiment_cancelled(guard) {
        let _ =
            write_experiment_manifest(config, scenario, expected_map, params, steps, "cancelled");
        return Err("experiment_cancelled".to_string());
    }
    Ok(())
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:37731".to_string(),
            dynout: PathBuf::from(r"D:\kr\dynout"),
            probe_root: PathBuf::from(r"D:\kr\probe\KRPhysicsProbe"),
            package_root: PathBuf::from(r"D:\kr\packages"),
        }
    }
}

pub(crate) fn run_daemon(args: &[String]) -> Result<(), String> {
    let mut config = DaemonConfig::default();
    let mut cursor = FlagCursor::new(args, 0);
    while let Some(flag) = cursor.flag() {
        match flag {
            "--bind" => config.bind = cursor.take_value("--bind requires an address")?.to_string(),
            "--dynout" => {
                config.dynout = PathBuf::from(cursor.take_value("--dynout requires a path")?)
            }
            "--probe-root" => {
                config.probe_root =
                    PathBuf::from(cursor.take_value("--probe-root requires a path")?)
            }
            "--package-root" => {
                config.package_root =
                    PathBuf::from(cursor.take_value("--package-root requires a path")?)
            }
            other => return Err(format!("unknown daemon option: {other}")),
        }
        cursor.advance();
    }

    let listener =
        TcpListener::bind(&config.bind).map_err(|err| format!("bind {}: {err}", config.bind))?;
    eprintln!(
        "kart-autoprobe daemon listening on {} dynout={} probe_root={} package_root={}",
        config.bind,
        config.dynout.display(),
        config.probe_root.display(),
        config.package_root.display()
    );
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                thread::spawn(move || handle_daemon_connection(stream, config));
            }
            Err(err) => eprintln!("daemon accept failed: {err}"),
        }
    }
    Ok(())
}

fn handle_daemon_connection(mut stream: TcpStream, config: DaemonConfig) {
    match panic::catch_unwind(AssertUnwindSafe(|| {
        handle_daemon_client(&mut stream, &config)
    })) {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            append_daemon_event(&config, "http_error", &err);
            let _ = write_http_json(
                &mut stream,
                500,
                &format!("{{\"ok\":false,\"error\":\"{}\"}}", json_escape(&err)),
            );
        }
        Err(payload) => {
            let detail = panic_payload_to_string(payload.as_ref());
            append_daemon_event(&config, "http_panic", &detail);
            let _ = write_http_json(
                &mut stream,
                500,
                &format!(
                    "{{\"ok\":false,\"error\":\"panic:{}\"}}",
                    json_escape(&detail)
                ),
            );
        }
    }
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(value) = payload.downcast_ref::<&str>() {
        (*value).to_string()
    } else if let Some(value) = payload.downcast_ref::<String>() {
        value.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn handle_daemon_client(stream: &mut TcpStream, config: &DaemonConfig) -> Result<(), String> {
    let request = read_http_request(stream)?;
    if request.is_empty() {
        return Ok(());
    }
    let mut lines = request.lines();
    let first = lines.next().ok_or("empty request")?;
    let mut parts = first.split_whitespace();
    let method = parts.next().ok_or("missing method")?;
    let target = parts.next().ok_or("missing target")?;
    let (path, query) = split_query(target);

    match (method, path) {
        ("GET", "/health") => write_http_json(stream, 200, &daemon_health_json(config)),
        ("GET", "/daemon/update-status") => {
            write_http_json(stream, 200, &daemon_update_status_json(config))
        }
        ("GET", "/process/status") => {
            write_http_json(stream, 200, &daemon_process_status_json(config))
        }
        ("GET", "/process/module-bytes") => {
            let params = parse_query(query);
            write_http_json(stream, 200, &daemon_process_module_bytes_json(&params))
        }
        ("GET", "/process/windows") => write_http_json(stream, 200, &daemon_process_windows_json()),
        ("POST", "/map/confirm") => {
            let params = parse_query(query);
            write_result_json(stream, confirm_live_map(config, &params), "map_confirm")
        }
        ("POST", "/launcher/start") => {
            let params = parse_query(query);
            write_result_json(stream, launcher_start(config, &params), "launcher_start")
        }
        ("POST", "/ui/clear-blockers") => {
            write_result_json(stream, ui_clear_blockers(config), "ui_clear_blockers")
        }
        ("POST", "/ui/return-room") => {
            let params = parse_query(query);
            write_result_json(stream, ui_return_room(config, &params), "ui_return_room")
        }
        ("POST", "/ui/select-practice-track") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                ui_select_practice_track(config, &params),
                "ui_select_practice_track",
            )
        }
        ("POST", "/desktop/esc-restart-probe") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                desktop_esc_restart_probe(config, &params),
                "desktop_esc_restart_probe",
            )
        }
        ("GET", "/diagnostics/crash-events") => {
            let params = parse_query(query);
            write_http_json(stream, 200, &daemon_crash_events_json(&params))
        }
        ("GET", "/diagnostics/d3d9") => write_http_json(stream, 200, &daemon_d3d9_json()),
        ("GET", "/diagnostics/input-security") => {
            write_http_json(stream, 200, &daemon_input_security_json(config))
        }
        ("POST", "/diagnostics/display-mode") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                daemon_set_display_mode_json(&params),
                "diagnostics_display_mode",
            )
        }
        ("GET", "/status") | ("GET", "/state") | ("GET", "/capture/status") => {
            write_http_json(stream, 200, &daemon_state_json(config))
        }
        ("GET", "/state/live") => write_http_json(stream, 200, &daemon_live_state_json(config)),
        ("GET", "/state/self") => write_http_json(stream, 200, &daemon_self_state_json(config)),
        ("GET", "/route/kap/nearest") | ("GET", "/route/kap") => {
            let params = parse_query(query);
            append_daemon_event(config, "kap_route_request", &params_object_json(&params));
            write_result_json(
                stream,
                daemon_kap_route_nearest_json(config, &params),
                "kap_route_nearest",
            )
        }
        ("GET", "/experiment/preflight") => {
            write_http_json(stream, 200, &daemon_preflight_json(config))
        }
        ("GET", "/experiment/status") => write_result_json(
            stream,
            Ok(daemon_experiment_json(config)),
            "experiment_status",
        ),
        ("POST", "/experiment/cancel") => {
            let params = parse_query(query);
            let wait_ms = params
                .get("wait_ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(1000);
            write_result_json(
                stream,
                experiment_cancel(config, wait_ms),
                "experiment_cancel",
            )
        }
        ("GET", "/events") => {
            let params = parse_query(query);
            let since_seq = params
                .get("since_seq")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let limit = params
                .get("limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(200);
            write_http_json(stream, 200, &daemon_events_json(config, since_seq, limit))
        }
        ("GET", "/logs/probe") => {
            let params = parse_query(query);
            let tail = params
                .get("tail")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(200);
            write_http_json(
                stream,
                200,
                &daemon_log_json(config, "probe_status.log", tail),
            )
        }
        ("GET", "/logs/event") => {
            let params = parse_query(query);
            let tail = params
                .get("tail")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(200);
            write_http_json(
                stream,
                200,
                &daemon_log_json(config, "probe_event.tsv", tail),
            )
        }
        ("GET", "/state/rail-coverage") => {
            let params = parse_query(query);
            let since_seq = params
                .get("since_seq")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            write_http_json(stream, 200, &daemon_rail_coverage_json(config, since_seq))
        }
        ("GET", "/packages/latest") | ("GET", "/capture/package/latest") => {
            write_result_json(stream, latest_package_json(config), "packages_latest")
        }
        ("GET", "/packages/by-run") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                package_by_run_json(config, &params),
                "packages_by_run",
            )
        }
        ("GET", "/packages/download") | ("GET", "/capture/package/download") => {
            let params = parse_query(query);
            let path = resolve_package_download_path(config, &params)?;
            write_http_file(stream, &path, "application/x-7z-compressed")
        }
        ("POST", "/capture/start") => {
            let params = parse_query(query);
            write_result_json(stream, capture_start(config, &params), "capture_start")
        }
        ("POST", "/capture/stop") => {
            let params = parse_query(query);
            let wait_ms = params
                .get("wait_ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(30_000);
            write_result_json(stream, capture_stop(config, wait_ms), "capture_stop")
        }
        ("POST", "/capture/restart") => {
            let params = parse_query(query);
            // Restart must be a serialized stop/ack/start transaction.  The
            // probe restores many live patch sites and one second is not a
            // safe fixed delay.  Starting another injection before the old
            // probe acknowledges quiescence can terminate the game process.
            capture_stop(config, 30_000)?;
            write_result_json(stream, capture_start(config, &params), "capture_restart")
        }
        ("POST", "/capture/package") => {
            let params = parse_query(query);
            write_result_json(stream, capture_package(config, &params), "capture_package")
        }
        ("POST", "/experiment/loop") => {
            let params = parse_query(query);
            write_result_json(stream, experiment_loop(config, &params), "experiment_loop")
        }
        ("POST", "/experiment/loop/start") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                experiment_start_detached(config, params, false),
                "experiment_loop_start",
            )
        }
        ("POST", "/experiment/room-loop") => {
            let mut params = parse_query(query);
            params.insert("wait_return_ready".to_string(), "1".to_string());
            write_result_json(
                stream,
                experiment_loop(config, &params),
                "experiment_room_loop",
            )
        }
        ("POST", "/experiment/room-loop/start") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                experiment_start_detached(config, params, true),
                "experiment_room_loop_start",
            )
        }
        ("POST", "/daemon/restart-into-next") => write_result_json(
            stream,
            daemon_restart_into_next(config),
            "daemon_restart_into_next",
        ),
        ("POST", "/hold") | ("POST", "/control/hold") => {
            let params = parse_query(query);
            let keys = params.get("keys").map(String::as_str).unwrap_or("F5");
            let ms = params
                .get("ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(90);
            let input_backend = parse_input_backend_param(&params)?;
            let keys = parse_keys(keys)?;
            let before = current_live_state(config);
            let before_self = current_self_state(config);
            if !before.control_allowed && !params_bool(&params, "force") {
                let result = control_denied_reason(&before);
                append_control_evidence(
                    config,
                    "hold",
                    &keys,
                    ms,
                    false,
                    &before,
                    &before,
                    &before_self,
                    &before_self,
                    &result,
                );
                return write_http_json(
                    stream,
                    409,
                    &format!(
                        "{{\"ok\":false,\"error\":\"{}\",\"before\":{}}}",
                        json_escape(&result),
                        before.to_json()
                    ),
                );
            }
            let focus = if input_backend.requires_foreground() {
                focus_kart_rider_retry(2_000)
            } else {
                false
            };
            let result = execute_with_backend(
                input_backend,
                &[Command::Hold {
                    keys: keys.clone(),
                    duration_ms: ms,
                }],
                &mut EventLog::from_optional_path(None)?,
            );
            let after = current_live_state(config);
            let after_self = current_self_state(config);
            match result {
                Ok(()) => {
                    append_control_evidence(
                        config,
                        "hold",
                        &keys,
                        ms,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        "ok",
                    );
                    write_http_json(
                        stream,
                        200,
                        &format!(
                            "{{\"ok\":true,\"focus\":{},\"backend\":\"{}\",\"before\":{},\"after\":{}}}",
                            bool_json(focus),
                            input_backend.name(),
                            before.to_json(),
                            after.to_json()
                        ),
                    )
                }
                Err(err) => {
                    append_control_evidence(
                        config,
                        "hold",
                        &keys,
                        ms,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        &err,
                    );
                    Err(err)
                }
            }
        }
        ("POST", "/control/trace-hold") => {
            let params = parse_query(query);
            write_result_json(
                stream,
                control_trace_hold(config, &params),
                "control_trace_hold",
            )
        }
        ("POST", "/start-if-needed") => {
            let params = parse_query(query);
            let ms = params
                .get("ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(90);
            let status = DaemonStatus::from_dynout(&config.dynout);
            let mut sent = false;
            let mut focus = false;
            if !status.drivable {
                focus = focus_kart_rider_retry(2_000);
                execute(
                    &[Command::Hold {
                        keys: vec![Key::F5],
                        duration_ms: ms,
                    }],
                    &mut EventLog::from_optional_path(None)?,
                )?;
                sent = true;
            }
            write_http_json(
                stream,
                200,
                &format!(
                    "{{\"ok\":true,\"sent\":{},\"focus\":{},\"status\":{}}}",
                    bool_json(sent),
                    bool_json(focus),
                    status.to_json()
                ),
            )
        }
        ("POST", "/control/start-if-needed") => {
            let params = parse_query(query);
            let ms = params
                .get("ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(90);
            let wait_ms = params
                .get("wait_ms")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(12000);
            write_result_json(
                stream,
                control_start_if_needed(config, ms, wait_ms),
                "control_start_if_needed",
            )
        }
        ("POST", "/release-all") => {
            write_result_json(stream, control_release_all(config), "control_release_all")
        }
        ("POST", "/control/release-all") => {
            write_result_json(stream, control_release_all(config), "control_release_all")
        }
        ("POST", "/script") | ("POST", "/control/script") => {
            let params = parse_query(query);
            let mut script_args = daemon_script_args(&params)?;
            let before = current_live_state(config);
            let before_self = current_self_state(config);
            if !before.control_allowed && !params_bool(&params, "force") {
                let result = control_denied_reason(&before);
                append_control_evidence(
                    config,
                    "script",
                    &[],
                    0,
                    false,
                    &before,
                    &before,
                    &before_self,
                    &before_self,
                    &result,
                );
                script_args.clear();
                return write_http_json(
                    stream,
                    409,
                    &format!(
                        "{{\"ok\":false,\"error\":\"{}\",\"before\":{}}}",
                        json_escape(&result),
                        before.to_json()
                    ),
                );
            }
            let focus = focus_kart_rider_retry(2_000);
            let result = run_script(&script_args);
            let _ = execute(
                &[Command::ReleaseAll],
                &mut EventLog::from_optional_path(None)?,
            );
            let after = current_live_state(config);
            let after_self = current_self_state(config);
            match result {
                Ok(()) => {
                    append_control_evidence(
                        config,
                        "script",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        "ok",
                    );
                    write_http_json(
                        stream,
                        200,
                        &format!(
                            "{{\"ok\":true,\"focus\":{},\"before\":{},\"after\":{}}}",
                            bool_json(focus),
                            before.to_json(),
                            after.to_json()
                        ),
                    )
                }
                Err(err) => {
                    append_control_evidence(
                        config,
                        "script",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        &err,
                    );
                    script_args.clear();
                    write_http_json(
                        stream,
                        500,
                        &format!(
                            "{{\"ok\":false,\"focus\":{},\"error\":\"{}\"}}",
                            bool_json(focus),
                            json_escape(&err)
                        ),
                    )
                }
            }
        }
        ("POST", "/drive-to") | ("POST", "/control/drive-to") => {
            let params = parse_query(query);
            let drive_args = daemon_drive_to_args(config, &params)?;
            let before = current_live_state(config);
            let before_self = current_self_state(config);
            if !before.control_allowed && !params_bool(&params, "force") {
                let result = control_denied_reason(&before);
                append_control_evidence(
                    config,
                    "drive_to",
                    &[],
                    0,
                    false,
                    &before,
                    &before,
                    &before_self,
                    &before_self,
                    &result,
                );
                return write_http_json(
                    stream,
                    409,
                    &format!(
                        "{{\"ok\":false,\"error\":\"{}\",\"before\":{}}}",
                        json_escape(&result),
                        before.to_json()
                    ),
                );
            }
            let focus = focus_kart_rider_retry(2_000);
            let pose_path = params
                .get("pose")
                .map(PathBuf::from)
                .unwrap_or_else(|| config.dynout.join("probe_pose.tsv"));
            if let Err(err) = seed_pose_tsv_from_self_if_needed(config, &pose_path, &before_self) {
                let after = current_live_state(config);
                let after_self = current_self_state(config);
                append_control_evidence(
                    config,
                    "drive_to",
                    &[],
                    0,
                    focus,
                    &before,
                    &after,
                    &before_self,
                    &after_self,
                    &err,
                );
                return write_http_json(
                    stream,
                    500,
                    &format!(
                        "{{\"ok\":false,\"focus\":{},\"error\":\"{}\"}}",
                        bool_json(focus),
                        json_escape(&err)
                    ),
                );
            }
            let result = run_drive_to(&drive_args);
            let _ = execute(
                &[Command::ReleaseAll],
                &mut EventLog::from_optional_path(None)?,
            );
            let after = current_live_state(config);
            let after_self = current_self_state(config);
            match result {
                Ok(()) => {
                    append_control_evidence(
                        config,
                        "drive_to",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        "ok",
                    );
                    write_http_json(
                        stream,
                        200,
                        &format!(
                            "{{\"ok\":true,\"focus\":{},\"before\":{},\"after\":{}}}",
                            bool_json(focus),
                            before.to_json(),
                            after.to_json()
                        ),
                    )
                }
                Err(err) => {
                    append_control_evidence(
                        config,
                        "drive_to",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        &err,
                    );
                    write_http_json(
                        stream,
                        500,
                        &format!(
                            "{{\"ok\":false,\"focus\":{},\"error\":\"{}\"}}",
                            bool_json(focus),
                            json_escape(&err)
                        ),
                    )
                }
            }
        }
        ("POST", "/drive-surface") | ("POST", "/control/drive-surface") => {
            let params = parse_query(query);
            append_daemon_event(
                config,
                "drive_surface_request",
                &format!(
                    "track={} pose={} kap_dir={} kap_track={} max_ms={} target_progress={} input_backend={}",
                    params.get("track").map(String::as_str).unwrap_or(""),
                    params.get("pose").map(String::as_str).unwrap_or(""),
                    params.get("kap_dir").map(String::as_str).unwrap_or(""),
                    params.get("kap_track").map(String::as_str).unwrap_or(""),
                    params.get("max_ms").map(String::as_str).unwrap_or(""),
                    params
                        .get("target_progress")
                        .map(String::as_str)
                        .unwrap_or(""),
                    params
                        .get("input_backend")
                        .map(String::as_str)
                        .unwrap_or_else(|| {
                            params
                                .get("drive_input_backend")
                                .map(String::as_str)
                                .unwrap_or("")
                        })
                ),
            );
            let mut drive_args = daemon_drive_surface_args(config, &params)?;
            append_daemon_event(
                config,
                "drive_surface_args",
                &format!("argc={} args={}", drive_args.len(), drive_args.join(" ")),
            );
            let before = current_live_state(config);
            let before_self = current_self_state(config);
            if !before.control_allowed && !params_bool(&params, "force") {
                let result = control_denied_reason(&before);
                append_control_evidence(
                    config,
                    "drive_surface",
                    &[],
                    0,
                    false,
                    &before,
                    &before,
                    &before_self,
                    &before_self,
                    &result,
                );
                drive_args.clear();
                return write_http_json(
                    stream,
                    409,
                    &format!(
                        "{{\"ok\":false,\"error\":\"{}\",\"before\":{}}}",
                        json_escape(&result),
                        before.to_json()
                    ),
                );
            }
            if let Err(result) = ensure_control_input_ready(config) {
                append_control_evidence(
                    config,
                    "drive_surface",
                    &[],
                    0,
                    false,
                    &before,
                    &before,
                    &before_self,
                    &before_self,
                    &result,
                );
                drive_args.clear();
                return write_http_json(
                    stream,
                    409,
                    &format!(
                        "{{\"ok\":false,\"error\":\"{}\",\"before\":{}}}",
                        json_escape(&result),
                        before.to_json()
                    ),
                );
            }
            let focus = focus_kart_rider_retry(2_000);
            if params_bool(&params, "seed_pose_from_self")
                || !params.contains_key("seed_pose_from_self")
            {
                let pose_path = PathBuf::from(
                    params
                        .get("pose")
                        .ok_or("drive-surface requires pose for seed_pose_from_self")?,
                );
                if let Err(err) =
                    seed_pose_tsv_from_self_if_needed(config, &pose_path, &before_self)
                {
                    append_daemon_event(config, "drive_surface_pose_seed_failed", &err);
                }
            }
            let result = run_drive_surface(&drive_args);
            let _ = execute(
                &[Command::ReleaseAll],
                &mut EventLog::from_optional_path(None)?,
            );
            let after = current_live_state(config);
            let after_self = current_self_state(config);
            // `--timeout-ok` only relaxes route progress/timeout acceptance.
            // It must never turn a vanished KartRider process into a
            // successful control run while the CLI keeps consuming the last
            // pose row.  Preserve this as the primary drive failure so a
            // later capture-stop acknowledgement timeout cannot hide it.
            let result = match result {
                Ok(()) if after.phase == "no_process" => Err(format!(
                    "control_authority_lost_during_drive:phase={} reason={}",
                    after.phase, after.control_reason
                )),
                other => other,
            };
            match result {
                Ok(()) => {
                    append_control_evidence(
                        config,
                        "drive_surface",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        "ok",
                    );
                    write_http_json(
                        stream,
                        200,
                        &format!(
                            "{{\"ok\":true,\"focus\":{},\"before\":{},\"after\":{}}}",
                            bool_json(focus),
                            before.to_json(),
                            after.to_json()
                        ),
                    )
                }
                Err(err) => {
                    append_control_evidence(
                        config,
                        "drive_surface",
                        &[],
                        0,
                        focus,
                        &before,
                        &after,
                        &before_self,
                        &after_self,
                        &err,
                    );
                    drive_args.clear();
                    write_http_json(
                        stream,
                        500,
                        &format!(
                            "{{\"ok\":false,\"focus\":{},\"error\":\"{}\"}}",
                            bool_json(focus),
                            json_escape(&err)
                        ),
                    )
                }
            }
        }
        _ => write_http_json(
            stream,
            404,
            &format!(
                "{{\"ok\":false,\"error\":\"unknown endpoint {} {}\"}}",
                json_escape(method),
                json_escape(path)
            ),
        ),
    }
}

fn read_http_request(stream: &mut TcpStream) -> Result<String, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|err| format!("set read timeout: {err}"))?;
    let mut request = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                request.extend_from_slice(&chunk[..n]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
                if request.len() > 65536 {
                    return Err("request header too large".to_string());
                }
            }
            Err(err)
                if err.kind() == io::ErrorKind::WouldBlock
                    || err.kind() == io::ErrorKind::TimedOut =>
            {
                if request.is_empty() {
                    return Err("request read timeout".to_string());
                }
                return Err("incomplete HTTP request header".to_string());
            }
            Err(err) => return Err(format!("read request: {err}")),
        }
    }
    if request.is_empty() {
        return Ok(String::new());
    }
    if !request.windows(4).any(|window| window == b"\r\n\r\n") {
        return Err("incomplete HTTP request header".to_string());
    }
    Ok(String::from_utf8_lossy(&request).into_owned())
}

fn daemon_script_args(params: &BTreeMap<String, String>) -> Result<Vec<String>, String> {
    let mut args = vec![params
        .get("script")
        .ok_or("script endpoint requires script")?
        .clone()];
    if let Some(log) = params.get("log") {
        args.push("--log".to_string());
        args.push(log.clone());
    }
    Ok(args)
}

fn daemon_drive_to_args(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let pose = params
        .get("pose")
        .cloned()
        .unwrap_or_else(|| default_live_pose_path(config).display().to_string());
    let x = params.get("x").ok_or("drive-to requires x")?.clone();
    let y = params.get("y").ok_or("drive-to requires y")?.clone();
    let z = params.get("z").cloned().unwrap_or_else(|| "0".to_string());
    let mut args = vec![pose, x, y, z];
    let default_log = config
        .dynout
        .join("probe_control.tsv")
        .display()
        .to_string();

    for (param, flag) in [
        ("model", "--model"),
        ("log", "--log"),
        ("collision", "--collision"),
        ("wall_hard", "--wall-hard"),
        ("cancel_file", "--cancel-file"),
        ("max_ms", "--max-ms"),
        ("step_ms", "--step-ms"),
        ("stop_distance", "--stop-distance"),
        ("heading_mode", "--heading-mode"),
        ("stop_on_state_kind", "--stop-on-state-kind"),
        ("stuck_timeout_ms", "--stuck-timeout-ms"),
        ("unstuck_reverse_ms", "--unstuck-reverse-ms"),
        ("max_unstuck_attempts", "--max-unstuck-attempts"),
        ("heading_calibration", "--heading-calibration"),
        ("terminal_brake_ms", "--terminal-brake-ms"),
        ("terminal_brake_step_ms", "--terminal-brake-step-ms"),
        ("terminal_stop_speed", "--terminal-stop-speed"),
        ("terminal_settle_ms", "--terminal-settle-ms"),
    ] {
        if let Some(value) = params.get(param) {
            args.push(flag.to_string());
            args.push(value.clone());
        }
    }
    if !params.contains_key("log") {
        args.push("--log".to_string());
        args.push(default_log);
    }
    if params_bool(params, "timeout_ok") {
        args.push("--timeout-ok".to_string());
    }
    Ok(args)
}

fn daemon_drive_surface_args(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let mut args = vec![
        params
            .get("track")
            .ok_or("drive-surface requires track")?
            .clone(),
        params
            .get("pose")
            .cloned()
            .unwrap_or_else(|| default_live_pose_path(config).display().to_string()),
    ];

    let has_target = params.contains_key("surface")
        || params.contains_key("rail")
        || params.contains_key("kap_dir");
    if !has_target {
        return Err("drive-surface requires surface, rail, or kap_dir".to_string());
    }
    if let Some(surface) = params.get("surface") {
        args.push("--surface".to_string());
        args.push(surface.clone());
    }
    if let Some(rail) = params.get("rail") {
        args.push("--rail".to_string());
        args.push(rail.clone());
    }
    if let Some(kap_dir) = params.get("kap_dir") {
        args.push("--kap-dir".to_string());
        args.push(kap_dir.clone());
    }

    // Keep every daemon-driven KAP attempt diagnosable.  Unlike drive-to,
    // drive-surface previously omitted --log unless the HTTP caller supplied
    // one, so terminal reasons and cursor/projection decisions disappeared
    // from otherwise valid captures.  A per-capture file is safe because the
    // capture start path clears dynout before the experiment begins.
    let default_log = config
        .dynout
        .join("probe_kap_control.tsv")
        .display()
        .to_string();

    for (param, flag) in [
        ("kap_track", "--kap-track"),
        ("kap_slot", "--kap-slot"),
        ("lookahead", "--lookahead"),
        ("max_cross_track", "--max-cross-track"),
        ("target_progress", "--target-progress"),
        ("start_path_s", "--start-path-s"),
        ("start_window_m", "--start-window-m"),
        ("lateral_offset", "--lateral-offset"),
        ("offset_start_s", "--offset-start-s"),
        ("offset_end_s", "--offset-end-s"),
        ("offset_ramp_m", "--offset-ramp-m"),
        ("lateral_offset_zones", "--lateral-offset-zones"),
        ("heading_mode", "--heading-mode"),
        ("speed_limit", "--speed-limit"),
        ("speed_zones", "--speed-zones"),
        ("kap_gap_transfer", "--kap-gap-transfer"),
        ("abort_above_speed", "--abort-above-speed"),
        ("abort_below_speed", "--abort-below-speed"),
        ("abort_below_speed_after_ms", "--abort-below-speed-after-ms"),
        ("longitudinal_cooldown_ms", "--longitudinal-cooldown-ms"),
        ("fixed_keys", "--fixed-keys"),
        ("fixed_add_keys", "--fixed-add-keys"),
        ("fixed_key_sequence", "--fixed-key-sequence"),
        ("fixed_add_key_sequence", "--fixed-add-key-sequence"),
        ("safe_cross_track", "--safe-cross-track"),
        ("panic_cross_track", "--panic-cross-track"),
        ("stuck_timeout_ms", "--stuck-timeout-ms"),
        ("cross_track_timeout_ms", "--cross-track-timeout-ms"),
        ("model", "--model"),
        ("physics_baseline", "--physics-baseline"),
        ("control_mode", "--control-mode"),
        ("log", "--log"),
        ("collision", "--collision"),
        ("wall_hard", "--wall-hard"),
        ("cancel_file", "--cancel-file"),
        ("max_ms", "--max-ms"),
        ("step_ms", "--step-ms"),
        ("stop_distance", "--stop-distance"),
        ("stop_on_surface_id", "--stop-on-surface-id"),
        ("stop_on_state_kind", "--stop-on-state-kind"),
        ("terminal_brake_ms", "--terminal-brake-ms"),
        ("terminal_brake_step_ms", "--terminal-brake-step-ms"),
        ("terminal_stop_speed", "--terminal-stop-speed"),
        ("terminal_settle_ms", "--terminal-settle-ms"),
        ("drive_input_backend", "--input-backend"),
        ("input_backend", "--input-backend"),
    ] {
        if let Some(value) = params.get(param) {
            if value.is_empty() {
                continue;
            }
            args.push(flag.to_string());
            args.push(value.clone());
        }
    }
    if !params.contains_key("log") {
        args.push("--log".to_string());
        args.push(default_log);
    }
    if params_bool(params, "stop_on_surface") {
        args.push("--stop-on-surface".to_string());
    }
    if params_bool(params, "timeout_ok") {
        args.push("--timeout-ok".to_string());
    }
    if params_bool(params, "ignore_reachability") {
        args.push("--ignore-reachability".to_string());
    }
    if !params.contains_key("wall_hard") {
        args.push("--wall-hard".to_string());
        args.push(
            config
                .dynout
                .join("probe_wall_hard.tsv")
                .display()
                .to_string(),
        );
    }
    Ok(args)
}

// Both files use a schema accepted by kart-autoprobe.  Some hooks are not hit
// continuously, so probe_pose.tsv can remain syntactically valid while being
// seconds behind the owner snapshot.  Default to whichever live source was
// updated most recently; an explicit request parameter still wins above.
fn default_live_pose_path(config: &DaemonConfig) -> PathBuf {
    let pose = config.dynout.join("probe_pose.tsv");
    let owner = config.dynout.join("probe_owner.tsv");
    let modified = |path: &Path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
    };
    match (modified(&pose), modified(&owner)) {
        (None, Some(_)) => owner,
        (Some(pose_time), Some(owner_time)) if owner_time > pose_time => owner,
        _ => pose,
    }
}

fn seed_pose_tsv_from_self_if_needed(
    config: &DaemonConfig,
    pose_path: &Path,
    self_state: &SelfStateView,
) -> Result<(), String> {
    // A data row is not proof that the pose belongs to the current race.  The
    // probe commonly stays injected across return-to-room/restart, while a
    // stationary kart may not produce a new pose-hook row.  Always append the
    // fresh daemon self projection immediately before launching the controller
    // so path selection cannot consume the previous race's terminal pose.
    let pose = self_pose_for_planning(self_state)?;
    let vel_x = self_state
        .vel_x
        .as_deref()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let vel_y = self_state
        .vel_y
        .as_deref()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let vel_z = self_state
        .vel_z
        .as_deref()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let yaw = pose.velocity_heading.unwrap_or(pose.yaw);
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    let this = if self_state.identity_this.is_empty() {
        "0x00000000"
    } else {
        self_state.identity_this.as_str()
    };
    if !pose_path.exists() {
        fs::write(pose_path, pose_tsv_header())
            .map_err(|err| format!("write pose seed header {}: {err}", pose_path.display()))?;
    }
    let row = format!(
        concat!(
            "0\t{}\t0\t0\t0x00000000\t900\t{}\t0",
            "\t{:.9}\t{:.9}\t{:.9}",
            "\t{:.9}\t{:.9}\t{:.9}",
            "\t0\t0\t0",
            "\t{:.9}\t{:.9}\t{:.9}",
            "\t{:.9}\t{:.9}\t0\t{:.9}\t{:.9}\t0\t0\t0\t1",
            "\t{:.9}\t{:.9}\t0\t{:.9}\t{:.9}\t0\t0\t0\t1",
            "\t0\t0\t{:.9}",
            "\t0\t0\t{:.9}",
            "\t0\t0\n"
        ),
        tick_millis(),
        this,
        pose.pos[0],
        pose.pos[1],
        pose.pos[2],
        vel_x,
        vel_y,
        vel_z,
        pose.pos[0],
        pose.pos[1],
        pose.pos[2],
        cos_yaw,
        sin_yaw,
        -sin_yaw,
        cos_yaw,
        cos_yaw,
        sin_yaw,
        -sin_yaw,
        cos_yaw,
        yaw,
        yaw
    );
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(pose_path)
        .map_err(|err| format!("open pose seed {}: {err}", pose_path.display()))?;
    file.write_all(row.as_bytes())
        .map_err(|err| format!("append pose seed {}: {err}", pose_path.display()))?;
    append_daemon_event(
        config,
        "drive_surface_pose_seeded",
        &format!(
            "pose={} x={:.3} y={:.3} z={:.3} yaw={:.3}",
            pose_path.display(),
            pose.pos[0],
            pose.pos[1],
            pose.pos[2],
            yaw
        ),
    );
    Ok(())
}

fn pose_tsv_header() -> &'static str {
    concat!(
        "seq\ttick_ms\tphysics_substep_id\tlinear_write_epoch\tsite_rva\tkind\tthis\tdt",
        "\tpre_pos_x\tpre_pos_y\tpre_pos_z",
        "\tvel_x\tvel_y\tvel_z",
        "\tsrc_x\tsrc_y\tsrc_z",
        "\tpost_pos_x\tpost_pos_y\tpost_pos_z",
        "\tpre_m00\tpre_m01\tpre_m02\tpre_m10\tpre_m11\tpre_m12\tpre_m20\tpre_m21\tpre_m22",
        "\tpost_m00\tpost_m01\tpost_m02\tpost_m10\tpost_m11\tpost_m12\tpost_m20\tpost_m21\tpost_m22",
        "\tpre_ang_x\tpre_ang_y\tpre_ang_z",
        "\tpost_ang_x\tpost_ang_y\tpost_ang_z",
        "\tb70\tstate_1cc\tstate_1d0\tlatch_658\tobf_b44",
        "\tsub_70c\tsub_ba0\tsub_a74\tcontact_188\tcontact_ptr_18c",
        "\tforce_614_x\tforce_614_y\tforce_614_z",
        "\ttorque_620_x\ttorque_620_y\ttorque_620_z",
        "\taxis_ab8_x\taxis_ab8_y\taxis_ab8_z",
        "\taxis_a48_x\taxis_a48_y\taxis_a48_z",
        "\trail_base_bf8_m00\trail_base_bf8_m01\trail_base_bf8_m02",
        "\trail_base_bf8_m10\trail_base_bf8_m11\trail_base_bf8_m12",
        "\trail_base_bf8_m20\trail_base_bf8_m21\trail_base_bf8_m22",
        "\trail_constraint_c1c_m00\trail_constraint_c1c_m01\trail_constraint_c1c_m02",
        "\trail_constraint_c1c_m10\trail_constraint_c1c_m11\trail_constraint_c1c_m12",
        "\trail_constraint_c1c_m20\trail_constraint_c1c_m21\trail_constraint_c1c_m22\n"
    )
}

fn has_drive_surface_target(params: &BTreeMap<String, String>) -> bool {
    params.contains_key("track")
        && (params.contains_key("surface")
            || params.contains_key("rail")
            || params.contains_key("kap_dir"))
}

fn basic_drive_duration_ms(params: &BTreeMap<String, String>) -> u64 {
    for key in ["basic_drive_ms", "drive_ms", "drive_max_ms"] {
        if let Some(value) = params.get(key).and_then(|value| value.parse::<u64>().ok()) {
            return value;
        }
    }
    3000
}

fn basic_drive_keys(params: &BTreeMap<String, String>) -> Result<Vec<Key>, String> {
    parse_keys(params.get("drive_keys").map(String::as_str).unwrap_or("W"))
}

fn control_log_data_rows(path: &Path) -> usize {
    fs::read_to_string(path)
        .map(|text| text.lines().skip(1).count())
        .unwrap_or(0)
}

fn surface_drive_control_delta(path: &Path, start_rows: usize) -> (Vec<Key>, u64) {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return (Vec::new(), 0),
    };
    let mut first_keys = Vec::new();
    let mut total_duration_ms = 0_u64;
    for (idx, line) in text.lines().skip(1).enumerate() {
        if idx < start_rows {
            continue;
        }
        let mut cols = line.split('\t');
        let _time_ms = cols.next();
        let event = cols.next().unwrap_or("");
        let keys = cols.next().unwrap_or("");
        let duration_ms = cols
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        if event.ends_with("_step_begin") || event == "hold_begin" {
            if first_keys.is_empty() && !keys.is_empty() {
                if let Ok(parsed) = parse_keys(keys) {
                    first_keys = parsed;
                }
            }
            total_duration_ms = total_duration_ms.saturating_add(duration_ms);
        }
    }
    (first_keys, total_duration_ms)
}

fn params_bool(params: &BTreeMap<String, String>, key: &str) -> bool {
    params
        .get(key)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn bool_text(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn write_http_json(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        409 => "Conflict",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .map_err(|err| format!("write response: {err}"))
}

fn write_http_file(stream: &mut TcpStream, path: &Path, content_type: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|err| format!("stat {}: {err}", path.display()))?
        .len();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
    )
    .map_err(|err| format!("write file response header: {err}"))?;
    io::copy(&mut file, stream).map_err(|err| format!("write file response body: {err}"))?;
    Ok(())
}

#[derive(Clone, Debug)]
struct DaemonStatus {
    owner_rows: bool,
    pose_rows: bool,
    ground_rows: bool,
    collision_rows: bool,
    drivable: bool,
    state: &'static str,
}

impl DaemonStatus {
    fn from_dynout(dynout: &Path) -> Self {
        let owner_rows = has_tsv_data_row(&dynout.join("probe_owner.tsv"));
        let pose_rows = has_tsv_data_row(&dynout.join("probe_pose.tsv"));
        let ground_rows = has_tsv_data_row(&dynout.join("probe_ground_drive.tsv"));
        let collision_rows = has_tsv_data_row(&dynout.join("probe_collision_face.tsv"));
        let drivable = owner_rows && (pose_rows || ground_rows || collision_rows);
        let state = if drivable {
            "drivable"
        } else if owner_rows {
            "owner_only"
        } else {
            "no_owner"
        };
        Self {
            owner_rows,
            pose_rows,
            ground_rows,
            collision_rows,
            drivable,
            state,
        }
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"state\":\"{}\",\"owner_rows\":{},\"pose_rows\":{},\"ground_rows\":{},\"collision_rows\":{},\"drivable\":{}}}",
            self.state,
            bool_json(self.owner_rows),
            bool_json(self.pose_rows),
            bool_json(self.ground_rows),
            bool_json(self.collision_rows),
            bool_json(self.drivable)
        )
    }
}

fn daemon_health_json(config: &DaemonConfig) -> String {
    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"daemon\":\"kart-autoprobe\",",
            "\"api\":\"kr-live-v1\",",
            "\"bind\":\"{}\",",
            "\"dynout\":\"{}\",",
            "\"probe_root\":\"{}\",",
            "\"package_root\":\"{}\",",
            "\"capabilities\":[",
            "\"http.concurrent\",",
            "\"process.status\",",
            "\"process.module_bytes\",",
            "\"process.windows\",",
            "\"diagnostics.crash_events\",",
            "\"diagnostics.d3d9\",",
            "\"diagnostics.input_security\",",
            "\"state\",",
            "\"state.live\",",
            "\"state.self\",",
            "\"map.confirm\",",
            "\"state.rail_coverage\",",
            "\"route.kap_nearest\",",
            "\"events\",",
            "\"ui.return_room\",",
            "\"ui.select_practice_track\",",
            "\"capture.start\",",
            "\"capture.stop\",",
            "\"capture.package\",",
            "\"experiment.loop\",",
            "\"experiment.room_loop\",",
            "\"experiment.detached_start\",",
            "\"experiment.preflight\",",
            "\"experiment.status\",",
            "\"experiment.cancel\",",
            "\"packages.latest\",",
            "\"packages.by_run\",",
            "\"packages.download\",",
            "\"daemon.update_status\",",
            "\"daemon.restart_into_next\",",
            "\"desktop.esc_restart_probe\",",
            "\"control.evidence\",",
            "\"control.trace_hold\",",
            "\"control.drive_to\",",
            "\"control.drive_surface\",",
            "\"control.kap_follow\",",
            "\"control.stop_on_state_kind\",",
            "\"experiment.live_state\",",
            "\"experiment.manifest\"",
            "]",
            "}}"
        ),
        json_escape(&config.bind),
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string())
    )
}

fn daemon_exe_paths() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let current = env::current_exe().map_err(|err| format!("resolve current exe: {err}"))?;
    let dir = current
        .parent()
        .ok_or_else(|| format!("current exe has no parent: {}", current.display()))?;
    let target = dir.join("kart-autoprobe-daemon.exe");
    let next = dir.join("kart-autoprobe-daemon.next.exe");
    Ok((current, target, next))
}

fn daemon_update_status_json(config: &DaemonConfig) -> String {
    let (current, target, next) = daemon_exe_paths().unwrap_or_else(|_| {
        (
            PathBuf::new(),
            PathBuf::from("kart-autoprobe-daemon.exe"),
            PathBuf::from("kart-autoprobe-daemon.next.exe"),
        )
    });
    let current_file = file_status(&current, u128::MAX);
    let target_file = file_status(&target, u128::MAX);
    let next_file = file_status(&next, u128::MAX);
    let next_matches_target = files_equal_by_len_and_bytes(&next, &target).unwrap_or(false);
    let next_matches_current = files_equal_by_len_and_bytes(&next, &current).unwrap_or(false);
    let pending_restart = next_file.exists && !(next_matches_target || next_matches_current);
    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"daemon\":{{\"pid\":{},\"api\":\"kr-live-v1\",\"bind\":\"{}\",\"dynout\":\"{}\",\"probe_root\":\"{}\",\"package_root\":\"{}\"}},",
            "\"exe\":{{\"current\":\"{}\",\"target\":\"{}\",\"next\":\"{}\",\"current_file\":{},\"target_file\":{},\"next_file\":{},\"next_matches_target\":{},\"next_matches_current\":{},\"pending_restart\":{}}}",
            "}}"
        ),
        std::process::id(),
        json_escape(&config.bind),
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string()),
        json_escape(&current.display().to_string()),
        json_escape(&target.display().to_string()),
        json_escape(&next.display().to_string()),
        current_file.to_json(),
        target_file.to_json(),
        next_file.to_json(),
        bool_json(next_matches_target),
        bool_json(next_matches_current),
        bool_json(pending_restart)
    )
}

fn daemon_process_status_json(config: &DaemonConfig) -> String {
    match live_process_status_json(config) {
        Ok(body) => body,
        Err(err) => {
            let process = read_key_value_tsv(&config.dynout.join("probe_process.tsv"));
            format!(
                concat!(
                    "{{",
                    "\"ok\":true,",
                    "\"schema\":\"kr_process_status_v1\",",
                    "\"source\":\"dynout_fallback\",",
                    "\"error\":\"{}\",",
                    "\"dynout_file\":{},",
                    "\"processes\":[{{\"name\":\"KartRider\",\"pid\":\"{}\",\"path\":\"{}\",\"responding\":\"{}\",\"start_time\":\"{}\"}}]",
                    "}}"
                ),
                json_escape(&err),
                file_status(&config.dynout.join("probe_process.tsv"), 2000).to_json(),
                json_escape(process.get("pid").map(String::as_str).unwrap_or("")),
                json_escape(process.get("executable_path").map(String::as_str).unwrap_or("")),
                json_escape(process.get("responding").map(String::as_str).unwrap_or("")),
                json_escape(process.get("start_time").map(String::as_str).unwrap_or(""))
            )
        }
    }
}

fn launcher_start(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let mode = params
        .get("mode")
        .map(String::as_str)
        .unwrap_or("enter")
        .to_ascii_lowercase();
    let keys: Vec<Key> = match mode.as_str() {
        "click" | "click_start" | "click-start" => Vec::new(),
        "enter" => vec![Key::Enter],
        "tab_enter" | "tab-enter" | "tabenter" => vec![Key::Tab, Key::Enter],
        "space" => vec![Key::Space],
        "tab_space" | "tab-space" | "tabspace" => vec![Key::Tab, Key::Space],
        "tab_tab_enter" | "tab-tab-enter" => vec![Key::Tab, Key::Tab, Key::Enter],
        other => return Err(format!("unsupported launcher start mode: {other}")),
    };
    let before = daemon_process_status_json(config);
    let mut sent = Vec::new();
    let force_launcher_restart = params
        .get("restart_hidden")
        .is_some_and(|value| value.eq_ignore_ascii_case("force"));
    let recover_hidden =
        !params.contains_key("restart_hidden") || params_bool(params, "restart_hidden");
    if force_launcher_restart || (recover_hidden && launcher_start_button_point().is_err()) {
        let launcher_path = params
            .get("launcher_path")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\kart\kart\Launcher.exe"));
        restart_hidden_launcher(&launcher_path)?;
        sent.push("LauncherWindowRestart".to_string());
    }
    if keys.is_empty() {
        launcher_click_start()?;
        sent.push("LauncherStartClick".to_string());
    }
    for key in keys {
        launcher_post_key(key, 90)?;
        sent.push(key.name().to_string());
        thread::sleep(Duration::from_millis(150));
    }
    thread::sleep(Duration::from_millis(
        params
            .get("wait_ms")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(3000),
    ));
    let after = daemon_process_status_json(config);
    let detail = format!("mode={} keys={}", mode, sent.join("+"));
    append_daemon_event(config, "launcher_start", &detail);
    Ok(format!(
        "{{\"mode\":\"{}\",\"keys\":[{}],\"before\":{},\"after\":{}}}",
        json_escape(&mode),
        sent.iter()
            .map(|key| format!("\"{}\"", json_escape(key)))
            .collect::<Vec<_>>()
            .join(","),
        before,
        after
    ))
}

#[cfg(windows)]
fn restart_hidden_launcher(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("launcher executable not found: {}", path.display()));
    }
    let _ = ProcessCommand::new("taskkill")
        .args(["/F", "/IM", "Launcher.exe"])
        .output();
    thread::sleep(Duration::from_millis(500));
    let mut command = ProcessCommand::new(path);
    if let Some(parent) = path.parent() {
        command.current_dir(parent);
    }
    command
        .spawn()
        .map_err(|err| format!("restart launcher {}: {err}", path.display()))?;
    let deadline = tick_millis() + 10_000;
    while tick_millis() < deadline {
        if launcher_start_button_point().is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err(format!(
        "launcher restarted but no visible window appeared: {}",
        path.display()
    ))
}

#[cfg(not(windows))]
fn restart_hidden_launcher(_path: &Path) -> Result<(), String> {
    Err("Launcher recovery is only available on Windows".to_string())
}

#[cfg(windows)]
fn launcher_start_button_point() -> Result<(i32, i32), String> {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM, RECT};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowRect, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible,
    };
    unsafe extern "system" fn find(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; len as usize + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        let title = String::from_utf16_lossy(&buf[..copied.max(0) as usize]);
        if title == "Launcher" {
            *(lparam as *mut HWND) = hwnd;
            return 0;
        }
        1
    }
    let mut hwnd: HWND = std::ptr::null_mut();
    unsafe {
        EnumWindows(Some(find), &mut hwnd as *mut HWND as LPARAM);
    }
    if hwnd.is_null() {
        return Err("visible Launcher window not found".to_string());
    }
    let mut rect = RECT {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    if unsafe { GetWindowRect(hwnd, &mut rect) } == 0 {
        return Err("GetWindowRect(Launcher) failed".to_string());
    }
    // The launcher start button occupies the left three-fifths of its client
    // row. The window midpoint lands on the button's right border at the
    // observed 260 px layout and intermittently misses. Aim at the interior.
    Ok((rect.left + (rect.right - rect.left) * 3 / 10, rect.top + 64))
}

#[cfg(windows)]
fn launcher_click_start() -> Result<(), String> {
    let (x, y) = launcher_start_button_point()?;
    desktop_click(x, y)?;
    thread::sleep(Duration::from_millis(90));
    desktop_click(x, y)
}

#[cfg(not(windows))]
fn launcher_start_button_point() -> Result<(i32, i32), String> {
    Err("Launcher click is only available on Windows".to_string())
}

#[cfg(not(windows))]
fn launcher_click_start() -> Result<(), String> {
    Err("Launcher click is only available on Windows".to_string())
}

fn ui_clear_blockers(config: &DaemonConfig) -> Result<String, String> {
    let before_windows = daemon_process_windows_json();
    let before_blockers = current_ui_blockers(&before_windows);
    let actions = clear_known_ui_blockers()?;
    thread::sleep(Duration::from_millis(1000));
    let after_windows = daemon_process_windows_json();
    let after_process = daemon_process_status_json(config);
    let after_blockers = current_ui_blockers(&after_windows);
    append_daemon_event(
        config,
        "ui_clear_blockers",
        &format!(
            "before={} after={} actions={}",
            before_blockers.join("+"),
            after_blockers.join("+"),
            actions.join("+")
        ),
    );
    Ok(format!(
        "{{\"before_blockers\":[{}],\"after_blockers\":[{}],\"actions\":[{}],\"after_process\":{},\"after_windows\":{}}}",
        string_array_json(&before_blockers),
        string_array_json(&after_blockers),
        string_array_json(&actions),
        after_process,
        after_windows
    ))
}

#[cfg(windows)]
fn desktop_esc_restart_probe(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let press_esc = !params.contains_key("press_esc") || params_bool(params, "press_esc");
    let click = params_bool(params, "click");
    let x = params
        .get("x")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0);
    let y = params
        .get("y")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0);
    if click && !params.contains_key("x") {
        return Err("click=1 requires x".to_string());
    }
    if click && !params.contains_key("y") {
        return Err("click=1 requires y".to_string());
    }
    let delay_ms = params
        .get("delay_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(800)
        .min(10_000);
    let out_png = params
        .get("out_png")
        .map(PathBuf::from)
        .unwrap_or_else(|| config.dynout.join("esc_restart_probe.png"));
    let rust_focused = if params_bool(params, "focus") {
        focus_kart_rider_retry(2_000)
    } else {
        false
    };

    if press_esc {
        execute_with_backend(
            InputBackend::PostMessage,
            &[Command::Hold {
                keys: vec![Key::Esc],
                duration_ms: 80,
            }],
            &mut EventLog::from_optional_path(None)?,
        )?;
    }
    thread::sleep(Duration::from_millis(delay_ms));
    if click {
        desktop_click(x, y)?;
        if params_bool(params, "double_click") {
            thread::sleep(Duration::from_millis(90));
            desktop_click(x, y)?;
        }
        thread::sleep(Duration::from_millis(delay_ms));
    }
    let (width, height) = capture_primary_screen_bmp(&out_png)?;
    Ok(format!(
        concat!(
            "{{",
            "\"schema\":\"kr_desktop_esc_restart_probe_v2\",",
            "\"press_esc\":{},",
            "\"clicked\":{},\"double_clicked\":{},",
            "\"click_x\":{},",
            "\"click_y\":{},",
            "\"focused\":{},",
            "\"out_image\":\"{}\",",
            "\"screen\":{{\"left\":0,\"top\":0,\"width\":{},\"height\":{}}},",
            "\"process\":{}",
            "}}"
        ),
        bool_json(press_esc),
        bool_json(click),
        bool_json(click && params_bool(params, "double_click")),
        x,
        y,
        bool_json(rust_focused),
        json_escape(&out_png.display().to_string()),
        width,
        height,
        daemon_process_status_json(config)
    ))
}

#[cfg(windows)]
fn ui_return_room(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let before = current_live_state(config);
    let focus = if !params.contains_key("focus") || params_bool(params, "focus") {
        focus_kart_rider_retry(2_000)
    } else {
        false
    };
    let press_esc = !params.contains_key("press_esc") || params_bool(params, "press_esc");
    if press_esc && before.phase != "game_pause" {
        execute_with_backend(
            InputBackend::PostMessage,
            &[Command::Hold {
                keys: vec![Key::Esc],
                duration_ms: 80,
            }],
            &mut EventLog::from_optional_path(None)?,
        )?;
    }
    let menu_delay_ms = params
        .get("menu_delay_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(800)
        .min(10_000);
    thread::sleep(Duration::from_millis(menu_delay_ms));

    let (screen_w, screen_h) = primary_screen_size()?;
    let default_reset_x = screen_w / 2 - 214;
    let default_y = screen_h / 2 - 47;
    let default_x = default_reset_x + 114;
    let x = params
        .get("x")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default_x);
    let y = params
        .get("y")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default_y);
    desktop_click(x, y)?;
    let click_delay_ms = params
        .get("click_delay_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_500)
        .min(30_000);
    thread::sleep(Duration::from_millis(click_delay_ms));
    let after = current_live_state(config);
    append_daemon_event(
        config,
        "ui_return_room",
        &format!(
            "focus={} press_esc={} click={},{} before_phase={} after_phase={} after_control={}",
            focus, press_esc, x, y, before.phase, after.phase, after.control_allowed
        ),
    );
    Ok(format!(
        concat!(
            "{{",
            "\"schema\":\"kr_ui_return_room_v1\",",
            "\"focus\":{},",
            "\"press_esc\":{},",
            "\"click_x\":{},",
            "\"click_y\":{},",
            "\"default_x\":{},",
            "\"default_y\":{},",
            "\"default_reset_x\":{},",
            "\"default_room_offset_x\":{},",
            "\"screen\":{{\"width\":{},\"height\":{}}},",
            "\"before\":{},",
            "\"after\":{}",
            "}}"
        ),
        bool_json(focus),
        bool_json(press_esc),
        x,
        y,
        default_x,
        default_y,
        default_reset_x,
        default_x - default_reset_x,
        screen_w,
        screen_h,
        before.to_json(),
        after.to_json()
    ))
}

#[cfg(windows)]
fn desktop_click(x: i32, y: i32) -> Result<(), String> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        mouse_event, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::SetCursorPos;
    unsafe {
        if SetCursorPos(x, y) == 0 {
            return Err(format!("SetCursorPos failed for {x},{y}"));
        }
        thread::sleep(Duration::from_millis(60));
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
        thread::sleep(Duration::from_millis(50));
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
    }
    Ok(())
}

fn search_result_grid_is_populated(non_background: u64, total: u64) -> bool {
    total >= 20 && non_background >= 4 && non_background * 20 >= total
}

#[cfg(windows)]
fn validate_track_search_result(path: &Path) -> Result<(bool, u64, u64), String> {
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Stop';",
            "Add-Type -AssemblyName System.Drawing;",
            "$bmp=[System.Drawing.Bitmap]::FromFile({});",
            "$sx=[double]$bmp.Width/1920.0;$sy=[double]$bmp.Height/1080.0;",
            "$bg=$bmp.GetPixel([int](1000*$sx),[int](500*$sy));",
            "$different=0;$total=0;",
            "for($x=330;$x -le 700;$x+=20){{",
            "for($y=170;$y -le 520;$y+=20){{",
            "$p=$bmp.GetPixel([int]($x*$sx),[int]($y*$sy));",
            "$delta=[math]::Abs([int]$p.R-[int]$bg.R)+[math]::Abs([int]$p.G-[int]$bg.G)+[math]::Abs([int]$p.B-[int]$bg.B);",
            "if($delta -gt 30){{$different++}};$total++",
            "}}}};",
            "$bmp.Dispose();",
            "[pscustomobject]@{{non_background=$different;total=$total}} | ConvertTo-Json -Compress"
        ),
        ps_quote_path(path)
    );
    let output = run_powershell(&ps)?;
    let body = output.trim();
    let non_background = json_extract_number_after(body, "", "\"non_background\":")
        .ok_or_else(|| format!("track-search validation returned no non_background: {body}"))?
        .parse::<u64>()
        .map_err(|_| format!("track-search validation returned invalid count: {body}"))?;
    let total = json_extract_number_after(body, "", "\"total\":")
        .ok_or_else(|| format!("track-search validation returned no total: {body}"))?
        .parse::<u64>()
        .map_err(|_| format!("track-search validation returned invalid total: {body}"))?;
    Ok((
        search_result_grid_is_populated(non_background, total),
        non_background,
        total,
    ))
}

#[cfg(windows)]
fn ui_select_practice_track(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let track_id = params
        .get("track_id")
        .map(String::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or("ui select-practice-track requires track_id")?;
    let display_name = params
        .get("display_name")
        .map(String::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or("ui select-practice-track requires display_name")?;
    let (screen_w, screen_h) = primary_screen_size()?;
    let sx = |x: i32| x * screen_w / 1920;
    let sy = |y: i32| y * screen_h / 1080;
    let evidence_dir = config.dynout.join("ui_map_select");
    fs::create_dir_all(&evidence_dir)
        .map_err(|err| format!("create {}: {err}", evidence_dir.display()))?;
    let before_png = evidence_dir.join(format!("{track_id}_before.png"));
    let search_png = evidence_dir.join(format!("{track_id}_search.png"));
    let after_png = evidence_dir.join(format!("{track_id}_after.png"));
    let started_png = evidence_dir.join(format!("{track_id}_started.png"));
    let focused = focus_kart_rider_retry(2_000);
    if params_bool(params, "clear_startup") {
        let login_wait_ms = params
            .get("login_wait_ms")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(10_000)
            .min(30_000);
        thread::sleep(Duration::from_millis(login_wait_ms));
        // The 2026 CN client can open the full-screen 惹头兑换所 after
        // login.  It is not a top-level Win32 modal, so window enumeration
        // cannot see it; close its observed in-game X before attempting the
        // lobby navigation clicks.  On an unobstructed lobby this point is
        // inert.
        desktop_click(sx(1458), sy(108))?;
        thread::sleep(Duration::from_millis(700));
        desktop_click(sx(800), sy(500))?;
        thread::sleep(Duration::from_millis(900));
        desktop_click(sx(802), sy(720))?;
        thread::sleep(Duration::from_millis(700));
    }
    // Enter the only supported MVP mode: single-player -> practice time trial.
    desktop_click(sx(595), sy(892))?;
    thread::sleep(Duration::from_millis(700));
    desktop_click(sx(786), sy(370))?;
    thread::sleep(Duration::from_millis(700));
    desktop_click(sx(265), sy(300))?;
    thread::sleep(Duration::from_millis(1_000));
    capture_primary_screen_bmp(&before_png)?;

    // MVP contract: caller has already entered the single-player time-trial room.
    // Open the track picker, focus its exact-name search box, and paste Unicode
    // through the interactive daemon session.
    desktop_click(sx(1325), sy(812))?;
    thread::sleep(Duration::from_millis(800));
    desktop_click(sx(1340), sy(137))?;
    let clipboard_ps = format!(
        "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetText({})",
        ps_quote(display_name)
    );
    run_powershell(&clipboard_ps)?;
    execute_with_backend(
        InputBackend::SendInput,
        &[
            Command::Hold {
                keys: vec![Key::Ctrl, Key::A],
                duration_ms: 80,
            },
            Command::Hold {
                keys: vec![Key::Ctrl, Key::V],
                duration_ms: 80,
            },
            Command::Hold {
                keys: vec![Key::Enter],
                duration_ms: 80,
            },
        ],
        &mut EventLog::from_optional_path(None)?,
    )?;
    thread::sleep(Duration::from_millis(1_000));
    capture_primary_screen_bmp(&search_png)?;
    let (search_result_populated, search_non_background, search_total) =
        validate_track_search_result(&search_png)?;
    if !search_result_populated {
        append_daemon_event(
            config,
            "ui_select_practice_track_rejected",
            &format!(
                "track_id={track_id} display_name={display_name} reason=empty_search_result \
                 non_background={search_non_background}/{search_total}"
            ),
        );
        return Err(format!(
            "exact track search produced no selectable result: track_id={track_id} \
             display_name={display_name} non_background={search_non_background}/{search_total} \
             evidence={}",
            search_png.display()
        ));
    }
    // Exact-name search is expected to leave the requested track as the first card.
    desktop_click(sx(460), sy(260))?;
    desktop_click(sx(725), sy(844))?;
    thread::sleep(Duration::from_millis(1_200));
    capture_primary_screen_bmp(&after_png)?;
    let started = params_bool(params, "start");
    if started {
        desktop_click(sx(1325), sy(668))?;
        // Do not send F5 after the click. Two independent fresh-process runs
        // showed that the button had already entered the track and a delayed
        // F5 then caused a clean client exit a few seconds later. Capturing
        // afterwards makes "started" inspectable without a destructive
        // redundant shortcut.
        thread::sleep(Duration::from_millis(1_500));
        capture_primary_screen_bmp(&started_png)?;
    }
    append_daemon_event(
        config,
        "ui_select_practice_track",
        &format!("track_id={track_id} display_name={display_name} focused={focused}"),
    );
    Ok(format!(
        concat!(
            "{{\"schema\":\"kr_ui_select_practice_track_v1\",",
            "\"track_id\":\"{}\",\"display_name\":\"{}\",\"focused\":{},\"started\":{},",
            "\"screen\":{{\"width\":{},\"height\":{}}},",
            "\"search_validation\":{{\"populated\":true,\"non_background\":{},\"total\":{}}},",
            "\"evidence\":{{\"before\":\"{}\",\"search\":\"{}\",\"after\":\"{}\",\"started\":\"{}\"}},",
            "\"runtime_validation\":\"required_after_race_start\"}}"
        ),
        json_escape(track_id),
        json_escape(display_name),
        bool_json(focused),
        bool_json(started),
        screen_w,
        screen_h,
        search_non_background,
        search_total,
        json_escape(&before_png.display().to_string()),
        json_escape(&search_png.display().to_string()),
        json_escape(&after_png.display().to_string()),
        if started {
            json_escape(&started_png.display().to_string())
        } else {
            String::new()
        }
    ))
}

#[cfg(windows)]
fn primary_screen_size() -> Result<(i32, i32), String> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

    unsafe {
        let width = GetSystemMetrics(SM_CXSCREEN);
        let height = GetSystemMetrics(SM_CYSCREEN);
        if width <= 0 || height <= 0 {
            return Err(format!("invalid primary screen size {width}x{height}"));
        }
        Ok((width, height))
    }
}

#[cfg(windows)]
fn capture_primary_screen_bmp(path: &Path) -> Result<(i32, i32), String> {
    let path_text = path.display().to_string();
    let parent_text = path
        .parent()
        .map(|parent| parent.display().to_string())
        .unwrap_or_else(|| ".".to_string());
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Stop';",
            "Add-Type -AssemblyName System.Windows.Forms;",
            "Add-Type -AssemblyName System.Drawing;",
            "$out={};",
            "$dir={};",
            "New-Item -ItemType Directory -Force -Path $dir | Out-Null;",
            "$bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;",
            "$bmp=New-Object System.Drawing.Bitmap $bounds.Width,$bounds.Height;",
            "$g=[System.Drawing.Graphics]::FromImage($bmp);",
            "$g.CopyFromScreen($bounds.Left,$bounds.Top,0,0,$bounds.Size);",
            "$bmp.Save($out,[System.Drawing.Imaging.ImageFormat]::Png);",
            "$g.Dispose();$bmp.Dispose();",
            "[pscustomobject]@{{ok=$true;width=$bounds.Width;height=$bounds.Height;out_png=$out}} | ConvertTo-Json -Compress -Depth 3"
        ),
        ps_quote(&path_text),
        ps_quote(&parent_text)
    );
    let output = run_powershell(&ps)?;
    let body = output.trim();
    let width = json_extract_number_after(body, "", "\"width\":")
        .ok_or_else(|| format!("screenshot powershell returned no width: {body}"))?
        .parse::<i32>()
        .map_err(|_| format!("screenshot powershell returned invalid width: {body}"))?;
    let height = json_extract_number_after(body, "", "\"height\":")
        .ok_or_else(|| format!("screenshot powershell returned no height: {body}"))?
        .parse::<i32>()
        .map_err(|_| format!("screenshot powershell returned invalid height: {body}"))?;
    Ok((width, height))
}

#[cfg(not(windows))]
fn desktop_esc_restart_probe(
    _config: &DaemonConfig,
    _params: &BTreeMap<String, String>,
) -> Result<String, String> {
    Err("desktop_esc_restart_probe is only available on Windows daemon hosts".to_string())
}

#[cfg(not(windows))]
fn ui_return_room(
    _config: &DaemonConfig,
    _params: &BTreeMap<String, String>,
) -> Result<String, String> {
    Err("ui_return_room is only available on Windows daemon hosts".to_string())
}

#[cfg(not(windows))]
fn ui_select_practice_track(
    _config: &DaemonConfig,
    _params: &BTreeMap<String, String>,
) -> Result<String, String> {
    Err("ui_select_practice_track is only available on Windows daemon hosts".to_string())
}

#[cfg(windows)]
fn live_process_status_json(_config: &DaemonConfig) -> Result<String, String> {
    let ps = concat!(
        "$ErrorActionPreference='Continue';",
        "$items=@();",
        "foreach($name in @('KartRider','Launcher')){",
        "Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {",
        "$path='';$baseHex='';$baseDec=0;$moduleSize=0;",
        "try{$m=$_.MainModule;$path=$m.FileName;$baseDec=[int64]$m.BaseAddress;$baseHex=('0x{0:X}' -f $baseDec);$moduleSize=[int64]$m.ModuleMemorySize}catch{};",
        "$items += [pscustomobject]@{",
        "name=$_.ProcessName;",
        "pid=$_.Id;",
        "responding=$_.Responding;",
        "main_window_title=$_.MainWindowTitle;",
        "main_window_handle=$_.MainWindowHandle;",
        "start_time=if($_.StartTime){$_.StartTime.ToUniversalTime().ToString('o')}else{''};",
        "path=$path;",
        "base_address=$baseDec;",
        "base_address_hex=$baseHex;",
        "module_size=$moduleSize",
        "}",
        "}",
        "};",
        "[pscustomobject]@{",
        "ok=$true;",
        "schema='kr_process_status_v1';",
        "source='powershell_live';",
        "time_ms=[int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds());",
        "processes=$items",
        "} | ConvertTo-Json -Compress -Depth 4"
    );
    let text = run_powershell(ps)?;
    let body = text.trim();
    if body.starts_with('{') {
        Ok(body.to_string())
    } else {
        Err(format!("process status did not return JSON: {body}"))
    }
}

#[cfg(not(windows))]
fn live_process_status_json(_config: &DaemonConfig) -> Result<String, String> {
    Err("live process status is only available on Windows".to_string())
}

#[cfg(windows)]
fn daemon_process_module_bytes_json(params: &BTreeMap<String, String>) -> String {
    let pid = params
        .get("pid")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let process_name = params
        .get("name")
        .map(String::as_str)
        .unwrap_or("KartRider");
    let rva_text = params.get("rva").map(String::as_str).unwrap_or("");
    let size = params
        .get("size")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(16)
        .clamp(1, 256);
    let rva = parse_u64_query(rva_text).unwrap_or(u64::MAX);
    if rva == u64::MAX {
        return format!(
            "{{\"ok\":false,\"schema\":\"kr_process_module_bytes_v1\",\"error\":\"invalid_rva\",\"rva\":\"{}\"}}",
            json_escape(rva_text)
        );
    }
    let selector = if pid != 0 {
        format!("$p=Get-Process -Id {} -ErrorAction Stop;", pid)
    } else {
        format!(
            "$p=Get-Process -Name {} -ErrorAction Stop | Select-Object -First 1;",
            ps_quote(process_name)
        )
    };
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Stop';",
            "{}",
            "$m=$p.MainModule;",
            "$base=[int64]$m.BaseAddress;",
            "$moduleSize=[int64]$m.ModuleMemorySize;",
            "$rva=[int64]{};",
            "$size=[int]{};",
            "if($rva -lt 0 -or ($rva + $size) -gt $moduleSize){{throw \"requested bytes outside main module\"}}",
            "$source=@\"",
            "\nusing System;",
            "\nusing System.Runtime.InteropServices;",
            "\npublic static class KRModuleBytes {{",
            "\n  [DllImport(\"kernel32.dll\", SetLastError=true)] public static extern IntPtr OpenProcess(UInt32 access, bool inherit, UInt32 pid);",
            "\n  [DllImport(\"kernel32.dll\", SetLastError=true)] public static extern bool ReadProcessMemory(IntPtr hProcess, IntPtr address, byte[] buffer, UIntPtr size, out UIntPtr bytesRead);",
            "\n  [DllImport(\"kernel32.dll\", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);",
            "\n}}",
            "\n\"@;",
            "Add-Type -TypeDefinition $source -ErrorAction SilentlyContinue;",
            "$h=[KRModuleBytes]::OpenProcess(0x0410,$false,[uint32]$p.Id);",
            "if($h -eq [IntPtr]::Zero){{throw \"OpenProcess failed\"}}",
            "$buf=New-Object byte[] $size;",
            "$read=[UIntPtr]::Zero;",
            "$uSize=[UIntPtr]::new([uint64]$size);",
            "$ok=[KRModuleBytes]::ReadProcessMemory($h,[IntPtr]($base+$rva),$buf,$uSize,[ref]$read);",
            "[void][KRModuleBytes]::CloseHandle($h);",
            "if(-not $ok){{throw \"ReadProcessMemory failed\"}}",
            "$n=[int]$read.ToUInt64();",
            "$hex=([BitConverter]::ToString($buf,0,$n)).Replace('-',' ').ToLowerInvariant();",
            "[pscustomobject]@{{ok=$true;schema='kr_process_module_bytes_v1';source='powershell_readprocessmemory';time_ms=[int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds());name=$p.ProcessName;pid=$p.Id;path=$m.FileName;base_address=$base;base_address_hex=('0x{{0:X}}' -f $base);module_size=$moduleSize;rva=$rva;rva_hex=('0x{{0:X}}' -f $rva);address=$base+$rva;address_hex=('0x{{0:X}}' -f ($base+$rva));size=$size;bytes_read=$n;hex=$hex}} | ConvertTo-Json -Compress -Depth 4"
        ),
        selector, rva, size
    );
    match run_powershell(&ps) {
        Ok(text) => {
            let body = text.trim();
            if body.starts_with('{') {
                body.to_string()
            } else {
                format!(
                    "{{\"ok\":false,\"schema\":\"kr_process_module_bytes_v1\",\"error\":\"{}\"}}",
                    json_escape(body)
                )
            }
        }
        Err(err) => format!(
            "{{\"ok\":false,\"schema\":\"kr_process_module_bytes_v1\",\"error\":\"{}\"}}",
            json_escape(&err)
        ),
    }
}

#[cfg(not(windows))]
fn daemon_process_module_bytes_json(_params: &BTreeMap<String, String>) -> String {
    "{\"ok\":false,\"schema\":\"kr_process_module_bytes_v1\",\"error\":\"process module byte reads are only available on Windows daemon hosts\"}".to_string()
}

#[cfg(windows)]
fn daemon_crash_events_json(params: &BTreeMap<String, String>) -> String {
    let hours = params
        .get("hours")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(6)
        .clamp(1, 72);
    let limit = params
        .get("limit")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20)
        .clamp(1, 100);
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Continue';",
            "$start=(Get-Date).AddHours(-{});",
            "$events=@(Get-WinEvent -FilterHashtable @{{LogName='Application'; StartTime=$start; Id=1000,1001,1002}} -ErrorAction SilentlyContinue | ",
            "Where-Object {{ $_.ProviderName -in @('Application Error','Windows Error Reporting') -or $_.Message -like '*KartRider*' -or $_.Message -like '*KRPhysicsProbe*' }} | ",
            "Sort-Object TimeCreated -Descending | Select-Object -First {} TimeCreated,Id,ProviderName,LevelDisplayName,@{{n='MessageHead';e={{$m=$_.Message -replace '[\\r\\n]+',' '; if($m.Length -gt 1200){{$m.Substring(0,1200)}}else{{$m}}}}}});",
            "$wer=@();",
            "foreach($root in @('C:\\ProgramData\\Microsoft\\Windows\\WER\\ReportArchive','C:\\ProgramData\\Microsoft\\Windows\\WER\\ReportQueue')){{",
            "if(Test-Path $root){{",
            "Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ",
            "Where-Object {{ $_.Name -like '*KartRider*' -or $_.Name -like '*KRPhysicsProbe*' -or $_.Name -like '*AppCrash*' }} | ",
            "Sort-Object LastWriteTime -Descending | Select-Object -First {} FullName,Name,LastWriteTime | ForEach-Object {{$wer += $_}}",
            "}}",
            "}};",
            "[pscustomobject]@{{ok=$true;schema='kr_crash_events_v1';source='windows_eventlog_wer';hours={};limit={};time_ms=[int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds());events=$events;wer_reports=$wer}} | ConvertTo-Json -Compress -Depth 5"
        ),
        hours, limit, limit, hours, limit
    );
    match run_powershell(&ps) {
        Ok(text) => {
            let body = text.trim();
            if body.starts_with('{') {
                body.to_string()
            } else {
                format!(
                    "{{\"ok\":false,\"schema\":\"kr_crash_events_v1\",\"source\":\"windows_eventlog_wer\",\"error\":\"{}\"}}",
                    json_escape(body)
                )
            }
        }
        Err(err) => format!(
            "{{\"ok\":false,\"schema\":\"kr_crash_events_v1\",\"source\":\"windows_eventlog_wer\",\"error\":\"{}\"}}",
            json_escape(&err)
        ),
    }
}

#[cfg(not(windows))]
fn daemon_crash_events_json(_params: &BTreeMap<String, String>) -> String {
    "{\"ok\":false,\"schema\":\"kr_crash_events_v1\",\"source\":\"unavailable\",\"error\":\"crash events are only available on Windows\"}".to_string()
}

#[cfg(windows)]
fn daemon_process_windows_json() -> String {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetClassNameW, GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW,
        GetWindowThreadProcessId, IsWindowVisible,
    };

    #[derive(Default)]
    struct WindowRows {
        rows: Vec<String>,
    }

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let rows = &mut *(lparam as *mut WindowRows);
        let len = GetWindowTextLengthW(hwnd);
        let mut title = String::new();
        if len > 0 {
            let mut buf = vec![0_u16; (len as usize) + 1];
            let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
            if copied > 0 {
                title = String::from_utf16_lossy(&buf[..copied as usize]);
            }
        }
        if title.is_empty() {
            return 1;
        }
        let mut pid = 0_u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        let mut class_buf = vec![0_u16; 256];
        let class_len = GetClassNameW(hwnd, class_buf.as_mut_ptr(), class_buf.len() as i32);
        let class_name = if class_len > 0 {
            String::from_utf16_lossy(&class_buf[..class_len as usize])
        } else {
            String::new()
        };
        let foreground = GetForegroundWindow();
        rows.rows.push(format!(
            "{{\"hwnd\":\"0x{:X}\",\"pid\":{},\"visible\":{},\"foreground\":{},\"class\":\"{}\",\"title\":\"{}\"}}",
            hwnd as usize,
            pid,
            bool_json(IsWindowVisible(hwnd) != 0),
            bool_json(foreground == hwnd),
            json_escape(&class_name),
            json_escape(&title)
        ));
        1
    }

    let mut rows = WindowRows::default();
    unsafe {
        EnumWindows(Some(enum_window), &mut rows as *mut WindowRows as LPARAM);
    }
    format!(
        "{{\"ok\":true,\"schema\":\"kr_process_windows_v1\",\"source\":\"enum_windows\",\"windows\":[{}]}}",
        rows.rows.join(",")
    )
}

#[cfg(not(windows))]
fn daemon_process_windows_json() -> String {
    "{\"ok\":true,\"schema\":\"kr_process_windows_v1\",\"source\":\"not_windows\",\"windows\":[]}"
        .to_string()
}

#[cfg(not(windows))]
fn daemon_d3d9_json() -> String {
    "{\"ok\":false,\"source\":\"unsupported\",\"error\":\"d3d9 diagnostics are only available on Windows\"}".to_string()
}

fn input_security_diagnosis(
    daemon_rid: u32,
    game_rid: u32,
    input_desktop: &str,
    foreground_present: bool,
) -> (&'static str, bool) {
    if daemon_rid < game_rid {
        ("daemon_integrity_below_game_uipi_blocks_sendinput", false)
    } else if input_desktop.is_empty() || !foreground_present {
        ("no_active_input_desktop", false)
    } else {
        ("integrity_levels_allow_sendinput", true)
    }
}

#[cfg(windows)]
fn ensure_control_input_ready(config: &DaemonConfig) -> Result<(), String> {
    let process_status = daemon_process_status_json_cached(config);
    let pid = live_kart_pid_from_process_status(&process_status)
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| "control input unavailable: KartRider PID unavailable".to_string())?;
    let daemon = process_integrity_level(None)?;
    let game = process_integrity_level(Some(pid))?;
    let desktop = input_desktop_context()?;
    let (diagnosis, allowed) = input_security_diagnosis(
        daemon.rid,
        game.rid,
        &desktop.input_desktop,
        desktop.foreground_present,
    );
    if !allowed {
        return Err(format!(
            "control input unavailable: {diagnosis}; daemon_integrity={} game_integrity={} input_desktop={} foreground_present={}",
            daemon.label,
            game.label,
            desktop.input_desktop,
            desktop.foreground_present
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn ensure_control_input_ready(_config: &DaemonConfig) -> Result<(), String> {
    Ok(())
}

fn daemon_input_security_json(config: &DaemonConfig) -> String {
    let process_status = daemon_process_status_json_cached(config);
    let pid = live_kart_pid_from_process_status(&process_status)
        .and_then(|value| value.parse::<u32>().ok());
    let daemon = process_integrity_level(None);
    let game = pid
        .ok_or_else(|| "KartRider PID unavailable".to_string())
        .and_then(|value| process_integrity_level(Some(value)));
    let desktop = input_desktop_context();
    match (daemon, game, desktop) {
        (Ok(daemon), Ok(game), Ok(desktop)) => {
            let (diagnosis, sendinput_integrity_allowed) = input_security_diagnosis(
                daemon.rid,
                game.rid,
                &desktop.input_desktop,
                desktop.foreground_present,
            );
            format!(
                concat!(
                    "{{\"ok\":true,\"schema\":\"kr_input_security_v1\",",
                    "\"kart_pid\":{},",
                    "\"daemon\":{{\"integrity_rid\":{},\"integrity\":\"{}\"}},",
                    "\"game\":{{\"integrity_rid\":{},\"integrity\":\"{}\"}},",
                    "\"desktop\":{{\"window_station\":\"{}\",\"daemon\":\"{}\",",
                    "\"input\":\"{}\",\"game\":\"{}\",\"foreground_present\":{},",
                    "\"input_struct_bytes\":{}}},",
                    "\"sendinput_integrity_allowed\":{},\"diagnosis\":\"{}\"}}"
                ),
                pid.unwrap_or(0),
                daemon.rid,
                daemon.label,
                game.rid,
                game.label,
                json_escape(&desktop.window_station),
                json_escape(&desktop.daemon_desktop),
                json_escape(&desktop.input_desktop),
                json_escape(&desktop.game_desktop),
                bool_json(desktop.foreground_present),
                desktop.input_struct_bytes,
                bool_json(sendinput_integrity_allowed),
                diagnosis
            )
        }
        (daemon, game, desktop) => format!(
            "{{\"ok\":false,\"schema\":\"kr_input_security_v1\",\"kart_pid\":{},\"daemon_error\":\"{}\",\"game_error\":\"{}\",\"desktop_error\":\"{}\"}}",
            pid.unwrap_or(0),
            json_escape(&daemon.err().unwrap_or_default()),
            json_escape(&game.err().unwrap_or_default()),
            json_escape(&desktop.err().unwrap_or_default())
        ),
    }
}

#[cfg(windows)]
fn daemon_d3d9_json() -> String {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_void};

    const D3D_SDK_VERSION: u32 = 32;
    const D3DDEVTYPE_HAL: u32 = 1;
    const D3DFMT_UNKNOWN: u32 = 0;
    const D3DFMT_X8R8G8B8: u32 = 22;
    const D3DFMT_R5G6B5: u32 = 23;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct D3dDisplayMode {
        width: u32,
        height: u32,
        refresh_rate: u32,
        format: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct D3dAdapterIdentifier9 {
        driver: [c_char; 512],
        description: [c_char; 512],
        device_name: [c_char; 32],
        driver_version: i64,
        vendor_id: u32,
        device_id: u32,
        subsys_id: u32,
        revision: u32,
        device_identifier: [u8; 16],
        whql_level: u32,
    }

    impl Default for D3dAdapterIdentifier9 {
        fn default() -> Self {
            Self {
                driver: [0; 512],
                description: [0; 512],
                device_name: [0; 32],
                driver_version: 0,
                vendor_id: 0,
                device_id: 0,
                subsys_id: 0,
                revision: 0,
                device_identifier: [0; 16],
                whql_level: 0,
            }
        }
    }

    type Direct3DCreate9Fn = unsafe extern "system" fn(u32) -> *mut c_void;
    type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
    type GetAdapterCountFn = unsafe extern "system" fn(*mut c_void) -> u32;
    type GetAdapterIdentifierFn =
        unsafe extern "system" fn(*mut c_void, u32, u32, *mut D3dAdapterIdentifier9) -> i32;
    type GetAdapterDisplayModeFn =
        unsafe extern "system" fn(*mut c_void, u32, *mut D3dDisplayMode) -> i32;
    type CheckDeviceTypeFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, i32) -> i32;

    #[link(name = "d3d9")]
    extern "system" {
        #[link_name = "Direct3DCreate9"]
        fn direct3d_create9(sdk_version: u32) -> *mut c_void;
    }

    unsafe fn vtbl<T: Copy>(object: *mut c_void, slot: usize) -> T {
        let table = *(object as *mut *mut *mut c_void);
        std::mem::transmute_copy(&*table.add(slot))
    }

    unsafe fn c_char_array_to_string(value: &[c_char]) -> String {
        if value.is_empty() || value[0] == 0 {
            return String::new();
        }
        CStr::from_ptr(value.as_ptr())
            .to_string_lossy()
            .into_owned()
    }

    fn hresult_json(value: i32) -> String {
        let raw = value as u32;
        let name = match raw {
            0x00000000 => "S_OK",
            0x8876086A => "D3DERR_NOTAVAILABLE",
            0x8876086C => "D3DERR_INVALIDCALL",
            0x8876086E => "D3DERR_OUTOFVIDEOMEMORY",
            0x88760868 => "D3DERR_DEVICELOST",
            0x88760869 => "D3DERR_DEVICENOTRESET",
            _ => "",
        };
        format!(
            "{{\"signed\":{},\"hex\":\"0x{:08X}\",\"name\":\"{}\"}}",
            value,
            raw,
            json_escape(name)
        )
    }

    unsafe {
        let create_fn: Direct3DCreate9Fn = direct3d_create9;
        let d3d = create_fn(D3D_SDK_VERSION);
        if d3d.is_null() {
            return "{\"ok\":false,\"source\":\"daemon_d3d9\",\"error\":\"Direct3DCreate9 returned null\"}".to_string();
        }
        let release: ReleaseFn = vtbl(d3d, 2);
        let get_adapter_count: GetAdapterCountFn = vtbl(d3d, 4);
        let get_adapter_identifier: GetAdapterIdentifierFn = vtbl(d3d, 5);
        let get_adapter_display_mode: GetAdapterDisplayModeFn = vtbl(d3d, 8);
        let check_device_type: CheckDeviceTypeFn = vtbl(d3d, 9);
        let adapter_count = get_adapter_count(d3d);
        let mut rows = Vec::new();
        for adapter in 0..adapter_count {
            let mut ident = D3dAdapterIdentifier9::default();
            let ident_hr = get_adapter_identifier(d3d, adapter, 0, &mut ident);
            let mut mode = D3dDisplayMode::default();
            let mode_hr = get_adapter_display_mode(d3d, adapter, &mut mode);
            let adapter_format = if mode_hr == 0 {
                mode.format
            } else {
                D3DFMT_X8R8G8B8
            };
            rows.push(format!(
                concat!(
                    "{{",
                    "\"adapter\":{},",
                    "\"identifier_hr\":{},",
                    "\"description\":\"{}\",",
                    "\"device_name\":\"{}\",",
                    "\"vendor_id\":{},",
                    "\"vendor_hex\":\"0x{:04X}\",",
                    "\"device_id\":{},",
                    "\"device_hex\":\"0x{:04X}\",",
                    "\"display_mode_hr\":{},",
                    "\"width\":{},",
                    "\"height\":{},",
                    "\"refresh_rate\":{},",
                    "\"adapter_format\":{},",
                    "\"check_x8r8g8b8_windowed\":{},",
                    "\"check_x8r8g8b8_fullscreen\":{},",
                    "\"check_r5g6b5_windowed\":{},",
                    "\"check_r5g6b5_fullscreen\":{},",
                    "\"check_unknown_x8_windowed\":{}",
                    "}}"
                ),
                adapter,
                hresult_json(ident_hr),
                json_escape(&c_char_array_to_string(&ident.description)),
                json_escape(&c_char_array_to_string(&ident.device_name)),
                ident.vendor_id,
                ident.vendor_id,
                ident.device_id,
                ident.device_id,
                hresult_json(mode_hr),
                mode.width,
                mode.height,
                mode.refresh_rate,
                adapter_format,
                hresult_json(check_device_type(
                    d3d,
                    adapter,
                    D3DDEVTYPE_HAL,
                    adapter_format,
                    D3DFMT_X8R8G8B8,
                    1
                )),
                hresult_json(check_device_type(
                    d3d,
                    adapter,
                    D3DDEVTYPE_HAL,
                    adapter_format,
                    D3DFMT_X8R8G8B8,
                    0
                )),
                hresult_json(check_device_type(
                    d3d,
                    adapter,
                    D3DDEVTYPE_HAL,
                    adapter_format,
                    D3DFMT_R5G6B5,
                    1
                )),
                hresult_json(check_device_type(
                    d3d,
                    adapter,
                    D3DDEVTYPE_HAL,
                    adapter_format,
                    D3DFMT_R5G6B5,
                    0
                )),
                hresult_json(check_device_type(
                    d3d,
                    adapter,
                    D3DDEVTYPE_HAL,
                    D3DFMT_UNKNOWN,
                    D3DFMT_X8R8G8B8,
                    1
                ))
            ));
        }
        release(d3d);
        format!(
            "{{\"ok\":true,\"source\":\"daemon_d3d9\",\"adapter_count\":{},\"adapters\":[{}]}}",
            adapter_count,
            rows.join(",")
        )
    }
}

#[cfg(not(windows))]
fn daemon_set_display_mode_json(_params: &BTreeMap<String, String>) -> Result<String, String> {
    Err("display mode changes are only available on Windows".to_string())
}

#[cfg(windows)]
fn daemon_set_display_mode_json(params: &BTreeMap<String, String>) -> Result<String, String> {
    const CCHDEVICENAME: usize = 32;
    const CCHFORMNAME: usize = 32;
    const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
    const DM_BITSPERPEL: u32 = 0x0004_0000;
    const DM_PELSWIDTH: u32 = 0x0008_0000;
    const DM_PELSHEIGHT: u32 = 0x0010_0000;
    const DM_DISPLAYFREQUENCY: u32 = 0x0040_0000;
    const CDS_UPDATEREGISTRY: u32 = 0x0000_0001;
    const CDS_TEST: u32 = 0x0000_0002;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DevModeW {
        dm_device_name: [u16; CCHDEVICENAME],
        dm_spec_version: u16,
        dm_driver_version: u16,
        dm_size: u16,
        dm_driver_extra: u16,
        dm_fields: u32,
        dm_position_x: i32,
        dm_position_y: i32,
        dm_display_orientation: u32,
        dm_display_fixed_output: u32,
        dm_color: i16,
        dm_duplex: i16,
        dm_y_resolution: i16,
        dm_tt_option: i16,
        dm_collate: i16,
        dm_form_name: [u16; CCHFORMNAME],
        dm_log_pixels: u16,
        dm_bits_per_pel: u32,
        dm_pels_width: u32,
        dm_pels_height: u32,
        dm_display_flags: u32,
        dm_display_frequency: u32,
        dm_icm_method: u32,
        dm_icm_intent: u32,
        dm_media_type: u32,
        dm_dither_type: u32,
        dm_reserved1: u32,
        dm_reserved2: u32,
        dm_panning_width: u32,
        dm_panning_height: u32,
    }

    impl Default for DevModeW {
        fn default() -> Self {
            Self {
                dm_device_name: [0; CCHDEVICENAME],
                dm_spec_version: 0,
                dm_driver_version: 0,
                dm_size: std::mem::size_of::<DevModeW>() as u16,
                dm_driver_extra: 0,
                dm_fields: 0,
                dm_position_x: 0,
                dm_position_y: 0,
                dm_display_orientation: 0,
                dm_display_fixed_output: 0,
                dm_color: 0,
                dm_duplex: 0,
                dm_y_resolution: 0,
                dm_tt_option: 0,
                dm_collate: 0,
                dm_form_name: [0; CCHFORMNAME],
                dm_log_pixels: 0,
                dm_bits_per_pel: 0,
                dm_pels_width: 0,
                dm_pels_height: 0,
                dm_display_flags: 0,
                dm_display_frequency: 0,
                dm_icm_method: 0,
                dm_icm_intent: 0,
                dm_media_type: 0,
                dm_dither_type: 0,
                dm_reserved1: 0,
                dm_reserved2: 0,
                dm_panning_width: 0,
                dm_panning_height: 0,
            }
        }
    }

    #[link(name = "user32")]
    extern "system" {
        #[link_name = "EnumDisplaySettingsW"]
        fn enum_display_settings_w(
            device_name: *const u16,
            mode_num: u32,
            dev_mode: *mut DevModeW,
        ) -> i32;
        #[link_name = "ChangeDisplaySettingsW"]
        fn change_display_settings_w(dev_mode: *mut DevModeW, flags: u32) -> i32;
    }

    let display_mode_json = |enum_ok: bool, mode: &DevModeW| -> String {
        format!(
            "{{\"enum_ok\":{},\"width\":{},\"height\":{},\"bits_per_pel\":{},\"frequency\":{}}}",
            bool_json(enum_ok),
            mode.dm_pels_width,
            mode.dm_pels_height,
            mode.dm_bits_per_pel,
            mode.dm_display_frequency
        )
    };
    let mut current = DevModeW::default();
    let enum_current =
        unsafe { enum_display_settings_w(std::ptr::null(), ENUM_CURRENT_SETTINGS, &mut current) }
            != 0;
    let reset = params_bool(params, "reset");
    if reset {
        let apply = unsafe { change_display_settings_w(std::ptr::null_mut(), 0) };
        let mut after = DevModeW::default();
        let enum_after =
            unsafe { enum_display_settings_w(std::ptr::null(), ENUM_CURRENT_SETTINGS, &mut after) }
                != 0;
        return Ok(format!(
            concat!(
                "{{\"reset\":true,\"before\":{},",
                "\"apply_result\":{},",
                "\"after\":{}}}"
            ),
            display_mode_json(enum_current, &current),
            apply,
            display_mode_json(enum_after, &after)
        ));
    }

    let width = params
        .get("width")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1024);
    let height = params
        .get("height")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(768);
    let bits_per_pel = params
        .get("bits")
        .or_else(|| params.get("bits_per_pel"))
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(32);
    let frequency = params
        .get("frequency")
        .or_else(|| params.get("hz"))
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(60);

    let mut requested = if enum_current {
        current
    } else {
        DevModeW::default()
    };
    requested.dm_size = std::mem::size_of::<DevModeW>() as u16;
    requested.dm_fields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
    requested.dm_pels_width = width;
    requested.dm_pels_height = height;
    requested.dm_bits_per_pel = bits_per_pel;
    requested.dm_display_frequency = frequency;

    let test = unsafe { change_display_settings_w(&mut requested, CDS_TEST) };
    let apply = if test == 0 {
        Some(unsafe { change_display_settings_w(&mut requested, CDS_UPDATEREGISTRY) })
    } else {
        None
    };
    let mut after = DevModeW::default();
    let enum_after =
        unsafe { enum_display_settings_w(std::ptr::null(), ENUM_CURRENT_SETTINGS, &mut after) }
            != 0;
    Ok(format!(
        concat!(
            "{{\"reset\":false,\"requested\":{{\"width\":{},\"height\":{},\"bits_per_pel\":{},\"frequency\":{}}},",
            "\"before\":{},\"test_result\":{},\"apply_result\":{},\"after\":{}}}"
        ),
        width,
        height,
        bits_per_pel,
        frequency,
        display_mode_json(enum_current, &current),
        test,
        apply
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        display_mode_json(enum_after, &after)
    ))
}

fn daemon_restart_into_next(config: &DaemonConfig) -> Result<String, String> {
    let (current, target, next) = daemon_exe_paths()?;
    if !next.exists() {
        return Err(format!("missing next daemon exe: {}", next.display()));
    }
    let work_dir = target
        .parent()
        .ok_or_else(|| format!("target exe has no parent: {}", target.display()))?;
    let args = vec![
        "daemon".to_string(),
        "--bind".to_string(),
        config.bind.clone(),
        "--dynout".to_string(),
        config.dynout.display().to_string(),
        "--probe-root".to_string(),
        config.probe_root.display().to_string(),
        "--package-root".to_string(),
        config.package_root.display().to_string(),
    ]
    .iter()
    .map(|arg| ps_quote(arg))
    .collect::<Vec<_>>()
    .join(",");
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Continue';",
            "$next={};$target={};$work={};",
            "Start-Sleep -Milliseconds 900;",
            "$promoted=$false;",
            "for($i=0;$i -lt 30;$i++){{",
            "try{{Copy-Item -LiteralPath $next -Destination $target -Force;$promoted=$true;break}}",
            "catch{{Start-Sleep -Milliseconds 250}}",
            "}}",
            "if($promoted){{",
            "Start-Process -FilePath $target -ArgumentList @({}) -WorkingDirectory $work -WindowStyle Minimized",
            "}}"
        ),
        ps_quote_path(&next),
        ps_quote_path(&target),
        ps_quote_path(work_dir),
        args
    );
    ProcessCommand::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-Command",
            &ps,
        ])
        .spawn()
        .map_err(|err| format!("spawn daemon restart helper: {err}"))?;
    append_daemon_event(
        config,
        "restart_into_next",
        &format!(
            "current={} target={} next={}",
            current.display(),
            target.display(),
            next.display()
        ),
    );
    thread::spawn(|| {
        thread::sleep(Duration::from_millis(1000));
        std::process::exit(0);
    });
    Ok(format!(
        "{{\"scheduled\":true,\"pid\":{},\"current\":\"{}\",\"target\":\"{}\",\"next\":\"{}\",\"delay_ms\":1000}}",
        std::process::id(),
        json_escape(&current.display().to_string()),
        json_escape(&target.display().to_string()),
        json_escape(&next.display().to_string())
    ))
}

fn daemon_state_json(config: &DaemonConfig) -> String {
    let status = DaemonStatus::from_dynout(&config.dynout);
    let scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
    let process = read_key_value_tsv(&config.dynout.join("probe_process.tsv"));
    let process_status = daemon_process_status_json_cached(config);
    let capture_process = capture_process_view(config, &process_status);
    let truth = read_key_value_tsv(&config.dynout.join("probe_stage_truth.tsv"));
    let gate = read_key_value_tsv(&config.dynout.join("autoprobe_gate_state.tsv"));
    let latest_pose = read_latest_tsv_row(&config.dynout.join("probe_pose.tsv"));
    let live_snapshot_path = config.dynout.join("probe_live_state.json");
    let live_snapshot_file = file_status(&live_snapshot_path, 2000);
    let live_snapshot = if live_snapshot_file.fresh {
        fs::read_to_string(&live_snapshot_path).ok()
    } else {
        None
    };
    let stage_file = file_status(&config.dynout.join("probe_stage.tsv"), 2000);
    let truth_file = file_status(&config.dynout.join("probe_stage_truth.tsv"), 2000);
    let pose_file = file_status(&config.dynout.join("probe_pose.tsv"), 2000);
    let owner_file = file_status(&config.dynout.join("probe_owner.tsv"), 2000);
    let collision_file = file_status(&config.dynout.join("probe_collision_face.tsv"), 2000);
    let stop_flag = file_status(&config.dynout.join("probe_stop.flag"), 2000);
    let manifest_seen = config.dynout.join("probe_manifest.tsv").exists();
    let hooks_rows = tsv_data_row_count(&config.dynout.join("probe_hooks.tsv"));
    let capture_state = if stop_flag.exists {
        "stopping_or_stopped"
    } else if scenario.contains_key("scenario") && manifest_seen {
        "running"
    } else if scenario.contains_key("scenario") {
        "preparing"
    } else {
        "idle"
    };
    let snapshot_phase = live_snapshot
        .as_deref()
        .and_then(|text| json_extract_string_after(text, "\"live\":", "\"phase\":\""));
    let snapshot_stage_class = live_snapshot
        .as_deref()
        .and_then(|text| json_extract_string_after(text, "\"live\":", "\"stage_class\":\""));
    let snapshot_control_allowed = live_snapshot
        .as_deref()
        .and_then(|text| json_extract_bool_after(text, "\"live\":", "\"control_allowed\":"));
    let snapshot_control_reason = live_snapshot
        .as_deref()
        .and_then(|text| json_extract_string_after(text, "\"live\":", "\"control_reason\":\""));
    let snapshot_selected_phase = live_snapshot.as_deref().and_then(|text| {
        let stage_class =
            json_extract_string_after(text, "\"live\":", "\"stage_class\":\"").unwrap_or_default();
        if stage_class == "game" {
            json_extract_number_after(text, "\"stage\":", "\"selected_phase\":")
        } else {
            None
        }
    });
    let snapshot_selected_step = live_snapshot.as_deref().and_then(|text| {
        let stage_class =
            json_extract_string_after(text, "\"live\":", "\"stage_class\":\"").unwrap_or_default();
        if stage_class == "game" {
            json_extract_number_after(text, "\"stage\":", "\"selected_step\":")
        } else {
            None
        }
    });

    let raw_live_fresh =
        live_snapshot_file.fresh || stage_file.fresh || pose_file.fresh || collision_file.fresh;
    let raw_truth_current = if live_snapshot_file.fresh {
        true
    } else {
        stage_file.fresh && truth_file.exists
    };
    let live_fresh = capture_process.live_kart_present && raw_live_fresh;
    let truth_current = capture_process.live_kart_present && raw_truth_current;
    let control_allowed = if live_snapshot_file.fresh {
        snapshot_control_allowed.unwrap_or(false)
    } else {
        truth
            .get("control_allowed")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            && truth_current
    } && capture_process.live_kart_present;
    let control_reason = if !capture_process.live_kart_present {
        "no_live_kart_process".to_string()
    } else if !truth_current {
        "stale_or_missing_stage_input".to_string()
    } else if let Some(reason) = snapshot_control_reason {
        reason
    } else {
        truth
            .get("control_reason")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    };
    let live_source = if live_snapshot_file.fresh {
        "dll_snapshot"
    } else {
        "dynout_bridge"
    };

    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"daemon\":{{\"name\":\"kart-autoprobe\",\"api\":\"kr-live-v1\",\"bind\":\"{}\",\"dynout\":\"{}\",\"probe_root\":\"{}\",\"package_root\":\"{}\"}},",
            "\"process\":{{\"pid\":\"{}\",\"exe\":\"{}\",\"responding\":\"{}\",\"start_time\":\"{}\"}},",
            "\"capture\":{{\"state\":\"{}\",\"scenario\":\"{}\",\"expected_map\":\"{}\",\"expected_map_display_name\":\"{}\",\"injected_pid\":\"{}\",\"stop_requested\":{},\"manifest_seen\":{},\"hooks_rows\":{},\"process\":{}}},",
            "\"process_live\":{},",
            "\"probe\":{{\"manifest_seen\":{},\"hooks_rows\":{},\"live_snapshot_file\":{},\"stage_file\":{},\"truth_file\":{},\"owner_file\":{},\"pose_file\":{},\"collision_file\":{}}},",
            "\"live\":{{\"source\":\"{}\",\"fresh\":{},\"truth_current\":{},\"phase\":\"{}\",\"stage_class\":\"{}\",\"stage_name\":\"{}\",\"selected_phase\":\"{}\",\"selected_step\":\"{}\",\"control_allowed\":{},\"control_reason\":\"{}\"}},",
            "\"kart\":{{\"owner_ready\":{},\"pose_ready\":{},\"pose\":\"{}\"}},",
            "\"control\":{},",
            "\"experiment\":{},",
            "\"legacy_status\":{}",
            "}}"
        ),
        json_escape(&config.bind),
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string()),
        json_escape(process.get("pid").map(String::as_str).unwrap_or("")),
        json_escape(process.get("executable_path").map(String::as_str).unwrap_or("")),
        json_escape(process.get("responding").map(String::as_str).unwrap_or("")),
        json_escape(process.get("start_time").map(String::as_str).unwrap_or("")),
        capture_state,
        json_escape(scenario.get("scenario").map(String::as_str).unwrap_or("")),
        json_escape(scenario.get("expected_map").map(String::as_str).unwrap_or("")),
        json_escape(scenario.get("expected_map_display_name").map(String::as_str).unwrap_or("")),
        json_escape(scenario.get("pid_at_inject").map(String::as_str).unwrap_or("")),
        bool_json(stop_flag.exists),
        bool_json(manifest_seen),
        hooks_rows,
        capture_process.to_json(),
        process_status,
        bool_json(manifest_seen),
        hooks_rows,
        live_snapshot_file.to_json(),
        stage_file.to_json(),
        truth_file.to_json(),
        owner_file.to_json(),
        pose_file.to_json(),
        collision_file.to_json(),
        live_source,
        bool_json(live_fresh),
        bool_json(truth_current),
        json_escape(if capture_process.live_kart_present {
            snapshot_phase.as_deref().or_else(|| truth.get("phase").map(String::as_str)).or_else(|| gate.get("stage_phase").map(String::as_str)).unwrap_or("")
        } else {
            "no_process"
        }),
        json_escape(if capture_process.live_kart_present {
            snapshot_stage_class.as_deref().or_else(|| truth.get("class").map(String::as_str)).or_else(|| gate.get("stage_class").map(String::as_str)).unwrap_or("")
        } else {
            "none"
        }),
        json_escape(truth.get("stage_name").map(String::as_str).or_else(|| gate.get("stage_name").map(String::as_str)).unwrap_or("")),
        json_escape(snapshot_selected_phase.as_deref().or_else(|| truth.get("selected_phase").map(String::as_str)).unwrap_or("")),
        json_escape(snapshot_selected_step.as_deref().or_else(|| truth.get("selected_step").map(String::as_str)).or_else(|| gate.get("selected_step").map(String::as_str)).unwrap_or("")),
        bool_json(control_allowed),
        json_escape(&control_reason),
        bool_json(owner_file.exists && status.owner_rows),
        bool_json(pose_file.exists && status.pose_rows),
        json_escape(&latest_pose_summary(&latest_pose)),
        daemon_control_json(config),
        daemon_experiment_json(config),
        status.to_json()
    )
}

fn resolve_live_map_identity(
    snapshot_id: &str,
    snapshot_name: &str,
    context_id: &str,
    context_name: &str,
    context_source: &str,
    expected_id: &str,
    expected_name: &str,
) -> (String, String, String, String) {
    if !snapshot_id.is_empty() {
        return (
            snapshot_id.to_string(),
            snapshot_name.to_string(),
            "live_snapshot".to_string(),
            "runtime".to_string(),
        );
    }
    if !context_id.is_empty()
        && (context_source.starts_with("runtime_file_open_")
            || context_source == "runtime_pose_to_kap_strong")
    {
        return (
            context_id.to_string(),
            context_name.to_string(),
            context_source.to_string(),
            "runtime".to_string(),
        );
    }
    if !context_id.is_empty() && context_source == "operator_confirmed" {
        return (
            context_id.to_string(),
            context_name.to_string(),
            context_source.to_string(),
            "operator_confirmed".to_string(),
        );
    }
    // expected_id and a declared map context describe operator intent, not an
    // observation from the game process. Keep them in map.expected/context_id,
    // but never project them as the resolved live map.
    let _ = (expected_id, expected_name);
    (
        String::new(),
        String::new(),
        String::new(),
        "unknown".to_string(),
    )
}

fn confirm_live_map(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let map_id = params
        .get("map_id")
        .map(String::as_str)
        .unwrap_or("")
        .trim();
    if map_id.len() < 3
        || map_id.eq_ignore_ascii_case("unknown")
        || !map_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err("invalid map_id; expected a registry id such as fairy_I03".to_string());
    }
    let process = read_key_value_tsv(&config.dynout.join("probe_process.tsv"));
    let pid = process.get("pid").map(String::as_str).unwrap_or("").trim();
    if pid.is_empty() {
        return Err("capture pid unavailable; start capture before confirming a map".to_string());
    }
    let map_name = params
        .get("map_name")
        .map(String::as_str)
        .unwrap_or("")
        .trim();
    if map_name.contains(['\r', '\n', '\t']) {
        return Err("invalid map_name".to_string());
    }
    let state_dir = config.probe_root.join("state");
    fs::create_dir_all(&state_dir).map_err(|error| format!("create map state dir: {error}"))?;
    let path = state_dir.join("probe_map_context.tsv");
    let previous = read_key_value_tsv(&path);
    let previous_id = previous.get("map_id").map(String::as_str).unwrap_or("");
    let previous_name = previous.get("map_name").map(String::as_str).unwrap_or("");
    let body = format!(
        "key\tvalue\r\npid\t{}\r\nmap_id\t{}\r\nmap_name\t{}\r\nsource\toperator_confirmed\r\n",
        pid, map_id, map_name
    );
    fs::write(&path, body).map_err(|error| format!("write {}: {error}", path.display()))?;
    let history_path = config.dynout.join("probe_map_confirmation.tsv");
    let new_history = !history_path.exists();
    let mut history = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&history_path)
        .map_err(|error| format!("open {}: {error}", history_path.display()))?;
    if new_history {
        history
            .write_all(
                b"time_ms\tpid\tmap_id\tmap_name\tprevious_map_id\tprevious_map_name\tsource\r\n",
            )
            .map_err(|error| format!("write history header: {error}"))?;
    }
    history
        .write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\toperator_confirmed\r\n",
                daemon_time_ms(),
                pid,
                map_id,
                map_name,
                previous_id,
                previous_name
            )
            .as_bytes(),
        )
        .map_err(|error| format!("append map confirmation history: {error}"))?;
    Ok(format!(
        "{{\"ok\":true,\"map_id\":\"{}\",\"map_name\":\"{}\",\"source\":\"operator_confirmed\",\"confidence\":\"operator_confirmed\",\"pid\":\"{}\"}}",
        json_escape(map_id), json_escape(map_name), json_escape(pid)
    ))
}

fn daemon_live_state_json(config: &DaemonConfig) -> String {
    let live = current_live_state(config);
    let status = DaemonStatus::from_dynout(&config.dynout);
    let scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
    let process = read_key_value_tsv(&config.dynout.join("probe_process.tsv"));
    let process_status = daemon_process_status_json_cached(config);
    let capture_process = capture_process_view(config, &process_status);
    let gate = read_key_value_tsv(&config.dynout.join("autoprobe_gate_state.tsv"));
    let latest_pose = read_latest_tsv_row(&config.dynout.join("probe_pose.tsv"));
    let latest_owner = read_latest_tsv_row(&config.dynout.join("probe_owner.tsv"));
    let wall_context = daemon_wall_context_json(config);
    let experiment_path = config.dynout.join("probe_experiment.json");
    let experiment_text = fs::read_to_string(&experiment_path).unwrap_or_default();
    let run_id =
        json_extract_string_after(&experiment_text, "", "\"run_id\":\"").unwrap_or_default();
    let experiment_map =
        json_extract_string_after(&experiment_text, "", "\"expected_map\":\"").unwrap_or_default();
    let expected_map = scenario
        .get("expected_map")
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&experiment_map);
    let expected_map_display_name = scenario
        .get("expected_map_display_name")
        .map(String::as_str)
        .unwrap_or("");
    let live_snapshot_path = config.dynout.join("probe_live_state.json");
    let live_snapshot_file = file_status(&live_snapshot_path, 2000);
    let live_snapshot_text = if live_snapshot_file.fresh {
        fs::read_to_string(&live_snapshot_path).ok()
    } else {
        None
    };
    let snapshot_map_id = live_snapshot_text
        .as_deref()
        .and_then(|text| json_extract_string_after(text, "\"map\":", "\"id\":\""))
        .unwrap_or_default();
    let snapshot_map_name = live_snapshot_text
        .as_deref()
        .and_then(|text| json_extract_string_after(text, "\"map\":", "\"name\":\""))
        .unwrap_or_default();
    let map_context = read_key_value_tsv(
        &config
            .probe_root
            .join("state")
            .join("probe_map_context.tsv"),
    );
    let context_pid_matches = map_context
        .get("pid")
        .zip(process.get("pid"))
        .map(|(context_pid, capture_pid)| context_pid == capture_pid)
        .unwrap_or(false);
    let context_map_id = if context_pid_matches {
        map_context.get("map_id").cloned().unwrap_or_default()
    } else {
        String::new()
    };
    let context_map_name = if context_pid_matches {
        map_context.get("map_name").cloned().unwrap_or_default()
    } else {
        String::new()
    };
    let context_map_source = if context_pid_matches {
        map_context.get("source").cloned().unwrap_or_default()
    } else {
        String::new()
    };
    let (resolved_map_id, resolved_map_name, resolved_map_source, resolved_map_confidence) =
        resolve_live_map_identity(
            &snapshot_map_id,
            &snapshot_map_name,
            &context_map_id,
            &context_map_name,
            &context_map_source,
            expected_map,
            expected_map_display_name,
        );

    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"schema\":\"kr_live_state_v1\",",
            "\"daemon\":{{\"name\":\"kart-autoprobe\",\"api\":\"kr-live-v1\",\"bind\":\"{}\",\"dynout\":\"{}\",\"probe_root\":\"{}\",\"package_root\":\"{}\"}},",
            "\"process\":{{\"pid\":\"{}\",\"exe\":\"{}\",\"responding\":\"{}\",\"start_time\":\"{}\"}},",
            "\"process_live\":{},",
            "\"capture_process\":{},",
            "\"game\":{{\"phase\":\"{}\",\"stage_class\":\"{}\",\"stage_name\":\"{}\",\"selected_phase\":\"{}\",\"selected_step\":\"{}\",\"truth_current\":{},\"source\":\"{}\",\"fresh\":{}}},",
            "\"control\":{{\"allowed\":{},\"reason\":\"{}\",\"evidence\":{}}},",
            "\"map\":{{\"expected\":\"{}\",\"expected_display_name\":\"{}\",\"snapshot_id\":\"{}\",\"snapshot_name\":\"{}\",\"context_id\":\"{}\",\"context_name\":\"{}\",\"context_source\":\"{}\",\"context_pid_matches\":{},\"resolved_id\":\"{}\",\"resolved_name\":\"{}\",\"resolved_source\":\"{}\",\"resolved_confidence\":\"{}\"}},",
            "\"self\":{},",
            "\"wall\":{},",
            "\"kart\":{{\"owner_ready\":{},\"pose_ready\":{},\"ground_ready\":{},\"collision_ready\":{},\"latest_pose\":\"{}\",\"owner\":{},\"pose\":{}}},",
            "\"probe\":{{\"files\":{{\"live_snapshot\":{},\"stage\":{},\"truth\":{},\"owner\":{},\"pose\":{},\"ground\":{},\"wall\":{},\"wall_hard\":{},\"collision\":{},\"events\":{}}},\"hooks_rows\":{},\"manifest_seen\":{}}},",
            "\"experiment\":{},",
            "\"package\":{},",
            "\"legacy_status\":{}",
            "}}"
        ),
        json_escape(&config.bind),
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string()),
        json_escape(process.get("pid").map(String::as_str).unwrap_or("")),
        json_escape(process.get("executable_path").map(String::as_str).unwrap_or("")),
        json_escape(process.get("responding").map(String::as_str).unwrap_or("")),
        json_escape(process.get("start_time").map(String::as_str).unwrap_or("")),
        process_status,
        capture_process.to_json(),
        json_escape(&live.phase),
        json_escape(&live.stage_class),
        json_escape(gate.get("stage_name").map(String::as_str).unwrap_or("")),
        json_escape(&live.selected_phase),
        json_escape(&live.selected_step),
        bool_json(live.truth_current),
        json_escape(&live.source),
        bool_json(live.fresh),
        bool_json(live.control_allowed),
        json_escape(&live.control_reason),
        daemon_control_json(config),
        json_escape(expected_map),
        json_escape(expected_map_display_name),
        json_escape(&snapshot_map_id),
        json_escape(&snapshot_map_name),
        json_escape(&context_map_id),
        json_escape(&context_map_name),
        json_escape(&context_map_source),
        bool_json(context_pid_matches),
        json_escape(&resolved_map_id),
        json_escape(&resolved_map_name),
        json_escape(&resolved_map_source),
        json_escape(&resolved_map_confidence),
        daemon_self_state_json(config),
        wall_context,
        bool_json(status.owner_rows),
        bool_json(status.pose_rows),
        bool_json(status.ground_rows),
        bool_json(status.collision_rows),
        json_escape(&latest_pose_summary(&latest_pose)),
        params_object_json(&latest_owner),
        params_object_json(&latest_pose),
        live_snapshot_file.to_json(),
        file_status(&config.dynout.join("probe_stage.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_stage_truth.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_owner.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_pose.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_ground_drive.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_wall.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_wall_hard.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_collision_face.tsv"), 2000).to_json(),
        file_status(&config.dynout.join("probe_event.tsv"), 5000).to_json(),
        tsv_data_row_count(&config.dynout.join("probe_hooks.tsv")),
        bool_json(config.dynout.join("probe_manifest.tsv").exists()),
        daemon_experiment_json(config),
        package_for_run_status_json(config, &run_id),
        status.to_json()
    )
}

fn row_value_json(row: &BTreeMap<String, String>, key: &str) -> String {
    row.get(key)
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn row_vec3_json(row: &BTreeMap<String, String>, prefix: &str) -> String {
    format!(
        "{{\"x\":{},\"y\":{},\"z\":{}}}",
        row_value_json(row, &format!("{}_x", prefix)),
        row_value_json(row, &format!("{}_y", prefix)),
        row_value_json(row, &format!("{}_z", prefix))
    )
}

fn wall_kind_label(kind: &str) -> &'static str {
    match kind {
        "1" => "sweep",
        "2" => "exit",
        "3" => "hard_pre",
        "4" => "hard_velocity_post",
        "5" => "hard_angular_post",
        "6" => "soft_bucket_post",
        "7" => "soft_charge_post",
        "8" => "soft_velocity_add",
        "9" => "soft_angular_add0",
        "10" => "soft_angular_add1",
        _ => "unknown",
    }
}

fn wall_row_json(source: &str, row: &BTreeMap<String, String>) -> String {
    let kind = row.get("kind").map(String::as_str).unwrap_or("");
    if row.is_empty() {
        return "{\"present\":false}".to_string();
    }
    format!(
        concat!(
            "{{",
            "\"present\":true,",
            "\"source\":\"{}\",",
            "\"kind\":\"{}\",",
            "\"kind_label\":\"{}\",",
            "\"site_rva\":{},",
            "\"tick_ms\":{},",
            "\"this\":{},",
            "\"result\":{},",
            "\"wall_result\":{},",
            "\"loop_index\":{},",
            "\"bucket_flag\":{},",
            "\"bucket_depth\":{},",
            "\"hit_normal\":{},",
            "\"hit_dot_vel\":{},",
            "\"pre_vel\":{},",
            "\"post_vel\":{},",
            "\"source_vec\":{},",
            "\"pre_ang\":{},",
            "\"post_ang\":{}",
            "}}"
        ),
        json_escape(source),
        json_escape(kind),
        wall_kind_label(kind),
        row_value_json(row, "site_rva"),
        row_value_json(row, "tick_ms"),
        row_value_json(row, "this"),
        row_value_json(row, "result"),
        row_value_json(row, "wall_result"),
        row_value_json(row, "loop_index"),
        row_value_json(row, "bucket_flag"),
        row_value_json(row, "bucket_depth"),
        row_vec3_json(row, "hit_n"),
        row_value_json(row, "hit_dot_vel"),
        row_vec3_json(row, "pre_vel"),
        row_vec3_json(row, "post_vel"),
        row_vec3_json(row, "source"),
        row_vec3_json(row, "pre_ang"),
        row_vec3_json(row, "post_ang")
    )
}

fn daemon_wall_context_json(config: &DaemonConfig) -> String {
    let wall_path = config.dynout.join("probe_wall.tsv");
    let hard_path = config.dynout.join("probe_wall_hard.tsv");
    let collision_path = config.dynout.join("probe_collision_face.tsv");
    let wall_file = file_status(&wall_path, 2000);
    let hard_file = file_status(&hard_path, 2000);
    let collision_file = file_status(&collision_path, 2000);
    let wall_row = read_latest_tsv_row_fast(&wall_path);
    let hard_row = read_latest_tsv_row_fast(&hard_path);
    let collision_row = read_latest_tsv_row_fast(&collision_path);
    let selected_source = match (
        row_tick_ms(&wall_row),
        row_tick_ms(&hard_row),
        wall_file.fresh,
        hard_file.fresh,
    ) {
        (Some(wall_tick), Some(hard_tick), true, true) if hard_tick >= wall_tick => {
            "probe_wall_hard.tsv"
        }
        (Some(_), Some(_), true, true) => "probe_wall.tsv",
        (Some(_), _, true, _) => "probe_wall.tsv",
        (_, Some(_), _, true) => "probe_wall_hard.tsv",
        (_, Some(_), _, _) => "probe_wall_hard.tsv",
        (Some(_), _, _, _) => "probe_wall.tsv",
        _ => "",
    };
    let selected_row = if selected_source == "probe_wall_hard.tsv" {
        &hard_row
    } else {
        &wall_row
    };
    format!(
        concat!(
            "{{",
            "\"fresh\":{},",
            "\"source\":\"{}\",",
            "\"selected\":{},",
            "\"latest_sweep\":{},",
            "\"latest_response\":{},",
            "\"latest_collision\":{{\"file\":{},\"row\":{}}},",
            "\"files\":{{\"wall\":{},\"wall_hard\":{},\"collision\":{}}}",
            "}}"
        ),
        bool_json(wall_file.fresh || hard_file.fresh || collision_file.fresh),
        json_escape(selected_source),
        wall_row_json(selected_source, selected_row),
        wall_row_json("probe_wall.tsv", &wall_row),
        wall_row_json("probe_wall_hard.tsv", &hard_row),
        collision_file.to_json(),
        params_object_json(&collision_row),
        wall_file.to_json(),
        hard_file.to_json(),
        collision_file.to_json()
    )
}

fn daemon_control_json(config: &DaemonConfig) -> String {
    let path = config.dynout.join("probe_control_evidence.tsv");
    let file = file_status(&path, 5000);
    let row = read_latest_tsv_row(&path);
    format!(
        concat!(
            "{{",
            "\"evidence_file\":{},",
            "\"last\":{{",
            "\"time_ms\":\"{}\",",
            "\"op\":\"{}\",",
            "\"keys\":\"{}\",",
            "\"duration_ms\":\"{}\",",
            "\"focus\":\"{}\",",
            "\"before_phase\":\"{}\",",
            "\"before_control_allowed\":\"{}\",",
            "\"before_pose\":\"{}\",",
            "\"after_phase\":\"{}\",",
            "\"after_control_allowed\":\"{}\",",
            "\"after_pose\":\"{}\",",
            "\"before_self\":\"{}\",",
            "\"after_self\":\"{}\",",
            "\"self_delta\":\"{}\",",
            "\"result\":\"{}\"",
            "}}",
            "}}"
        ),
        file.to_json(),
        json_escape(row.get("time_ms").map(String::as_str).unwrap_or("")),
        json_escape(row.get("op").map(String::as_str).unwrap_or("")),
        json_escape(row.get("keys").map(String::as_str).unwrap_or("")),
        json_escape(row.get("duration_ms").map(String::as_str).unwrap_or("")),
        json_escape(row.get("focus").map(String::as_str).unwrap_or("")),
        json_escape(row.get("before_phase").map(String::as_str).unwrap_or("")),
        json_escape(
            row.get("before_control_allowed")
                .map(String::as_str)
                .unwrap_or("")
        ),
        json_escape(row.get("before_pose").map(String::as_str).unwrap_or("")),
        json_escape(row.get("after_phase").map(String::as_str).unwrap_or("")),
        json_escape(
            row.get("after_control_allowed")
                .map(String::as_str)
                .unwrap_or("")
        ),
        json_escape(row.get("after_pose").map(String::as_str).unwrap_or("")),
        json_escape(row.get("before_self").map(String::as_str).unwrap_or("")),
        json_escape(row.get("after_self").map(String::as_str).unwrap_or("")),
        json_escape(row.get("self_delta").map(String::as_str).unwrap_or("")),
        json_escape(row.get("result").map(String::as_str).unwrap_or(""))
    )
}

#[derive(Clone, Debug)]
struct SelfStateView {
    owner_file: FileStatus,
    active_file: FileStatus,
    pose_file: FileStatus,
    state_file: FileStatus,
    motion_file: FileStatus,
    item_resource_file: FileStatus,
    owner: BTreeMap<String, String>,
    active: BTreeMap<String, String>,
    pose: BTreeMap<String, String>,
    state: BTreeMap<String, String>,
    motion: BTreeMap<String, String>,
    item_resource: BTreeMap<String, String>,
    items_json: String,
    observed: bool,
    fresh: bool,
    confidence: String,
    warnings: Vec<String>,
    identity_this: String,
    identity_candidate: String,
    identity_candidate_score: String,
    position_source: String,
    velocity_source: String,
    projection_source: String,
    x: Option<String>,
    y: Option<String>,
    z: Option<String>,
    yaw: Option<String>,
    vel_x: Option<String>,
    vel_y: Option<String>,
    vel_z: Option<String>,
    speed: Option<String>,
    proj_row0: Option<String>,
    proj_row1: Option<String>,
    proj_row2: Option<String>,
    proj_col0: Option<String>,
    proj_col1: Option<String>,
    proj_col2: Option<String>,
    state_raw: Option<String>,
    state_timer_raw: Option<String>,
    surface_mode: Option<String>,
    b1c: Option<String>,
    rail_state: Option<String>,
    bcharge_active: Option<String>,
    pit_active: Option<String>,
    pit_accum: Option<String>,
    pit_rate: Option<String>,
    pit_mode: Option<String>,
}

impl SelfStateView {
    fn speed_f64(&self) -> Option<f64> {
        self.speed.as_deref()?.parse::<f64>().ok()
    }

    fn forward_speed_f64(&self) -> Option<f64> {
        // Probe pose matrices map local axes to world columns. The validated
        // driving convention is forward = -local Y = -column-1 projection.
        self.proj_col1
            .as_deref()?
            .parse::<f64>()
            .ok()
            .map(|value| -value)
    }

    fn drive_gate_reason(&self, max_speed: f64) -> Option<String> {
        if !self.fresh {
            return Some(format!("self_not_fresh:confidence={}", self.confidence));
        }
        if self.identity_this.is_empty() {
            return Some("missing_self_identity".to_string());
        }
        if self.warnings.iter().any(|warning| {
            warning.starts_with("invalid_owner") || warning.starts_with("invalid_active")
        }) {
            return Some(format!("invalid_self_state:{}", self.warnings.join(",")));
        }
        if self.x.is_none() || self.y.is_none() || self.z.is_none() {
            return Some("missing_self_position".to_string());
        }
        if self.vel_x.is_none() || self.vel_y.is_none() || self.vel_z.is_none() {
            return Some("missing_self_velocity".to_string());
        }
        let Some(speed) = self.speed_f64() else {
            return Some("missing_self_speed".to_string());
        };
        if speed.abs() > max_speed {
            return Some(format!(
                "self_not_stationary:speed={speed:.6}>max={max_speed:.6}"
            ));
        }
        None
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema\":\"kr_self_state_v1\",",
                "\"observed\":{},",
                "\"fresh\":{},",
                "\"confidence\":\"{}\",",
                "\"warnings\":[{}],",
                "\"sources\":{{\"position\":\"{}\",\"velocity\":\"{}\"}},",
                "\"files\":{{\"owner\":{},\"active\":{},\"pose\":{},\"state\":{},\"motion\":{},\"item_resource\":{}}},",
                "\"identity\":{{\"this\":\"{}\",\"candidate\":\"{}\",\"candidate_score\":\"{}\"}},",
                "\"pose\":{{\"x\":{},\"y\":{},\"z\":{},\"yaw\":{}}},",
                "\"velocity\":{{\"x\":{},\"y\":{},\"z\":{},\"speed\":{},\"matrix_projection\":{{\"source\":\"{}\",\"row0\":{},\"row1\":{},\"row2\":{},\"col0\":{},\"col1\":{},\"col2\":{}}}}},",
                "\"physics\":{{\"surface_mode\":{},\"state_raw\":{},\"state_timer_raw\":{},\"b1c\":{},\"rail_state\":{}}},",
                "\"resources\":{{\"bcharge_active\":{},\"pit_active\":{},\"pit_accum\":{},\"pit_rate\":{},\"pit_mode\":{},\"nitro_level\":null,\"items\":{}}},",
                "\"motion\":{{\"source\":\"{}\",\"samples\":{},\"max_pos_delta\":{},\"max_speed\":{}}},",
                "\"raw\":{{\"owner\":{},\"active\":{},\"pose\":{},\"state\":{},\"motion\":{},\"item_resource\":{}}}",
                "}}"
            ),
            bool_json(self.observed),
            bool_json(self.fresh),
            json_escape(&self.confidence),
            string_array_json(&self.warnings),
            json_escape(&self.position_source),
            json_escape(&self.velocity_source),
            self.owner_file.to_json(),
            self.active_file.to_json(),
            self.pose_file.to_json(),
            self.state_file.to_json(),
            self.motion_file.to_json(),
            self.item_resource_file.to_json(),
            json_escape(&self.identity_this),
            json_escape(&self.identity_candidate),
            json_escape(&self.identity_candidate_score),
            json_number_or_null(self.x.as_deref()),
            json_number_or_null(self.y.as_deref()),
            json_number_or_null(self.z.as_deref()),
            json_number_or_null(self.yaw.as_deref()),
            json_number_or_null(self.vel_x.as_deref()),
            json_number_or_null(self.vel_y.as_deref()),
            json_number_or_null(self.vel_z.as_deref()),
            json_number_or_null(self.speed.as_deref()),
            json_escape(&self.projection_source),
            json_number_or_null(self.proj_row0.as_deref()),
            json_number_or_null(self.proj_row1.as_deref()),
            json_number_or_null(self.proj_row2.as_deref()),
            json_number_or_null(self.proj_col0.as_deref()),
            json_number_or_null(self.proj_col1.as_deref()),
            json_number_or_null(self.proj_col2.as_deref()),
            json_string_or_null(self.surface_mode.as_deref()),
            json_string_or_null(self.state_raw.as_deref()),
            json_string_or_null(self.state_timer_raw.as_deref()),
            json_string_or_null(self.b1c.as_deref()),
            json_string_or_null(self.rail_state.as_deref()),
            json_string_or_null(self.bcharge_active.as_deref()),
            json_string_or_null(self.pit_active.as_deref()),
            json_number_or_null(self.pit_accum.as_deref()),
            json_number_or_null(self.pit_rate.as_deref()),
            json_string_or_null(self.pit_mode.as_deref()),
            self.items_json,
            json_escape(self.motion.get("source").map(String::as_str).unwrap_or("")),
            json_number_or_null(self.motion.get("samples").map(String::as_str)),
            json_number_or_null(self.motion.get("max_pos_delta").map(String::as_str)),
            json_number_or_null(self.motion.get("max_speed").map(String::as_str)),
            params_object_json(&self.owner),
            params_object_json(&self.active),
            params_object_json(&self.pose),
            params_object_json(&self.state),
            params_object_json(&self.motion),
            params_object_json(&self.item_resource)
        )
    }

    fn summary(&self) -> String {
        format!(
            "observed={} fresh={} confidence={} pos=({},{},{}) yaw={} vel=({},{},{}) speed={} proj=row({},{},{}) col({},{},{}) surface={} state={} b1c={} rail={} bcharge={} pit={} warnings={}",
            bool_json(self.observed),
            bool_json(self.fresh),
            self.confidence,
            self.x.as_deref().unwrap_or(""),
            self.y.as_deref().unwrap_or(""),
            self.z.as_deref().unwrap_or(""),
            self.yaw.as_deref().unwrap_or(""),
            self.vel_x.as_deref().unwrap_or(""),
            self.vel_y.as_deref().unwrap_or(""),
            self.vel_z.as_deref().unwrap_or(""),
            self.speed.as_deref().unwrap_or(""),
            self.proj_row0.as_deref().unwrap_or(""),
            self.proj_row1.as_deref().unwrap_or(""),
            self.proj_row2.as_deref().unwrap_or(""),
            self.proj_col0.as_deref().unwrap_or(""),
            self.proj_col1.as_deref().unwrap_or(""),
            self.proj_col2.as_deref().unwrap_or(""),
            self.surface_mode.as_deref().unwrap_or(""),
            self.state_raw.as_deref().unwrap_or(""),
            self.b1c.as_deref().unwrap_or(""),
            self.rail_state.as_deref().unwrap_or(""),
            self.bcharge_active.as_deref().unwrap_or(""),
            self.pit_active.as_deref().unwrap_or(""),
            self.warnings.join(",")
        )
    }

    fn delta_from(&self, before: &SelfStateView) -> String {
        let dpos = option_distance(
            before.x.as_deref(),
            before.y.as_deref(),
            before.z.as_deref(),
            self.x.as_deref(),
            self.y.as_deref(),
            self.z.as_deref(),
        );
        let dspeed = option_delta(before.speed.as_deref(), self.speed.as_deref());
        let drow0 = option_delta(before.proj_row0.as_deref(), self.proj_row0.as_deref());
        let drow1 = option_delta(before.proj_row1.as_deref(), self.proj_row1.as_deref());
        let drow2 = option_delta(before.proj_row2.as_deref(), self.proj_row2.as_deref());
        let dcol0 = option_delta(before.proj_col0.as_deref(), self.proj_col0.as_deref());
        let dcol1 = option_delta(before.proj_col1.as_deref(), self.proj_col1.as_deref());
        let dcol2 = option_delta(before.proj_col2.as_deref(), self.proj_col2.as_deref());
        format!(
            "dpos={} dspeed={} dproj=row({},{},{}) col({},{},{}) surface={}->{} state={}->{} b1c={}->{} rail={}->{} bcharge={}->{} pit={}->{}",
            dpos.unwrap_or_default(),
            dspeed.unwrap_or_default(),
            drow0.unwrap_or_default(),
            drow1.unwrap_or_default(),
            drow2.unwrap_or_default(),
            dcol0.unwrap_or_default(),
            dcol1.unwrap_or_default(),
            dcol2.unwrap_or_default(),
            before.surface_mode.as_deref().unwrap_or(""),
            self.surface_mode.as_deref().unwrap_or(""),
            before.state_raw.as_deref().unwrap_or(""),
            self.state_raw.as_deref().unwrap_or(""),
            before.b1c.as_deref().unwrap_or(""),
            self.b1c.as_deref().unwrap_or(""),
            before.rail_state.as_deref().unwrap_or(""),
            self.rail_state.as_deref().unwrap_or(""),
            before.bcharge_active.as_deref().unwrap_or(""),
            self.bcharge_active.as_deref().unwrap_or(""),
            before.pit_active.as_deref().unwrap_or(""),
            self.pit_active.as_deref().unwrap_or("")
        )
    }
}

fn item_resource_items_json(file: &FileStatus, row: &BTreeMap<String, String>) -> String {
    if !file.fresh || row.get("phase").map(String::as_str) != Some("1") {
        return "null".to_string();
    }
    let Some(count) = row
        .get("entry_count")
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return "null".to_string();
    };
    if count > 2 {
        return "null".to_string();
    }
    let mut items = Vec::new();
    for index in 0..count {
        let item_key = format!("item{index}");
        let quantity_key = format!("quantity{index}");
        let Some(item_id) = row
            .get(&item_key)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            return "null".to_string();
        };
        let Some(quantity) = row
            .get(&quantity_key)
            .and_then(|value| value.parse::<i32>().ok())
        else {
            return "null".to_string();
        };
        if item_id != 0 && quantity > 0 {
            items.push(format!("{{\"item_id\":{item_id},\"quantity\":{quantity}}}"));
        }
    }
    format!("[{}]", items.join(","))
}

fn current_self_state(config: &DaemonConfig) -> SelfStateView {
    let owner_path = config.dynout.join("probe_owner.tsv");
    let active_path = config.dynout.join("probe_active.tsv");
    let pose_path = config.dynout.join("probe_pose.tsv");
    let state_path = config.dynout.join("probe_state.tsv");
    let surface_path = config.dynout.join("probe_surface.tsv");
    let phase_surface_path = config.dynout.join("probe_phase_surface.tsv");
    let motion_path = config.dynout.join("probe_motion.tsv");
    let item_resource_path = config.dynout.join("probe_item_resource.tsv");
    let live_snapshot_path = config.dynout.join("probe_live_state.json");
    let owner_file = file_status(&owner_path, 2000);
    let active_file = file_status(&active_path, 2000);
    let pose_file = file_status(&pose_path, 2000);
    let state_file = file_status(&state_path, 2000);
    let surface_file = file_status(&surface_path, 2000);
    let phase_surface_file = file_status(&phase_surface_path, 2000);
    let motion_file = file_status(&motion_path, 5000);
    let item_resource_file = file_status(&item_resource_path, 5000);
    let live_snapshot_file = file_status(&live_snapshot_path, 2000);
    let live_snapshot = if live_snapshot_file.fresh {
        fs::read_to_string(&live_snapshot_path)
            .ok()
            .map(|text| live_snapshot_self_fields(&text))
            .unwrap_or_default()
    } else {
        BTreeMap::new()
    };
    let active = read_latest_tsv_row_fast(&active_path);
    let active_this = if active_row_is_invalid(&active) {
        None
    } else {
        active.get("this").map(String::as_str)
    };
    let owner = if let Some(this) = active_this {
        read_latest_tsv_row_matching_fast(&owner_path, "this", this).unwrap_or_default()
    } else {
        read_latest_tsv_row_fast(&owner_path)
    };
    let pose = if let Some(this) = active_this {
        read_latest_tsv_row_matching_fast(&pose_path, "this", this).unwrap_or_default()
    } else {
        read_latest_tsv_row_fast(&pose_path)
    };
    let state = read_latest_tsv_row_fast(&state_path);
    let surface = if let Some(this) = active_this {
        read_latest_tsv_row_matching_fast(&surface_path, "this", this).unwrap_or_default()
    } else {
        read_latest_tsv_row_fast(&surface_path)
    };
    let phase_surface = if let Some(this) = active_this {
        read_latest_tsv_row_matching_fast(&phase_surface_path, "this", this).unwrap_or_default()
    } else {
        read_latest_tsv_row_fast(&phase_surface_path)
    };
    let motion = read_latest_tsv_row_fast(&motion_path);
    let item_resource = read_latest_tsv_row_fast(&item_resource_path);
    let items_json = item_resource_items_json(&item_resource_file, &item_resource);

    let mut warnings = Vec::new();
    let active_for_state = if active_row_is_invalid(&active) {
        warnings.push(format!(
            "invalid_active:{}",
            active.get("reason").map(String::as_str).unwrap_or("")
        ));
        BTreeMap::new()
    } else {
        active.clone()
    };
    let owner_for_state = if owner_row_has_zero_pose_velocity(&owner) {
        warnings.push("invalid_owner_plausibility".to_string());
        BTreeMap::new()
    } else {
        owner.clone()
    };
    let legacy_identity_this = active_for_state
        .get("this")
        .or_else(|| owner_for_state.get("this"))
        .map(String::as_str)
        .unwrap_or("")
        .to_string();
    let snapshot_pose_this = live_snapshot
        .get("pose_this")
        .map(String::as_str)
        .unwrap_or("");
    let snapshot_owner_this = live_snapshot
        .get("owner_this")
        .map(String::as_str)
        .unwrap_or("");
    // A live snapshot is self-contained only when its selected owner and pose
    // refer to the same vehicle.  It may otherwise straddle a room restart:
    // the owner selector updates before the first pose hook for the new kart.
    let snapshot_identity_coherent = pointer_identity_eq(snapshot_pose_this, snapshot_owner_this);
    let identity_this = if legacy_identity_this.is_empty() && snapshot_identity_coherent {
        snapshot_pose_this.to_string()
    } else {
        legacy_identity_this
    };
    let reference_tick_ms =
        latest_row_tick_ms(&[&active_for_state, &owner_for_state, &pose, &state]);
    let surface_for_state =
        fresh_row_near_tick(&surface_file, &surface, reference_tick_ms, 1500).unwrap_or_default();
    let phase_surface_for_state =
        fresh_row_near_tick(&phase_surface_file, &phase_surface, reference_tick_ms, 1500)
            .unwrap_or_default();
    let identity_candidate = owner_for_state
        .get("candidate")
        .map(String::as_str)
        .unwrap_or("")
        .to_string();
    let identity_candidate_score = owner_for_state
        .get("candidate_score")
        .map(String::as_str)
        .unwrap_or("")
        .to_string();
    let (legacy_x, legacy_position_source) = fresh_row_field(
        &active_file,
        &active_for_state,
        &["pos_x"],
        &pose_file,
        &pose,
        &["post_pos_x", "x"],
        &owner_file,
        &owner_for_state,
        &["pos_x"],
        "position",
    );
    let legacy_y = matching_source_field(
        &legacy_position_source,
        &active_for_state,
        &["pos_y"],
        &pose,
        &["post_pos_y", "y"],
        &owner_for_state,
        &["pos_y"],
    );
    let legacy_z = matching_source_field(
        &legacy_position_source,
        &active_for_state,
        &["pos_z"],
        &pose,
        &["post_pos_z", "z"],
        &owner_for_state,
        &["pos_z"],
    );
    let legacy_yaw = matching_source_field(
        &legacy_position_source,
        &active_for_state,
        &[],
        &pose,
        &["post_ang_z", "yaw"],
        &owner_for_state,
        &["ang_z"],
    );
    let (legacy_vel_x, legacy_velocity_source) = fresh_row_field(
        &active_file,
        &active_for_state,
        &["vel_x"],
        &pose_file,
        &pose,
        &["vel_x"],
        &owner_file,
        &owner_for_state,
        &["vel_x"],
        "velocity",
    );
    let legacy_vel_y = matching_source_field(
        &legacy_velocity_source,
        &active_for_state,
        &["vel_y"],
        &pose,
        &["vel_y"],
        &owner_for_state,
        &["vel_y"],
    );
    let legacy_vel_z = matching_source_field(
        &legacy_velocity_source,
        &active_for_state,
        &["vel_z"],
        &pose,
        &["vel_z"],
        &owner_for_state,
        &["vel_z"],
    );
    let snapshot_pose_fields_ready = ["x", "y", "z", "vel_x", "vel_y", "vel_z"]
        .iter()
        .all(|key| live_snapshot.contains_key(*key));
    let snapshot_pose_identity_matches =
        pointer_identity_eq(snapshot_pose_this, identity_this.as_str());
    let snapshot_pose_ready = snapshot_pose_fields_ready && snapshot_pose_identity_matches;
    if snapshot_pose_fields_ready && !snapshot_pose_identity_matches {
        warnings.push(format!(
            "snapshot_pose_identity_mismatch:{}!={}",
            snapshot_pose_this, identity_this
        ));
    }
    let snapshot_physics_ready = pointer_identity_eq(snapshot_owner_this, identity_this.as_str());
    if !live_snapshot.is_empty() && !snapshot_physics_ready {
        warnings.push(format!(
            "snapshot_owner_identity_mismatch:{}!={}",
            snapshot_owner_this, identity_this
        ));
    }
    let position_source = if snapshot_pose_ready {
        "dll_snapshot".to_string()
    } else {
        legacy_position_source
    };
    let velocity_source = if snapshot_pose_ready {
        "dll_snapshot".to_string()
    } else {
        legacy_velocity_source
    };
    let snapshot_pose_field = |key: &str| {
        if snapshot_pose_ready {
            live_snapshot.get(key).cloned()
        } else {
            None
        }
    };
    let raw_x = snapshot_pose_field("x").or(legacy_x);
    let raw_y = snapshot_pose_field("y").or(legacy_y);
    let raw_z = snapshot_pose_field("z").or(legacy_z);
    let raw_yaw = snapshot_pose_field("yaw").or(legacy_yaw);
    let raw_vel_x = snapshot_pose_field("vel_x").or(legacy_vel_x);
    let raw_vel_y = snapshot_pose_field("vel_y").or(legacy_vel_y);
    let raw_vel_z = snapshot_pose_field("vel_z").or(legacy_vel_z);
    let x = valid_f64_abs_field(raw_x, 1_000_000.0, "pos_x", &mut warnings);
    let y = valid_f64_abs_field(raw_y, 1_000_000.0, "pos_y", &mut warnings);
    let z = valid_f64_abs_field(raw_z, 1_000_000.0, "pos_z", &mut warnings);
    let yaw = valid_f64_abs_field(raw_yaw, 100_000.0, "yaw", &mut warnings);
    let vel_x = valid_f64_abs_field(raw_vel_x, 100_000.0, "vel_x", &mut warnings);
    let vel_y = valid_f64_abs_field(raw_vel_y, 100_000.0, "vel_y", &mut warnings);
    let vel_z = valid_f64_abs_field(raw_vel_z, 100_000.0, "vel_z", &mut warnings);
    let speed = valid_f64_abs_field(
        snapshot_pose_field("speed").or_else(|| {
            self_speed_value(
                active_file.fresh,
                pose_file.fresh,
                owner_file.fresh,
                &active_for_state,
                &pose,
                &owner_for_state,
                vel_x.as_deref(),
                vel_y.as_deref(),
                vel_z.as_deref(),
            )
        }),
        100_000.0,
        "speed",
        &mut warnings,
    );
    let (proj_row0, proj_row1, proj_row2, proj_col0, proj_col1, proj_col2) =
        if position_source == "pose" && velocity_source == "pose" && pose_file.fresh {
            matrix_velocity_projection(&pose, vel_x.as_deref(), vel_y.as_deref(), vel_z.as_deref())
        } else {
            (None, None, None, None, None, None)
        };
    let projection_source = if proj_row0.is_some() || proj_col0.is_some() {
        "pose_matrix_candidate"
    } else if pose_file.fresh {
        "missing"
    } else {
        "stale_or_missing_pose"
    };
    let state_raw = valid_u32_field(
        first_field(&[(&owner_for_state, "state_raw"), (&state, "state_raw")]),
        255,
        "state_raw",
        &mut warnings,
    );
    let state_timer_raw = first_field(&[
        (&owner_for_state, "state_timer_raw"),
        (&state, "state_timer_raw"),
    ]);
    let surface_mode = valid_u32_field(
        (if snapshot_physics_ready {
            live_snapshot.get("surface_mode").cloned()
        } else {
            None
        })
        .or_else(|| {
            first_field(&[
                (&owner_for_state, "surface_mode"),
                (&state, "surface_mode"),
                (&phase_surface_for_state, "surface_mode"),
                (&surface_for_state, "new_surface_mode"),
                (&surface_for_state, "surface_mode"),
            ])
        }),
        8,
        "surface_mode",
        &mut warnings,
    );
    let b1c = valid_u32_field(
        (if snapshot_physics_ready {
            live_snapshot.get("b1c").cloned()
        } else {
            None
        })
        .or_else(|| {
            first_field(&[
                (&owner_for_state, "b1c"),
                (&state, "b1c"),
                (&phase_surface_for_state, "b1c"),
                (&surface_for_state, "b1c"),
            ])
        }),
        1,
        "b1c",
        &mut warnings,
    );
    let rail_state = valid_u32_field(
        (if snapshot_physics_ready {
            live_snapshot.get("rail_state").cloned()
        } else {
            None
        })
        .or_else(|| {
            first_field(&[
                (&owner_for_state, "rail_state"),
                (&state, "rail_state"),
                (&phase_surface_for_state, "rail_state"),
                (&surface_for_state, "rail_state"),
            ])
        }),
        16,
        "rail_state",
        &mut warnings,
    );
    let bcharge_active = valid_u32_field(
        first_field(&[
            (&owner_for_state, "bcharge_active"),
            (&state, "bcharge_active"),
        ]),
        1,
        "bcharge_active",
        &mut warnings,
    );
    let pit_active = valid_u32_field(
        first_field(&[(&owner_for_state, "pit_active"), (&state, "pit_active")]),
        1,
        "pit_active",
        &mut warnings,
    );
    let pit_accum = first_field(&[(&owner_for_state, "pit_accum"), (&state, "pit_accum")]);
    let pit_rate = first_field(&[(&owner_for_state, "pit_rate"), (&state, "pit_rate")]);
    let pit_mode = first_field(&[(&owner_for_state, "pit_mode"), (&state, "pit_mode")]);
    let active_present = !active_for_state.is_empty();
    let owner_present = !owner_for_state.is_empty();
    let pose_present = !pose.is_empty();
    let confidence = if snapshot_pose_ready {
        "dll_snapshot_fresh"
    } else if active_file.fresh && active_present && pose_file.fresh && pose_present {
        "active_pose_fresh"
    } else if active_file.fresh && active_present {
        "active_fresh"
    } else if owner_file.fresh && owner_present && pose_file.fresh && pose_present {
        "owner_pose_fresh"
    } else if owner_file.fresh && owner_present {
        "owner_fresh"
    } else if pose_file.fresh && pose_present {
        "pose_fresh"
    } else if (active_file.exists && !active.is_empty())
        || (owner_file.exists && !owner.is_empty())
        || (pose_file.exists && pose_present)
    {
        "stale"
    } else {
        "missing"
    };
    let observed = live_snapshot_file.exists
        || owner_file.exists
        || active_file.exists
        || pose_file.exists
        || state_file.exists;
    let fresh = snapshot_pose_ready
        || (active_file.fresh && active_present)
        || (owner_file.fresh && owner_present)
        || (pose_file.fresh && pose_present)
        || state_file.fresh;
    SelfStateView {
        owner_file,
        active_file,
        pose_file,
        state_file,
        motion_file,
        item_resource_file,
        owner,
        active,
        pose,
        state,
        motion,
        item_resource,
        items_json,
        observed,
        fresh,
        confidence: confidence.to_string(),
        warnings,
        identity_this,
        identity_candidate,
        identity_candidate_score,
        position_source: position_source.to_string(),
        velocity_source: velocity_source.to_string(),
        projection_source: projection_source.to_string(),
        x,
        y,
        z,
        yaw,
        vel_x,
        vel_y,
        vel_z,
        speed,
        proj_row0,
        proj_row1,
        proj_row2,
        proj_col0,
        proj_col1,
        proj_col2,
        state_raw,
        state_timer_raw,
        surface_mode,
        b1c,
        rail_state,
        bcharge_active,
        pit_active,
        pit_accum,
        pit_rate,
        pit_mode,
    }
}

fn daemon_self_state_json(config: &DaemonConfig) -> String {
    current_self_state(config).to_json()
}

fn daemon_kap_route_nearest_json(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let live = current_live_state(config);
    if !live.fresh || !live.truth_current || !live.control_allowed {
        return Err(format!(
            "live_not_plannable:fresh={} truth_current={} control_allowed={} phase={} reason={}",
            live.fresh, live.truth_current, live.control_allowed, live.phase, live.control_reason
        ));
    }
    let self_state = current_self_state(config);
    // Explicit XYZ is a read-only route query and remains useful when the
    // interactive desktop is disconnected and self telemetry is stale.  Do
    // not require fresh runtime pose just to project an operator-supplied
    // world point; live lifecycle truth is still required above.
    let explicit_xyz = match (params.get("x"), params.get("y"), params.get("z")) {
        (Some(x), Some(y), Some(z)) => Some([
            x.parse::<f32>()
                .map_err(|_| "x must be a number".to_string())?,
            y.parse::<f32>()
                .map_err(|_| "y must be a number".to_string())?,
            z.parse::<f32>()
                .map_err(|_| "z must be a number".to_string())?,
        ]),
        (None, None, None) => None,
        _ => return Err("x, y, and z must be supplied together".to_string()),
    };
    let pose = if let Some(pos) = explicit_xyz {
        Pose {
            pos,
            yaw: 0.0,
            speed: 0.0,
            velocity_heading: None,
            tick_ms: None,
        }
    } else {
        self_pose_for_planning(&self_state)?
    };
    let kap_dir = params
        .get("kap_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_daemon_kap_dir(config));
    let track_id = params
        .get("kap_track")
        .or_else(|| params.get("track"))
        .cloned()
        .or_else(|| {
            let scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
            scenario.get("expected_map").cloned()
        })
        .filter(|value| !value.trim().is_empty() && value != "unknown")
        .ok_or("kap route planning requires kap_track or scenario expected_map")?;
    let slot = params
        .get("kap_slot")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "kap_slot must be an integer".to_string())
        })
        .transpose()?;
    let request = KapFollowRequest {
        kap_dir: Some(kap_dir.clone()),
        track_id: Some(track_id.clone()),
        slot,
        ..Default::default()
    };
    let paths = load_kap_follow_paths(&request)?;
    let path = choose_kap_follow_path(&paths, pose, &request)
        .ok_or_else(|| "no KAP path could project current self pose".to_string())?;
    let cursor_s = params
        .get("cursor_s")
        .or_else(|| params.get("path_s"))
        .map(|value| {
            value
                .parse::<f32>()
                .map_err(|_| "cursor_s must be a number".to_string())
        })
        .transpose()?;
    let cursor_window_m = params
        .get("cursor_window_m")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(90.0)
        .max(1.0);
    let backward_tolerance_m = params
        .get("backward_tolerance_m")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(10.0)
        .max(0.0);
    let (projection, projection_mode) = if let Some(cursor_s) = cursor_s {
        let projection = path
            .project_near_forward(pose.pos, cursor_s, cursor_window_m, backward_tolerance_m)
            .ok_or_else(|| format!(
                "kap_continuity_gate_failed:path={} cursor_s={} window_m={} backward_tolerance_m={}",
                path.path_id, cursor_s, cursor_window_m, backward_tolerance_m
            ))?;
        (projection, "cursor_near_forward")
    } else {
        let projection = path
            .project(pose.pos)
            .ok_or_else(|| format!("KAP path {} cannot project self pose", path.path_id))?;
        (projection, "global_nearest")
    };
    let lookahead = params
        .get("lookahead")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(20.0);
    let target = path
        .point_at(projection.path_s + lookahead)
        .ok_or_else(|| format!("KAP path {} cannot compute lookahead", path.path_id))?;
    let horizon_m = params
        .get("horizon_m")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(220.0);
    let min_distance_ahead = params
        .get("min_distance_ahead")
        .or_else(|| params.get("min_ahead_m"))
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let selected_window_json = "null";
    let candidate_count = 0;

    Ok(format!(
        concat!(
            "{{",
            "\"schema\":\"kr_kap_route_nearest_v1\",",
            "\"track_id\":\"{}\",",
            "\"kap_dir\":\"{}\",",
            "\"live\":{},",
            "\"self\":{},",
            "\"path\":{{\"id\":\"{}\",\"slot\":{},\"points\":{},\"length\":{},\"closed\":{}}},",
            "\"projection\":{{\"path_s\":{},\"progress\":{},\"cross_track\":{},\"mode\":\"{}\",\"cursor_s\":{},\"cursor_window_m\":{},\"backward_tolerance_m\":{}}},",
            "\"lookahead\":{{\"distance\":{},\"target\":{{\"x\":{},\"y\":{},\"z\":{}}}}},",
            "\"window_filter\":{{\"horizon_m\":{},\"min_distance_ahead\":{}}},",
            "\"selected_window\":{},",
            "\"candidate_count\":{}",
            "}}"
        ),
        json_escape(&track_id),
        json_escape(&kap_dir.display().to_string()),
        live.to_json(),
        self_state.to_json(),
        json_escape(&path.path_id),
        path.slot
            .map(|slot| slot.to_string())
            .unwrap_or_else(|| "null".to_string()),
        path.points.len(),
        path.length,
        bool_json(path.closed),
        projection.path_s,
        projection.path_s / path.length.max(f32::EPSILON),
        projection.cross_track,
        projection_mode,
        cursor_s.map(|value| value.to_string()).unwrap_or_else(|| "null".to_string()),
        cursor_window_m,
        backward_tolerance_m,
        lookahead,
        target[0],
        target[1],
        target[2],
        horizon_m,
        min_distance_ahead,
        selected_window_json,
        candidate_count
    ))
}

fn self_pose_for_planning(self_state: &SelfStateView) -> Result<Pose, String> {
    if !self_state.fresh {
        return Err(format!(
            "self_not_fresh:confidence={}",
            self_state.confidence
        ));
    }
    let x = parse_self_f32(self_state.x.as_deref(), "x")?;
    let y = parse_self_f32(self_state.y.as_deref(), "y")?;
    let z = parse_self_f32(self_state.z.as_deref(), "z")?;
    let scalar_yaw = self_state
        .yaw
        .as_deref()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let yaw = owner_forward_yaw(&self_state.owner).unwrap_or(scalar_yaw);
    let speed = self_state
        .speed
        .as_deref()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let velocity_heading = match (
        self_state
            .vel_x
            .as_deref()
            .and_then(|value| value.parse::<f32>().ok()),
        self_state
            .vel_y
            .as_deref()
            .and_then(|value| value.parse::<f32>().ok()),
    ) {
        (Some(vx), Some(vy)) if (vx * vx + vy * vy).sqrt() >= 0.05 => Some(vy.atan2(vx)),
        _ => None,
    };
    Ok(Pose {
        pos: [x, y, z],
        yaw,
        speed,
        velocity_heading,
        tick_ms: None,
    })
}

fn owner_forward_yaw(owner: &BTreeMap<String, String>) -> Option<f32> {
    // The validated vehicle convention is forward = -local Y.  Matrix columns
    // map local axes into world space, so the XY forward vector is -column 1.
    // ang_z is an angular-state field and can be zero while the kart faces pi.
    let m01 = owner.get("m01")?.parse::<f32>().ok()?;
    let m11 = owner.get("m11")?.parse::<f32>().ok()?;
    let forward_x = -m01;
    let forward_y = -m11;
    ((forward_x * forward_x + forward_y * forward_y).sqrt() >= 0.5)
        .then(|| forward_y.atan2(forward_x))
}

#[derive(Clone, Debug)]
struct KapNearestTrack {
    track_id: String,
    path_id: String,
    slot: Option<usize>,
    cross_track: f32,
    path_s: f32,
}

fn pre_drive_kap_pose_gate(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
    self_state: &SelfStateView,
) -> Result<String, String> {
    let explicit_kap_track = params
        .get("kap_track")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let requested_track = params
        .get("kap_track")
        .or_else(|| params.get("track"))
        .or_else(|| params.get("expected_map"))
        .map(String::as_str)
        .unwrap_or("")
        .trim();
    if requested_track.is_empty() || requested_track == "unknown" {
        return Ok("{\"enabled\":false,\"reason\":\"missing_requested_track\"}".to_string());
    }
    let pose = self_pose_for_planning(self_state)?;
    let kap_dir = params
        .get("kap_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_daemon_kap_dir(config));
    let requested_slot = params
        .get("kap_slot")
        .and_then(|value| value.parse::<usize>().ok());
    let requested_start_s = params
        .get("start_path_s")
        .and_then(|value| value.parse::<f32>().ok());
    let requested_start_window = params
        .get("start_window_m")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(80.0);
    let requested = nearest_kap_track_in_dir(
        &kap_dir,
        pose,
        Some(requested_track),
        requested_slot,
        requested_start_s,
        requested_start_window,
    )?
    .ok_or_else(|| format!("no requested KAP path for track={requested_track}"))?;
    let global = nearest_kap_track_in_dir(&kap_dir, pose, None, None, None, 0.0)?
        .ok_or_else(|| format!("no KAP path in {}", kap_dir.display()))?;
    let max_requested_cross_track = params
        .get("pre_drive_max_kap_cross_track")
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(20.0);
    let better_margin = params
        .get("pre_drive_kap_pose_gate_margin")
        .or_else(|| params.get("pre_drive_kap_map_gate_margin"))
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(2.0);
    let better_ratio = params
        .get("pre_drive_kap_pose_gate_ratio")
        .or_else(|| params.get("pre_drive_kap_map_gate_ratio"))
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.70);
    let same_track = global.track_id == requested.track_id;
    let global_clearly_better = !explicit_kap_track
        && !same_track
        && global.cross_track + better_margin < requested.cross_track
        && global.cross_track <= requested.cross_track * better_ratio;
    let requested_too_far = requested.cross_track > max_requested_cross_track;
    let body = format!(
        concat!(
            "{{",
            "\"enabled\":true,",
            "\"requested_track\":\"{}\",",
            "\"requested_slot\":{},",
            "\"requested_start_path_s\":{},",
            "\"requested_start_window_m\":{},",
            "\"requested\":{},",
            "\"global_best\":{},",
            "\"max_requested_cross_track\":{},",
            "\"better_margin\":{},",
            "\"better_ratio\":{},",
            "\"same_track\":{},",
            "\"explicit_kap_track\":{},",
            "\"global_clearly_better\":{},",
            "\"requested_too_far\":{}",
            "}}"
        ),
        json_escape(requested_track),
        requested_slot
            .map(|slot| slot.to_string())
            .unwrap_or_else(|| "null".to_string()),
        requested_start_s
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        requested_start_window,
        kap_nearest_track_json(&requested),
        kap_nearest_track_json(&global),
        max_requested_cross_track,
        better_margin,
        better_ratio,
        bool_json(same_track),
        bool_json(explicit_kap_track),
        bool_json(global_clearly_better),
        bool_json(requested_too_far)
    );
    if global_clearly_better || requested_too_far {
        return Err(format!("kap_pose_gate_failed:{body}"));
    }
    Ok(body)
}

fn nearest_kap_track_in_dir(
    kap_dir: &Path,
    pose: Pose,
    track_filter: Option<&str>,
    slot_filter: Option<usize>,
    cursor_s: Option<f32>,
    cursor_window_m: f32,
) -> Result<Option<KapNearestTrack>, String> {
    let mut best: Option<KapNearestTrack> = None;
    visit_files(kap_dir, &mut |path| {
        if path.extension().and_then(|s| s.to_str()) != Some("kap") {
            return;
        }
        let Some(track_id) = kap_track_id(path) else {
            return;
        };
        if track_filter.is_some_and(|filter| filter != track_id) {
            return;
        }
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
        let path_id = path.display().to_string();
        let slot = kap_slot(path);
        if slot_filter.is_some() && slot != slot_filter {
            return;
        }
        let follow_path = KapFollowPath::new(path_id.clone(), track_id, slot, kap.points);
        let projection = if let Some(cursor_s) = cursor_s {
            follow_path.project_near_strict(pose.pos, cursor_s, cursor_window_m.max(1.0))
        } else {
            follow_path.project(pose.pos)
        };
        let Some(projection) = projection else {
            return;
        };
        let candidate = KapNearestTrack {
            track_id: follow_path.track_id.clone(),
            path_id,
            slot,
            cross_track: projection.cross_track,
            path_s: projection.path_s,
        };
        let replace = best
            .as_ref()
            .map(|old| candidate.cross_track < old.cross_track)
            .unwrap_or(true);
        if replace {
            best = Some(candidate);
        }
    })?;
    Ok(best)
}

fn kap_nearest_track_json(nearest: &KapNearestTrack) -> String {
    format!(
        concat!(
            "{{",
            "\"track_id\":\"{}\",",
            "\"path_id\":\"{}\",",
            "\"slot\":{},",
            "\"cross_track\":{},",
            "\"path_s\":{}",
            "}}"
        ),
        json_escape(&nearest.track_id),
        json_escape(&nearest.path_id),
        nearest
            .slot
            .map(|slot| slot.to_string())
            .unwrap_or_else(|| "null".to_string()),
        nearest.cross_track,
        nearest.path_s
    )
}

fn default_daemon_kap_dir(config: &DaemonConfig) -> PathBuf {
    config
        .probe_root
        .parent()
        .map(|parent| parent.join("autoprobe").join("aiPath"))
        .unwrap_or_else(|| config.probe_root.join("autoprobe").join("aiPath"))
}

fn parse_self_f32(value: Option<&str>, name: &str) -> Result<f32, String> {
    value
        .ok_or_else(|| format!("missing_self_{name}"))?
        .parse::<f32>()
        .map_err(|_| format!("invalid_self_{name}"))
}

#[allow(dead_code)]
fn kap_route_window_json(
    window: &KapRouteWindow,
    path: &KapFollowPath,
    current_s: f32,
    horizon_m: f32,
) -> String {
    let distance_ahead = path.forward_delta_bounded(current_s, window.start_s, horizon_m);
    format!(
        concat!(
            "{{",
            "\"path_id\":\"{}\",",
            "\"slot\":{},",
            "\"start_s\":{},\"end_s\":{},\"center_s\":{},",
            "\"distance_ahead\":{},",
            "\"progress_start\":{},\"progress_end\":{},",
            "\"length_m\":{},",
            "\"heading_change_rad\":{},",
            "\"max_segment_heading_delta_rad\":{},",
            "\"straightness\":{},",
            "\"grade_abs_max\":{},",
            "\"score\":{},",
            "\"kind\":\"{}\",",
            "\"verdict\":\"{}\",",
            "\"start\":{{\"x\":{},\"y\":{},\"z\":{}}},",
            "\"end\":{{\"x\":{},\"y\":{},\"z\":{}}}",
            "}}"
        ),
        json_escape(&window.path_id),
        window
            .slot
            .map(|slot| slot.to_string())
            .unwrap_or_else(|| "null".to_string()),
        window.start_s,
        window.end_s,
        window.center_s,
        distance_ahead,
        window.progress_start,
        window.progress_end,
        window.length_m,
        window.heading_change_rad,
        window.max_segment_heading_delta_rad,
        window.straightness,
        window.grade_abs_max,
        window.score,
        window.kind,
        window.verdict,
        window.start[0],
        window.start[1],
        window.start[2],
        window.end[0],
        window.end[1],
        window.end[2]
    )
}

fn current_ui_blockers(process_windows_json: &str) -> Vec<String> {
    let mut blockers = Vec::new();
    if process_windows_json.contains("Direct3D Failure") {
        blockers.push("direct3d_failure_modal".to_string());
    }
    blockers
}

fn effective_d3d9_ready(
    probe_ready: bool,
    kart_present: bool,
    live_fresh: bool,
    truth_current: bool,
    phase: &str,
) -> (bool, bool) {
    let runtime_override = kart_present
        && live_fresh
        && truth_current
        && !phase.is_empty()
        && !matches!(phase, "no_stage" | "unknown");
    (probe_ready || runtime_override, runtime_override)
}

fn daemon_experiment_json(config: &DaemonConfig) -> String {
    let path = config.dynout.join("probe_experiment.json");
    let file = file_status(&path, 5000);
    let text = if file.exists {
        fs::read_to_string(&path).unwrap_or_default()
    } else {
        String::new()
    };
    let step_count = text.matches("\"step\":\"").count();
    format!(
        concat!(
            "{{",
            "\"manifest_file\":{},",
            "\"schema\":\"{}\",",
            "\"run_id\":\"{}\",",
            "\"status\":\"{}\",",
            "\"scenario\":\"{}\",",
            "\"expected_map\":\"{}\",",
            "\"step_count\":{},",
            "\"last_step\":\"{}\"",
            "}}"
        ),
        file.to_json(),
        json_escape(&json_extract_string_after(&text, "", "\"schema\":\"").unwrap_or_default()),
        json_escape(&json_extract_string_after(&text, "", "\"run_id\":\"").unwrap_or_default()),
        json_escape(&json_extract_string_after(&text, "", "\"status\":\"").unwrap_or_default()),
        json_escape(&json_extract_string_after(&text, "", "\"scenario\":\"").unwrap_or_default()),
        json_escape(
            &json_extract_string_after(&text, "", "\"expected_map\":\"").unwrap_or_default()
        ),
        step_count,
        json_escape(&json_extract_last_string(&text, "\"step\":\"").unwrap_or_default())
    )
}

fn package_for_run_status_json(config: &DaemonConfig, run_id: &str) -> String {
    if run_id.trim().is_empty() {
        return "{\"run_id\":\"\",\"recorded\":false,\"reason\":\"no_run_id\"}".to_string();
    }
    let mut params = BTreeMap::new();
    params.insert("run_id".to_string(), run_id.to_string());
    match package_by_run_json(config, &params) {
        Ok(package) => format!(
            "{{\"run_id\":\"{}\",\"recorded\":true,\"result\":{}}}",
            json_escape(run_id),
            package
        ),
        Err(err) => format!(
            "{{\"run_id\":\"{}\",\"recorded\":false,\"error\":\"{}\"}}",
            json_escape(run_id),
            json_escape(&err)
        ),
    }
}

fn daemon_preflight_json(config: &DaemonConfig) -> String {
    let process_status = daemon_process_status_json_cached(config);
    let process_windows = daemon_process_windows_json();
    let ui_blockers = current_ui_blockers(&process_windows);
    let d3d9 = daemon_d3d9_json();
    let input_security = daemon_input_security_json(config);
    let input_desktop_ready = input_security.contains("\"sendinput_integrity_allowed\":true");
    let d3d9_probe_ready = d3d9.contains("\"ok\":true")
        && d3d9.contains("\"adapter_count\":")
        && !d3d9.contains("\"adapter_count\":0")
        && (d3d9.contains("\"check_x8r8g8b8_windowed\":{\"signed\":0")
            || d3d9.contains("\"check_x8r8g8b8_fullscreen\":{\"signed\":0"));
    let capture_process = capture_process_view(config, &process_status);
    let live_state = daemon_live_state_json(config);
    let experiment = daemon_experiment_json(config);
    let current_status = current_experiment_status(config);
    let experiment_manifest = file_status(&config.dynout.join("probe_experiment.json"), 5000);
    let experiment_active =
        experiment_manifest_blocks_start(experiment_manifest.fresh, &current_status);
    let live = current_live_state(config);
    let prepare_script = config
        .probe_root
        .join("scripts")
        .join("remote_capture_prepare.ps1");
    let package_script = config
        .probe_root
        .join("scripts")
        .join("remote_capture_package.ps1");
    let wait_script = config
        .probe_root
        .join("scripts")
        .join("remote_capture_wait.ps1");
    let default_profile = config.probe_root.join("profiles").join("default.tsv");
    let package_root_status = file_status(&config.package_root, u128::MAX);
    let dynout_status = file_status(&config.dynout, u128::MAX);
    let kart_present = process_status.contains("\"name\":\"KartRider\"")
        && !process_status.contains("\"pid\":\"\"");
    let (d3d9_ready, d3d9_runtime_override) = effective_d3d9_ready(
        d3d9_probe_ready,
        kart_present,
        live.fresh,
        live.truth_current,
        &live.phase,
    );
    let d3d9_ready_source = if d3d9_probe_ready {
        "capability_probe"
    } else if d3d9_runtime_override {
        "live_game_runtime_override"
    } else {
        "unavailable"
    };
    let capture_tooling_ready = prepare_script.exists()
        && package_script.exists()
        && wait_script.exists()
        && default_profile.exists();
    let storage_ready = config.package_root.exists() && config.dynout.exists();
    let mut blockers = Vec::new();
    if !kart_present {
        blockers.push("kart_process_missing".to_string());
    }
    if !d3d9_ready {
        blockers.push("d3d9_hal_not_available".to_string());
    }
    for blocker in &ui_blockers {
        blockers.push(blocker.clone());
    }
    if !prepare_script.exists() {
        blockers.push("missing_remote_capture_prepare".to_string());
    }
    if !package_script.exists() {
        blockers.push("missing_remote_capture_package".to_string());
    }
    if !wait_script.exists() {
        blockers.push("missing_remote_capture_wait".to_string());
    }
    if !default_profile.exists() {
        blockers.push("missing_default_profile".to_string());
    }
    if !config.package_root.exists() {
        blockers.push("missing_package_root".to_string());
    }
    if !config.dynout.exists() {
        blockers.push("missing_dynout".to_string());
    }
    if experiment_active {
        blockers.push(format!("experiment_active:{current_status}"));
    }
    let ready = blockers.is_empty();
    let control_ready = live.fresh && live.truth_current && live.control_allowed;
    let room_wait_ready = live.fresh && live.truth_current && is_room_wait_phase(&live.phase);
    let mut loop_blockers = Vec::new();
    if !ready {
        loop_blockers.push("infrastructure_not_ready".to_string());
    }
    if !(live.fresh && live.truth_current) {
        loop_blockers.push("live_truth_not_fresh".to_string());
    }
    if !input_desktop_ready {
        loop_blockers.push("input_desktop_unavailable".to_string());
    }
    if live.fresh && live.truth_current && !(room_wait_ready || control_ready) {
        loop_blockers.push(format!("not_room_wait_or_control:{}", live.phase));
    }
    if capture_process.pid_mismatch {
        loop_blockers.push(format!(
            "capture_process_restarted:{}->{}",
            capture_process.injected_pid, capture_process.current_pid
        ));
    } else if !capture_process.injected_pid.is_empty()
        && capture_process.live_source == "powershell_live"
        && !capture_process.live_kart_present
    {
        loop_blockers.push(format!(
            "capture_process_dead:{}",
            capture_process.injected_pid
        ));
    }
    let loop_ready = loop_blockers.is_empty();
    let recommended_action = if loop_ready {
        "run_room_loop"
    } else if !input_desktop_ready {
        "reconnect_or_unlock_interactive_desktop"
    } else if !ui_blockers.is_empty() {
        "clear_ui_blocker"
    } else if !d3d9_ready {
        "restart_display_stack"
    } else if capture_process.pid_mismatch
        || (!capture_process.injected_pid.is_empty()
            && capture_process.live_source == "powershell_live"
            && !capture_process.live_kart_present)
    {
        "restart_capture"
    } else if !(live.fresh && live.truth_current) {
        "start_or_restart_capture"
    } else if live.fresh && live.truth_current && !(room_wait_ready || control_ready) {
        "wait_or_enter_room"
    } else if !ready {
        "fix_infrastructure"
    } else {
        "inspect_preflight"
    };
    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"schema\":\"kr_experiment_preflight_v1\",",
            "\"ready\":{},",
            "\"loop_ready\":{},",
            "\"recommended_action\":\"{}\",",
            "\"blockers\":[{}],",
            "\"loop_blockers\":[{}],",
            "\"capabilities\":{{\"process\":{},\"d3d9\":{},\"input_desktop\":{},\"capture_tooling\":{},\"storage\":{},\"live_truth\":{},\"room_wait\":{},\"control\":{},\"capture_process\":{}}},",
            "\"ui_blockers\":[{}],",
            "\"diagnostics\":{{\"d3d9\":{},\"input_security\":{},\"d3d9_probe_ready\":{},\"d3d9_runtime_override\":{},\"d3d9_ready_source\":\"{}\"}},",
            "\"paths\":{{\"dynout\":\"{}\",\"probe_root\":\"{}\",\"package_root\":\"{}\",\"prepare_script\":\"{}\",\"package_script\":\"{}\",\"wait_script\":\"{}\",\"default_profile\":\"{}\"}},",
            "\"files\":{{\"dynout\":{},\"package_root\":{},\"prepare_script\":{},\"package_script\":{},\"wait_script\":{},\"default_profile\":{}}},",
            "\"process\":{},",
            "\"capture_process\":{},",
            "\"live\":{},",
            "\"experiment\":{}",
            "}}"
        ),
        bool_json(ready),
        bool_json(loop_ready),
        json_escape(recommended_action),
        string_array_json(&blockers),
        string_array_json(&loop_blockers),
        bool_json(kart_present),
        bool_json(d3d9_ready),
        bool_json(input_desktop_ready),
        bool_json(capture_tooling_ready),
        bool_json(storage_ready),
        bool_json(live.fresh && live.truth_current),
        bool_json(room_wait_ready),
        bool_json(control_ready),
        bool_json(capture_process.alive),
        string_array_json(&ui_blockers),
        d3d9,
        input_security,
        bool_json(d3d9_probe_ready),
        bool_json(d3d9_runtime_override),
        d3d9_ready_source,
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string()),
        json_escape(&prepare_script.display().to_string()),
        json_escape(&package_script.display().to_string()),
        json_escape(&wait_script.display().to_string()),
        json_escape(&default_profile.display().to_string()),
        dynout_status.to_json(),
        package_root_status.to_json(),
        file_status(&prepare_script, u128::MAX).to_json(),
        file_status(&package_script, u128::MAX).to_json(),
        file_status(&wait_script, u128::MAX).to_json(),
        file_status(&default_profile, u128::MAX).to_json(),
        process_status,
        capture_process.to_json(),
        live_state,
        experiment
    )
}

fn daemon_events_json(config: &DaemonConfig, since_seq: usize, limit: usize) -> String {
    let path = config.dynout.join("probe_event.tsv");
    let Ok(file) = File::open(&path) else {
        return format!(
            "{{\"ok\":true,\"path\":\"{}\",\"events\":[]}}",
            json_escape(&path.display().to_string())
        );
    };
    let mut rows = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate().skip(1) {
        let seq = idx;
        if seq <= since_seq {
            continue;
        }
        if rows.len() >= limit {
            break;
        }
        let line = line.unwrap_or_default();
        let cols: Vec<_> = line.split('\t').collect();
        rows.push(format!(
            "{{\"seq\":{},\"time\":\"{}\",\"event\":\"{}\",\"detail\":\"{}\"}}",
            seq,
            json_escape(cols.first().copied().unwrap_or("")),
            json_escape(cols.get(1).copied().unwrap_or("")),
            json_escape(cols.get(2).copied().unwrap_or(""))
        ));
    }
    format!(
        "{{\"ok\":true,\"path\":\"{}\",\"events\":[{}]}}",
        json_escape(&path.display().to_string()),
        rows.join(",")
    )
}

fn daemon_log_json(config: &DaemonConfig, name: &str, tail: usize) -> String {
    let path = config.dynout.join(name);
    let Ok(file) = File::open(&path) else {
        return format!(
            "{{\"ok\":true,\"path\":\"{}\",\"lines\":[]}}",
            json_escape(&path.display().to_string())
        );
    };
    let mut lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|line| line.unwrap_or_default())
        .collect();
    if lines.len() > tail {
        lines = lines.split_off(lines.len() - tail);
    }
    let body = lines
        .iter()
        .map(|line| format!("\"{}\"", json_escape(line)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"ok\":true,\"path\":\"{}\",\"lines\":[{}]}}",
        json_escape(&path.display().to_string()),
        body
    )
}

fn daemon_rail_coverage_json(config: &DaemonConfig, since_seq: u64) -> String {
    let path = config.dynout.join("probe_rail.tsv");
    let Ok(mut file) = File::open(&path) else {
        return format!(
            "{{\"ok\":true,\"schema\":\"kr_live_rail_coverage_v2\",\"path\":\"{}\",\"since_seq\":{},\"rows\":0,\"last_seq\":0,\"kind_counts\":{{\"20\":0,\"21\":0,\"23\":0}},\"missing_kinds\":[20,21,23],\"tail_truncated\":false}}",
            json_escape(&path.display().to_string()), since_seq
        );
    };
    const TAIL_BYTES: u64 = 4 * 1024 * 1024;
    let len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let start = len.saturating_sub(TAIL_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return "{\"ok\":false,\"error\":\"rail_seek_failed\"}".to_string();
    }
    let mut text = String::new();
    if file.read_to_string(&mut text).is_err() {
        return "{\"ok\":false,\"error\":\"rail_read_failed\"}".to_string();
    }
    if start > 0 {
        if let Some(index) = text.find('\n') {
            text = text[index + 1..].to_string();
        }
    }
    let mut counts = [0_u64; 3];
    let mut rows = 0_u64;
    let mut last_seq = 0_u64;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let seq = fields
            .first()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        last_seq = last_seq.max(seq);
        if seq <= since_seq {
            continue;
        }
        rows += 1;
        if let Some(kind) = fields.get(5).and_then(|value| value.parse::<u32>().ok()) {
            if let Some(index) = [20_u32, 21, 23].iter().position(|value| *value == kind) {
                counts[index] += 1;
            }
        }
    }
    let missing = [20_u32, 21, 23]
        .iter()
        .enumerate()
        .filter_map(|(index, kind)| (counts[index] == 0).then_some(kind.to_string()))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"ok\":true,\"schema\":\"kr_live_rail_coverage_v2\",\"path\":\"{}\",\"since_seq\":{},\"rows\":{},\"last_seq\":{},\"kind_counts\":{{\"20\":{},\"21\":{},\"23\":{}}},\"missing_kinds\":[{}],\"tail_truncated\":{}}}",
        json_escape(&path.display().to_string()), since_seq, rows, last_seq, counts[0], counts[1], counts[2], missing, bool_json(start > 0)
    )
}

fn latest_package_json(config: &DaemonConfig) -> Result<String, String> {
    let path = find_latest_package(config)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_string();
    Ok(format!(
        "{{\"package_root\":\"{}\",\"path\":\"{}\",\"name\":\"{}\",\"file\":{},\"download\":\"/packages/download?name={}\"}}",
        json_escape(&config.package_root.display().to_string()),
        json_escape(&path.display().to_string()),
        json_escape(&name),
        file_status(&path, u128::MAX).to_json(),
        url_encode(&name)
    ))
}

fn package_json_for_path(
    config: &DaemonConfig,
    path: &Path,
    extra: &str,
) -> Result<String, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_string();
    Ok(format!(
        "{{{}\"package_root\":\"{}\",\"path\":\"{}\",\"name\":\"{}\",\"file\":{},\"download\":\"/packages/download?name={}\"}}",
        extra,
        json_escape(&config.package_root.display().to_string()),
        json_escape(&path.display().to_string()),
        json_escape(&name),
        file_status(path, u128::MAX).to_json(),
        url_encode(&name)
    ))
}

fn package_by_run_json(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let run_id = params
        .get("run_id")
        .ok_or("packages/by-run requires run_id")?
        .trim();
    if run_id.is_empty() {
        return Err("packages/by-run requires non-empty run_id".to_string());
    }
    let ledger = config.dynout.join("probe_package_by_run.tsv");
    let Ok(file) = File::open(&ledger) else {
        return Err(format!("missing package run ledger: {}", ledger.display()));
    };
    let mut matched: Option<BTreeMap<String, String>> = None;
    for row in read_tsv_rows(file) {
        if row.get("run_id").map(String::as_str) == Some(run_id) {
            matched = Some(row);
        }
    }
    let row = matched.ok_or_else(|| format!("no package recorded for run_id={run_id}"))?;
    let path = row
        .get("package_path")
        .map(PathBuf::from)
        .ok_or_else(|| format!("package ledger row has no package_path for run_id={run_id}"))?;
    let root = fs::canonicalize(&config.package_root).map_err(|err| {
        format!(
            "canonicalize package_root {}: {err}",
            config.package_root.display()
        )
    })?;
    let path = fs::canonicalize(&path)
        .map_err(|err| format!("canonicalize package path {}: {err}", path.display()))?;
    if !path.starts_with(&root) {
        return Err(format!(
            "package path escapes package_root: {}",
            path.display()
        ));
    }
    package_json_for_path(
        config,
        &path,
        &format!(
            "\"run_id\":\"{}\",\"ledger\":\"{}\",",
            json_escape(run_id),
            json_escape(&ledger.display().to_string())
        ),
    )
}

fn find_latest_package(config: &DaemonConfig) -> Result<PathBuf, String> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    let entries = fs::read_dir(&config.package_root)
        .map_err(|err| format!("read package_root {}: {err}", config.package_root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("read package entry: {err}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext != "7z" && ext != "zip" {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH);
        if best
            .as_ref()
            .map(|(best_time, _)| modified > *best_time)
            .unwrap_or(true)
        {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path)
        .ok_or_else(|| format!("no package files in {}", config.package_root.display()))
}

fn resolve_package_download_path(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<PathBuf, String> {
    let requested = if let Some(name) = params.get("name") {
        if name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(format!("invalid package name: {name}"));
        }
        config.package_root.join(name)
    } else if let Some(path) = params.get("path") {
        PathBuf::from(path)
    } else {
        find_latest_package(config)?
    };
    let root = fs::canonicalize(&config.package_root).map_err(|err| {
        format!(
            "canonicalize package_root {}: {err}",
            config.package_root.display()
        )
    })?;
    let path = fs::canonicalize(&requested)
        .map_err(|err| format!("canonicalize package path {}: {err}", requested.display()))?;
    if !path.starts_with(&root) {
        return Err(format!(
            "package path escapes package_root: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[derive(Clone, Debug)]
struct FileStatus {
    exists: bool,
    bytes: u64,
    age_ms: Option<u128>,
    fresh: bool,
}

impl FileStatus {
    fn to_json(&self) -> String {
        format!(
            "{{\"exists\":{},\"bytes\":{},\"age_ms\":{},\"fresh\":{}}}",
            bool_json(self.exists),
            self.bytes,
            self.age_ms
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_string()),
            bool_json(self.fresh)
        )
    }
}

fn file_status(path: &Path, stale_ms: u128) -> FileStatus {
    let Ok(meta) = fs::metadata(path) else {
        return FileStatus {
            exists: false,
            bytes: 0,
            age_ms: None,
            fresh: false,
        };
    };
    let age_ms = meta
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .map(|age| age.as_millis());
    FileStatus {
        exists: true,
        bytes: meta.len(),
        age_ms,
        fresh: age_ms.map(|age| age <= stale_ms).unwrap_or(false),
    }
}

fn files_equal_by_len_and_bytes(left: &Path, right: &Path) -> io::Result<bool> {
    let left_meta = fs::metadata(left)?;
    let right_meta = fs::metadata(right)?;
    if left_meta.len() != right_meta.len() {
        return Ok(false);
    }
    let mut left_file = File::open(left)?;
    let mut right_file = File::open(right)?;
    let mut left_buf = [0u8; 64 * 1024];
    let mut right_buf = [0u8; 64 * 1024];
    loop {
        let left_n = left_file.read(&mut left_buf)?;
        let right_n = right_file.read(&mut right_buf)?;
        if left_n != right_n {
            return Ok(false);
        }
        if left_n == 0 {
            return Ok(true);
        }
        if left_buf[..left_n] != right_buf[..right_n] {
            return Ok(false);
        }
    }
}

fn read_key_value_tsv(path: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(file) = File::open(path) else {
        return out;
    };
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.unwrap_or_default();
        if idx == 0 && line.starts_with("key\t") {
            continue;
        }
        let mut cols = line.splitn(2, '\t');
        let Some(key) = cols.next() else { continue };
        let Some(value) = cols.next() else { continue };
        if !key.is_empty() {
            out.insert(key.to_string(), value.to_string());
        }
    }
    out
}

fn read_latest_tsv_row(path: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(file) = File::open(path) else {
        return out;
    };
    let mut lines = BufReader::new(file).lines();
    let Some(Ok(header)) = lines.next() else {
        return out;
    };
    let headers: Vec<_> = header.split('\t').map(str::to_string).collect();
    let mut latest = None;
    for line in lines {
        let line = line.unwrap_or_default();
        if !line.trim().is_empty() {
            latest = Some(line);
        }
    }
    let Some(latest) = latest else {
        return out;
    };
    for (key, value) in headers.iter().zip(latest.split('\t')) {
        out.insert(key.clone(), value.to_string());
    }
    out
}

pub(crate) fn read_tsv_header(path: &Path) -> Option<Vec<String>> {
    let file = File::open(path).ok()?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next()?.ok()?;
    Some(header.split('\t').map(str::to_string).collect())
}

pub(crate) fn read_tail_lines(path: &Path, tail_bytes: u64) -> Vec<String> {
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    let len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let start = len.saturating_sub(tail_bytes);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut text = String::new();
    if file
        .take(len.saturating_sub(start))
        .read_to_string(&mut text)
        .is_err()
    {
        return Vec::new();
    }
    let mut lines: Vec<String> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect();
    if start > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    lines
}

fn row_from_header_line(headers: &[String], line: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, value) in headers.iter().zip(line.split('\t')) {
        out.insert(key.clone(), value.to_string());
    }
    out
}

fn read_latest_tsv_row_fast(path: &Path) -> BTreeMap<String, String> {
    let Some(headers) = read_tsv_header(path) else {
        return BTreeMap::new();
    };
    if let Some(line) = read_tail_lines(path, 256 * 1024).into_iter().last() {
        return row_from_header_line(&headers, &line);
    }
    read_latest_tsv_row(path)
}

fn read_latest_tsv_row_matching(
    path: &Path,
    match_key: &str,
    match_value: &str,
) -> Option<BTreeMap<String, String>> {
    let Ok(file) = File::open(path) else {
        return None;
    };
    let mut latest = None;
    for row in read_tsv_rows(file) {
        if row.get(match_key).map(String::as_str) == Some(match_value) {
            latest = Some(row);
        }
    }
    latest
}

fn read_latest_tsv_row_matching_fast(
    path: &Path,
    match_key: &str,
    match_value: &str,
) -> Option<BTreeMap<String, String>> {
    let headers = read_tsv_header(path)?;
    for line in read_tail_lines(path, 512 * 1024).into_iter().rev() {
        let row = row_from_header_line(&headers, &line);
        if row.get(match_key).map(String::as_str) == Some(match_value) {
            return Some(row);
        }
    }
    read_latest_tsv_row_matching(path, match_key, match_value)
}

fn tsv_header_contains(path: &Path, column: &str) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut lines = BufReader::new(file).lines();
    let Some(Ok(header)) = lines.next() else {
        return false;
    };
    header.split('\t').any(|item| item == column)
}

fn read_tsv_rows(file: File) -> Vec<BTreeMap<String, String>> {
    let mut rows = Vec::new();
    let mut lines = BufReader::new(file).lines();
    let Some(Ok(header)) = lines.next() else {
        return rows;
    };
    let headers: Vec<_> = header.split('\t').map(str::to_string).collect();
    for line in lines {
        let line = line.unwrap_or_default();
        if line.trim().is_empty() {
            continue;
        }
        let mut row = BTreeMap::new();
        for (key, value) in headers.iter().zip(line.split('\t')) {
            row.insert(key.clone(), value.to_string());
        }
        rows.push(row);
    }
    rows
}

fn tsv_data_row_count(path: &Path) -> usize {
    let Ok(file) = File::open(path) else {
        return 0;
    };
    BufReader::new(file)
        .lines()
        .skip(1)
        .filter(|line| {
            line.as_ref()
                .map(|line| !line.trim().is_empty())
                .unwrap_or(false)
        })
        .count()
}

fn latest_pose_summary(row: &BTreeMap<String, String>) -> String {
    if row.is_empty() {
        return String::new();
    }
    let x = row
        .get("post_pos_x")
        .or_else(|| row.get("x"))
        .map(String::as_str)
        .unwrap_or("");
    let y = row
        .get("post_pos_y")
        .or_else(|| row.get("y"))
        .map(String::as_str)
        .unwrap_or("");
    let z = row
        .get("post_pos_z")
        .or_else(|| row.get("z"))
        .map(String::as_str)
        .unwrap_or("");
    let yaw = row
        .get("post_ang_z")
        .or_else(|| row.get("yaw"))
        .map(String::as_str)
        .unwrap_or("");
    let speed = row.get("speed").map(String::as_str).unwrap_or("");
    format!("x={x} y={y} z={z} yaw={yaw} speed={speed}")
}

fn json_extract_string_after(text: &str, after: &str, marker: &str) -> Option<String> {
    let start = text.find(after).map(|idx| idx + after.len()).unwrap_or(0);
    let scoped = &text[start..];
    let idx = scoped.find(marker)? + marker.len();
    let rest = &scoped[idx..];
    let end = rest.find('"')?;
    Some(rest[..end].replace("\\\"", "\"").replace("\\\\", "\\"))
}

fn json_extract_last_string(text: &str, marker: &str) -> Option<String> {
    let idx = text.rfind(marker)? + marker.len();
    let rest = &text[idx..];
    let end = rest.find('"')?;
    Some(rest[..end].replace("\\\"", "\"").replace("\\\\", "\\"))
}

fn json_extract_bool_after(text: &str, after: &str, marker: &str) -> Option<bool> {
    let start = text.find(after).map(|idx| idx + after.len()).unwrap_or(0);
    let scoped = &text[start..];
    let idx = scoped.find(marker)? + marker.len();
    let rest = scoped[idx..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_extract_number_after(text: &str, after: &str, marker: &str) -> Option<String> {
    let start = text.find(after).map(|idx| idx + after.len()).unwrap_or(0);
    let scoped = &text[start..];
    let idx = scoped.find(marker)? + marker.len();
    let rest = scoped[idx..].trim_start();
    let end = rest
        .find(|ch: char| !ch.is_ascii_digit() && ch != '-' && ch != '+')
        .unwrap_or(rest.len());
    if end == 0 {
        None
    } else {
        Some(rest[..end].to_string())
    }
}

fn json_extract_float_after(text: &str, after: &str, marker: &str) -> Option<String> {
    let start = text.find(after).map(|idx| idx + after.len()).unwrap_or(0);
    let scoped = &text[start..];
    let idx = scoped.find(marker)? + marker.len();
    let rest = scoped[idx..].trim_start();
    let end = rest
        .find(|ch: char| {
            !ch.is_ascii_digit() && ch != '-' && ch != '+' && ch != '.' && ch != 'e' && ch != 'E'
        })
        .unwrap_or(rest.len());
    if end == 0 || rest[..end].parse::<f64>().is_err() {
        None
    } else {
        Some(rest[..end].to_string())
    }
}

fn live_snapshot_self_fields(text: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    if json_extract_number_after(text, "\"pose\":", "\"present\":").as_deref() != Some("1") {
        return fields;
    }
    if let Some(value) = json_extract_pid_after(text, "\"owner\":", "\"object\":") {
        fields.insert("owner_this".to_string(), value);
    }
    if let Some(value) = json_extract_pid_after(text, "\"pose\":", "\"object\":") {
        fields.insert("pose_this".to_string(), value);
    }
    for key in ["x", "y", "z", "yaw", "speed", "vel_x", "vel_y", "vel_z"] {
        if let Some(value) = json_extract_float_after(text, "\"pose\":", &format!("\"{key}\":")) {
            fields.insert(key.to_string(), value);
        }
    }
    for key in ["surface_mode", "b1c", "rail_state"] {
        if let Some(value) = json_extract_number_after(text, "\"physics\":", &format!("\"{key}\":"))
        {
            fields.insert(key.to_string(), value);
        }
    }
    fields
}

fn pointer_identity_eq(left: &str, right: &str) -> bool {
    fn parse(value: &str) -> Option<u64> {
        let value = value.trim();
        if value.is_empty() {
            return None;
        }
        let digits = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .unwrap_or(value);
        u64::from_str_radix(digits, 16)
            .ok()
            .filter(|value| *value != 0)
    }
    matches!((parse(left), parse(right)), (Some(left), Some(right)) if left == right)
}

pub(crate) fn json_extract_pid_after(text: &str, after: &str, marker: &str) -> Option<String> {
    let start = if after.is_empty() {
        0
    } else {
        text.find(after)? + after.len()
    };
    let scoped = &text[start..];
    let idx = scoped.find(marker)? + marker.len();
    let rest = scoped[idx..].trim_start();
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        let value = &quoted[..end];
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    } else {
        let end = rest
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(rest.len());
        if end == 0 {
            None
        } else {
            Some(rest[..end].to_string())
        }
    }
}

#[derive(Clone, Debug)]
struct CaptureProcessView {
    injected_pid: String,
    current_pid: String,
    live_source: String,
    live_kart_present: bool,
    alive: bool,
    pid_mismatch: bool,
}

impl CaptureProcessView {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"injected_pid\":\"{}\",",
                "\"current_pid\":\"{}\",",
                "\"live_source\":\"{}\",",
                "\"live_kart_present\":{},",
                "\"alive\":{},",
                "\"pid_mismatch\":{}",
                "}}"
            ),
            json_escape(&self.injected_pid),
            json_escape(&self.current_pid),
            json_escape(&self.live_source),
            bool_json(self.live_kart_present),
            bool_json(self.alive),
            bool_json(self.pid_mismatch)
        )
    }
}

fn capture_process_view(config: &DaemonConfig, process_status: &str) -> CaptureProcessView {
    let scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
    let process = read_key_value_tsv(&config.dynout.join("probe_process.tsv"));
    let injected_pid = scenario
        .get("pid_at_inject")
        .or_else(|| process.get("pid"))
        .cloned()
        .unwrap_or_default();
    let live_source =
        json_extract_string_after(process_status, "", "\"source\":\"").unwrap_or_default();
    let current_pid = if live_source == "powershell_live" {
        live_kart_pid_from_process_status(process_status).unwrap_or_default()
    } else {
        String::new()
    };
    let live_kart_present = !current_pid.is_empty();
    let pid_mismatch =
        !injected_pid.is_empty() && !current_pid.is_empty() && injected_pid != current_pid;
    let alive = !injected_pid.is_empty() && live_kart_present && !pid_mismatch;
    CaptureProcessView {
        injected_pid,
        current_pid,
        live_source,
        live_kart_present,
        alive,
        pid_mismatch,
    }
}

fn live_kart_pid_from_process_status(process_status: &str) -> Option<String> {
    let live_source =
        json_extract_string_after(process_status, "", "\"source\":\"").unwrap_or_default();
    if live_source != "powershell_live" {
        return None;
    }
    json_extract_pid_after(process_status, "\"name\":\"KartRider\"", "\"pid\":")
}

#[derive(Clone, Debug)]
struct LiveStateView {
    source: String,
    fresh: bool,
    truth_current: bool,
    phase: String,
    stage_class: String,
    selected_phase: String,
    selected_step: String,
    pose_summary: String,
    control_allowed: bool,
    control_reason: String,
}

impl LiveStateView {
    fn to_json(&self) -> String {
        format!(
            "{{\"source\":\"{}\",\"fresh\":{},\"truth_current\":{},\"phase\":\"{}\",\"stage_class\":\"{}\",\"selected_phase\":\"{}\",\"selected_step\":\"{}\",\"pose\":\"{}\",\"control_allowed\":{},\"control_reason\":\"{}\"}}",
            json_escape(&self.source),
            bool_json(self.fresh),
            bool_json(self.truth_current),
            json_escape(&self.phase),
            json_escape(&self.stage_class),
            json_escape(&self.selected_phase),
            json_escape(&self.selected_step),
            json_escape(&self.pose_summary),
            bool_json(self.control_allowed),
            json_escape(&self.control_reason)
        )
    }
}

fn current_live_state(config: &DaemonConfig) -> LiveStateView {
    let pose_summary =
        latest_pose_summary(&read_latest_tsv_row(&config.dynout.join("probe_pose.tsv")));
    let process_status = daemon_process_status_json_cached(config);
    if live_kart_pid_from_process_status(&process_status).is_none() {
        return LiveStateView {
            source: "powershell_live".to_string(),
            fresh: false,
            truth_current: false,
            phase: "no_process".to_string(),
            stage_class: "none".to_string(),
            selected_phase: String::new(),
            selected_step: String::new(),
            pose_summary,
            control_allowed: false,
            control_reason: "no_live_kart_process".to_string(),
        };
    }
    let live_snapshot_path = config.dynout.join("probe_live_state.json");
    let live_snapshot_file = file_status(&live_snapshot_path, 2000);
    if live_snapshot_file.fresh {
        if let Ok(text) = fs::read_to_string(&live_snapshot_path) {
            let phase = json_extract_string_after(&text, "\"live\":", "\"phase\":\"")
                .unwrap_or_else(|| "unknown".to_string());
            let stage_class = json_extract_string_after(&text, "\"live\":", "\"stage_class\":\"")
                .unwrap_or_else(|| "unknown".to_string());
            let control_allowed =
                json_extract_bool_after(&text, "\"live\":", "\"control_allowed\":")
                    .unwrap_or(false);
            let control_reason =
                json_extract_string_after(&text, "\"live\":", "\"control_reason\":\"")
                    .unwrap_or_else(|| "unknown".to_string());
            let selected_phase = if stage_class == "game" {
                json_extract_number_after(&text, "\"stage\":", "\"selected_phase\":")
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let selected_step = if stage_class == "game" {
                json_extract_number_after(&text, "\"stage\":", "\"selected_step\":")
                    .unwrap_or_default()
            } else {
                String::new()
            };
            return LiveStateView {
                source: "dll_snapshot".to_string(),
                fresh: true,
                truth_current: true,
                phase,
                stage_class,
                selected_phase,
                selected_step,
                pose_summary,
                control_allowed,
                control_reason,
            };
        }
    }

    let stage_file = file_status(&config.dynout.join("probe_stage.tsv"), 2000);
    let truth_file = file_status(&config.dynout.join("probe_stage_truth.tsv"), 2000);
    let truth = read_key_value_tsv(&config.dynout.join("probe_stage_truth.tsv"));
    let phase = truth
        .get("phase")
        .cloned()
        .unwrap_or_else(|| "no_stage".to_string());
    let stage_class = truth
        .get("class")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let truth_current = stage_file.fresh && truth_file.exists;
    let control_allowed = truth_current
        && truth
            .get("control_allowed")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let control_reason = if truth_current {
        truth
            .get("control_reason")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "stale_or_missing_stage_input".to_string()
    };
    LiveStateView {
        source: "dynout_bridge".to_string(),
        fresh: stage_file.fresh,
        truth_current,
        phase,
        stage_class,
        selected_phase: truth.get("selected_phase").cloned().unwrap_or_default(),
        selected_step: truth.get("selected_step").cloned().unwrap_or_default(),
        pose_summary,
        control_allowed,
        control_reason,
    }
}

fn wait_for_live<F>(
    config: &DaemonConfig,
    timeout_ms: u64,
    poll_ms: u64,
    mut predicate: F,
) -> Result<LiveStateView, String>
where
    F: FnMut(&LiveStateView) -> bool,
{
    let start = SystemTime::now();
    loop {
        let state = current_live_state(config);
        if predicate(&state) {
            return Ok(state);
        }
        let elapsed = SystemTime::now()
            .duration_since(start)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        if elapsed >= timeout_ms {
            return Err(format!(
                "timeout waiting for live state after {timeout_ms}ms: {}",
                state.to_json()
            ));
        }
        thread::sleep(Duration::from_millis(poll_ms.max(50)));
    }
}

fn wait_for_live_or_cancel<F>(
    config: &DaemonConfig,
    timeout_ms: u64,
    poll_ms: u64,
    guard: &ExperimentRunGuard,
    mut predicate: F,
) -> Result<LiveStateView, String>
where
    F: FnMut(&LiveStateView) -> bool,
{
    let start = SystemTime::now();
    loop {
        if experiment_cancelled(guard) {
            return Err("experiment_cancelled".to_string());
        }
        let state = current_live_state(config);
        if predicate(&state) {
            return Ok(state);
        }
        let elapsed = SystemTime::now()
            .duration_since(start)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        if elapsed >= timeout_ms {
            return Err(format!(
                "timeout waiting for live state after {timeout_ms}ms: {}",
                state.to_json()
            ));
        }
        thread::sleep(Duration::from_millis(poll_ms.max(50)));
    }
}

fn is_startable_phase(phase: &str) -> bool {
    matches!(
        phase,
        "no_stage" | "unknown" | "ready_waiting_start" | "ready_unknown" | "ready_starting"
    )
}

fn is_terminal_nonstart_phase(phase: &str) -> bool {
    matches!(
        phase,
        "game_clear_hold" | "game_pause" | "game_result" | "final_settlement"
    )
}

fn is_room_wait_phase(phase: &str) -> bool {
    matches!(
        phase,
        "ready_waiting_start" | "ready_unknown" | "ready_starting"
    )
}

fn is_start_input_observed(state: &LiveStateView) -> bool {
    if !state.fresh || !state.truth_current {
        return false;
    }
    state.control_allowed || state.stage_class == "game" || state.phase == "ready_starting"
}

fn append_daemon_event(config: &DaemonConfig, event: &str, detail: &str) {
    let path = config.dynout.join("probe_event.tsv");
    let exists = path.exists();
    if fs::create_dir_all(&config.dynout).is_err() {
        return;
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    if !exists {
        let _ = file.write_all(b"time_local\tevent\tdetail\n");
    }
    let _ = writeln!(
        file,
        "{}\tdaemon_{}\t{}",
        daemon_time_ms(),
        event,
        detail.replace('\t', " ")
    );
}

fn append_control_evidence(
    config: &DaemonConfig,
    op: &str,
    keys: &[Key],
    duration_ms: u64,
    focus: bool,
    before: &LiveStateView,
    after: &LiveStateView,
    before_self: &SelfStateView,
    after_self: &SelfStateView,
    result: &str,
) {
    let path = config.dynout.join("probe_control_evidence.tsv");
    let mut exists = path.exists();
    if fs::create_dir_all(&config.dynout).is_err() {
        return;
    }
    if exists && !tsv_header_contains(&path, "self_delta") {
        let legacy = config.dynout.join(format!(
            "probe_control_evidence.legacy.{}.tsv",
            daemon_time_ms()
        ));
        let _ = fs::rename(&path, legacy);
        exists = false;
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    if !exists {
        let _ = file.write_all(
            b"time_ms\top\tkeys\tduration_ms\tfocus\tbefore_phase\tbefore_control_allowed\tbefore_pose\tafter_phase\tafter_control_allowed\tafter_pose\tbefore_self\tafter_self\tself_delta\tresult\n",
        );
    }
    let self_delta = after_self.delta_from(before_self);
    let _ = writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        daemon_time_ms(),
        clean(op),
        clean(&key_names(keys)),
        duration_ms,
        bool_json(focus),
        clean(&before.phase),
        bool_json(before.control_allowed),
        clean(&before.pose_summary),
        clean(&after.phase),
        bool_json(after.control_allowed),
        clean(&after.pose_summary),
        clean(&before_self.summary()),
        clean(&after_self.summary()),
        clean(&self_delta),
        clean(result)
    );
}

fn control_denied_reason(state: &LiveStateView) -> String {
    format!(
        "denied_control_not_allowed:phase={} reason={} truth_current={} fresh={}",
        state.phase,
        state.control_reason,
        bool_json(state.truth_current),
        bool_json(state.fresh)
    )
}

fn wait_for_self_drive_gate(
    config: &DaemonConfig,
    timeout_ms: u64,
    poll_ms: u64,
    max_speed: f64,
) -> Result<SelfStateView, String> {
    let start = tick_millis();
    let mut last = current_self_state(config);
    loop {
        if last.drive_gate_reason(max_speed).is_none() {
            return Ok(last);
        }
        if tick_millis().saturating_sub(start) >= u128::from(timeout_ms) {
            let reason = last
                .drive_gate_reason(max_speed)
                .unwrap_or_else(|| "unknown_self_gate_failure".to_string());
            return Err(format!("{reason}; self={}", last.summary()));
        }
        thread::sleep(Duration::from_millis(poll_ms));
        last = current_self_state(config);
    }
}

fn pose_component(summary: &str, key: &str) -> Option<f64> {
    let prefix = format!("{key}=");
    let start = summary.find(&prefix)? + prefix.len();
    let rest = &summary[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

fn pose_distance(a: &str, b: &str) -> Option<f64> {
    let ax = pose_component(a, "x")?;
    let ay = pose_component(a, "y")?;
    let az = pose_component(a, "z")?;
    let bx = pose_component(b, "x")?;
    let by = pose_component(b, "y")?;
    let bz = pose_component(b, "z")?;
    Some(((bx - ax).powi(2) + (by - ay).powi(2) + (bz - az).powi(2)).sqrt())
}

fn append_control_motion_evidence(
    config: &DaemonConfig,
    op: &str,
    keys: &[Key],
    duration_ms: u64,
    before: &LiveStateView,
    after: &LiveStateView,
    settled: &LiveStateView,
    settled_observed: bool,
) {
    let path = config.dynout.join("probe_control_motion.tsv");
    let exists = path.exists();
    if fs::create_dir_all(&config.dynout).is_err() {
        return;
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    if !exists {
        let _ = file.write_all(
            b"time_ms\top\tkeys\tduration_ms\tbefore_pose\tafter_pose\tsettled_pose\tsettled_observed\tdistance_before_after\tdistance_before_settled\n",
        );
    }
    let before_after = pose_distance(&before.pose_summary, &after.pose_summary)
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default();
    let before_settled = pose_distance(&before.pose_summary, &settled.pose_summary)
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default();
    let _ = writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        daemon_time_ms(),
        clean(op),
        clean(&key_names(keys)),
        duration_ms,
        clean(&before.pose_summary),
        clean(&after.pose_summary),
        clean(&settled.pose_summary),
        bool_json(settled_observed),
        before_after,
        before_settled
    );
}

fn key_names(keys: &[Key]) -> String {
    keys.iter()
        .map(|key| key.name())
        .collect::<Vec<_>>()
        .join("+")
}

fn daemon_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn wait_after_start_input(
    config: &DaemonConfig,
    _before: &LiveStateView,
    timeout_ms: u64,
) -> (LiveStateView, bool, &'static str) {
    let mut last = current_live_state(config);
    if timeout_ms == 0 {
        return (last, false, "sent_no_wait");
    }
    let start = SystemTime::now();
    loop {
        if is_start_input_observed(&last) {
            return (last, true, "sent_wait_observed");
        }
        let elapsed = SystemTime::now()
            .duration_since(start)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        if elapsed >= timeout_ms {
            return (last, false, "sent_wait_timeout");
        }
        thread::sleep(Duration::from_millis(100));
        last = current_live_state(config);
    }
}

fn control_start_if_needed(config: &DaemonConfig, ms: u64, wait_ms: u64) -> Result<String, String> {
    let state = current_live_state(config);
    let before_self = current_self_state(config);
    let mut sent = false;
    let mut focus = false;
    let mut send_label = "skipped";
    let mut result_label = "skipped".to_string();
    let mut observed = false;
    let mut after = state.clone();
    if !state.control_allowed && is_startable_phase(&state.phase) {
        focus = focus_kart_rider_retry(2_000);
        if focus {
            execute(
                &[Command::Hold {
                    keys: vec![Key::F5],
                    duration_ms: ms,
                }],
                &mut EventLog::from_optional_path(None)?,
            )?;
            send_label = "sent_focus";
        } else if post_key_to_kart_rider(Key::F5, ms) {
            send_label = "sent_postmessage";
        } else {
            execute(
                &[Command::Hold {
                    keys: vec![Key::F5],
                    duration_ms: ms,
                }],
                &mut EventLog::from_optional_path(None)?,
            )?;
            send_label = "sent_without_focus";
        }
        sent = true;
        let wait_result = wait_after_start_input(config, &state, wait_ms);
        after = wait_result.0;
        observed = wait_result.1;
        result_label = format!("{send_label}:{}", wait_result.2);
        append_control_evidence(
            config,
            "start_if_needed",
            &[Key::F5],
            ms,
            focus,
            &state,
            &after,
            &before_self,
            &current_self_state(config),
            &result_label,
        );
        append_daemon_event(
            config,
            "start_if_needed",
            &format!(
                "sent=1 before_phase={} after_phase={} observed={} result={result_label}",
                state.phase, after.phase, observed
            ),
        );
    } else {
        append_control_evidence(
            config,
            "start_if_needed",
            &[],
            0,
            focus,
            &state,
            &state,
            &before_self,
            &before_self,
            "skipped",
        );
        append_daemon_event(
            config,
            "start_if_needed_skip",
            &format!(
                "sent=0 phase={} control_allowed={}",
                state.phase, state.control_allowed
            ),
        );
    }
    Ok(format!(
        "{{\"sent\":{},\"focus\":{},\"observed\":{},\"send_result\":\"{}\",\"result\":\"{}\",\"wait_ms\":{},\"before\":{},\"after\":{},\"state\":{}}}",
        bool_json(sent),
        bool_json(focus),
        bool_json(observed),
        json_escape(send_label),
        json_escape(&result_label),
        wait_ms,
        state.to_json(),
        after.to_json(),
        after.to_json()
    ))
}

fn wait_for_game_active_with_start_retries(
    config: &DaemonConfig,
    timeout_ms: u64,
    poll_ms: u64,
    start_ms: u64,
    start_wait_ms: u64,
    retry_interval_ms: u64,
) -> Result<(LiveStateView, u32), String> {
    let start = SystemTime::now();
    let mut retries = 0_u32;
    let mut last_retry_ms = 0_u64;
    loop {
        let state = current_live_state(config);
        let self_state = current_self_state(config);
        if state.fresh
            && state.truth_current
            && state.phase == "game_active"
            && state.control_allowed
            && self_state.drive_gate_reason(f64::MAX).is_none()
        {
            return Ok((state, retries));
        }
        let elapsed = SystemTime::now()
            .duration_since(start)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        if elapsed >= timeout_ms {
            return Err(format!(
                "timeout waiting for game_active with usable self after {timeout_ms}ms retries={retries}: state={} self={}",
                state.to_json(),
                json_escape(&self_state.summary())
            ));
        }
        if state.fresh
            && is_startable_phase(&state.phase)
            && elapsed.saturating_sub(last_retry_ms) >= retry_interval_ms
        {
            retries = retries.saturating_add(1);
            last_retry_ms = elapsed;
            append_daemon_event(
                config,
                "loop_start_retry",
                &format!("retry={retries} elapsed_ms={elapsed} phase={}", state.phase),
            );
            let _ = control_start_if_needed(config, start_ms, start_wait_ms.min(5000));
        }
        thread::sleep(Duration::from_millis(poll_ms.max(50)));
    }
}

fn compact_self_trace_sample(label: &str, t0_ms: u128, sample: &SelfStateView) -> String {
    let t_ms = tick_millis().saturating_sub(t0_ms);
    format!(
        concat!(
            "{{",
            "\"t_ms\":{},",
            "\"label\":\"{}\",",
            "\"fresh\":{},",
            "\"confidence\":\"{}\",",
            "\"identity\":\"{}\",",
            "\"pose\":{{\"x\":{},\"y\":{},\"z\":{},\"yaw\":{}}},",
            "\"velocity\":{{\"x\":{},\"y\":{},\"z\":{},\"speed\":{}}},",
            "\"physics\":{{\"surface_mode\":{},\"state_raw\":{},\"state_timer_raw\":{},\"b1c\":{},\"rail_state\":{}}},",
            "\"resources\":{{\"bcharge_active\":{},\"pit_active\":{},\"pit_accum\":{},\"pit_rate\":{},\"pit_mode\":{}}},",
            "\"warnings\":[{}]",
            "}}"
        ),
        t_ms,
        json_escape(label),
        bool_json(sample.fresh),
        json_escape(&sample.confidence),
        json_escape(&sample.identity_this),
        json_number_or_null(sample.x.as_deref()),
        json_number_or_null(sample.y.as_deref()),
        json_number_or_null(sample.z.as_deref()),
        json_number_or_null(sample.yaw.as_deref()),
        json_number_or_null(sample.vel_x.as_deref()),
        json_number_or_null(sample.vel_y.as_deref()),
        json_number_or_null(sample.vel_z.as_deref()),
        json_number_or_null(sample.speed.as_deref()),
        json_string_or_null(sample.surface_mode.as_deref()),
        json_string_or_null(sample.state_raw.as_deref()),
        json_string_or_null(sample.state_timer_raw.as_deref()),
        json_string_or_null(sample.b1c.as_deref()),
        json_string_or_null(sample.rail_state.as_deref()),
        json_string_or_null(sample.bcharge_active.as_deref()),
        json_string_or_null(sample.pit_active.as_deref()),
        json_number_or_null(sample.pit_accum.as_deref()),
        json_number_or_null(sample.pit_rate.as_deref()),
        json_string_or_null(sample.pit_mode.as_deref()),
        string_array_json(&sample.warnings)
    )
}

fn push_trace_samples_until(
    config: &DaemonConfig,
    samples: &mut Vec<String>,
    t0_ms: u128,
    label: &str,
    until_ms: u128,
    sample_ms: u64,
) {
    while tick_millis().saturating_sub(t0_ms) < until_ms {
        samples.push(compact_self_trace_sample(
            label,
            t0_ms,
            &current_self_state(config),
        ));
        thread::sleep(Duration::from_millis(sample_ms.max(1)));
    }
}

fn push_trace_samples_until_speed(
    config: &DaemonConfig,
    samples: &mut Vec<String>,
    t0_ms: u128,
    until_ms: u128,
    sample_ms: u64,
    target_speed: f64,
    tolerance: f64,
    forward_only: bool,
) -> (bool, Option<f64>) {
    let threshold = (target_speed - tolerance).max(0.0);
    let mut last_speed = None;
    while tick_millis().saturating_sub(t0_ms) < until_ms {
        let state = current_self_state(config);
        last_speed = if forward_only {
            state.forward_speed_f64()
        } else {
            state.speed_f64()
        };
        samples.push(compact_self_trace_sample("preload", t0_ms, &state));
        if state.fresh && last_speed.is_some_and(|speed| speed >= threshold) {
            return (true, last_speed);
        }
        thread::sleep(Duration::from_millis(sample_ms.max(1)));
    }
    (false, last_speed)
}

fn control_trace_hold(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let keys_raw = params.get("keys").map(String::as_str).unwrap_or("F5");
    let keys = parse_keys(keys_raw)?;
    let preload_keys_raw = params.get("preload_keys").map(String::as_str).unwrap_or("");
    let preload_keys = if preload_keys_raw.is_empty() {
        Vec::new()
    } else {
        parse_keys(preload_keys_raw)?
    };
    let switch_keys_raw = params.get("switch_keys").map(String::as_str).unwrap_or("");
    let switch_keys = if switch_keys_raw.is_empty() {
        Vec::new()
    } else {
        parse_keys(switch_keys_raw)?
    };
    let switch_ms = params
        .get("switch_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .min(10_000);
    if switch_keys.is_empty() != (switch_ms == 0) {
        return Err("switch_keys_and_switch_ms_must_be_set_together".to_string());
    }
    let switch2_keys_raw = params.get("switch2_keys").map(String::as_str).unwrap_or("");
    let switch2_keys = if switch2_keys_raw.is_empty() {
        Vec::new()
    } else {
        parse_keys(switch2_keys_raw)?
    };
    let switch2_ms = params
        .get("switch2_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .min(10_000);
    if switch2_keys.is_empty() != (switch2_ms == 0) {
        return Err("switch2_keys_and_switch2_ms_must_be_set_together".to_string());
    }
    if !switch2_keys.is_empty() && switch_keys.is_empty() {
        return Err("switch2_requires_switch1".to_string());
    }
    let preload_ms = params
        .get("preload_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .min(10_000);
    let preload_target_speed = params
        .get("preload_target_speed")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let preload_target_forward_speed = params_bool(params, "preload_target_forward_speed");
    let preload_target_tolerance = params
        .get("preload_target_tolerance")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.25)
        .min(2.0);
    let release_preload_before_hold = params_bool(params, "release_preload_before_hold");
    let preload_release_settle_ms = params
        .get("preload_release_settle_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .min(1_000);
    let hold_ms = params
        .get("ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(90)
        .min(10_000);
    let pre_ms = params
        .get("pre_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(100)
        .min(5_000);
    let post_ms = params
        .get("post_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(250)
        .min(10_000);
    let sample_ms = params
        .get("sample_ms")
        .or_else(|| params.get("sample_interval_ms"))
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(20)
        .clamp(5, 1_000);
    let input_backend = parse_input_backend_param(params)?;
    let before = current_live_state(config);
    let before_self = current_self_state(config);
    if !before.control_allowed && !params_bool(params, "force") {
        let result = control_denied_reason(&before);
        append_control_evidence(
            config,
            "trace_hold",
            &keys,
            hold_ms,
            false,
            &before,
            &before,
            &before_self,
            &before_self,
            &result,
        );
        return Err(result);
    }
    if let Some(required_rail_state) = params.get("require_rail_state") {
        let observed = before_self.rail_state.as_deref().unwrap_or("");
        if !before_self.fresh || observed != required_rail_state {
            let result = format!(
                "required_rail_state_not_met:required={required_rail_state}:observed={observed}:self_fresh={}",
                before_self.fresh
            );
            append_control_evidence(
                config,
                "trace_hold",
                &keys,
                hold_ms,
                false,
                &before,
                &before,
                &before_self,
                &before_self,
                &result,
            );
            return Err(result);
        }
    }
    if preload_target_speed.is_some() && preload_keys.is_empty() {
        return Err("preload_target_speed_requires_preload_keys".to_string());
    }
    let before_preload_speed = if preload_target_forward_speed {
        before_self.forward_speed_f64()
    } else {
        before_self.speed_f64()
    };
    if let (Some(target), Some(speed)) = (preload_target_speed, before_preload_speed) {
        if speed > target + preload_target_tolerance {
            let result = format!(
                "preload_target_speed_already_exceeded:speed={speed:.6}:target={target:.6}:tolerance={preload_target_tolerance:.6}"
            );
            append_control_evidence(
                config,
                "trace_hold",
                &keys,
                hold_ms,
                false,
                &before,
                &before,
                &before_self,
                &before_self,
                &result,
            );
            return Err(result);
        }
    }

    let focus = if input_backend.requires_foreground() {
        focus_kart_rider_retry(2_000)
    } else {
        false
    };
    let mut input = InputDriver::new(input_backend);
    let mut samples = Vec::new();
    let t0_ms = tick_millis();
    samples.push(compact_self_trace_sample("before", t0_ms, &before_self));
    push_trace_samples_until(
        config,
        &mut samples,
        t0_ms,
        "pre",
        pre_ms as u128,
        sample_ms,
    );
    let preload_down_ms = tick_millis().saturating_sub(t0_ms);
    let mut preload_target_reached = preload_target_speed.is_none();
    let mut preload_reached_speed = None;
    if !preload_keys.is_empty() {
        if let Err(error) = input.key_down_many(&preload_keys) {
            let _ = input.release_all();
            return Err(error);
        }
        if let Some(target_speed) = preload_target_speed {
            (preload_target_reached, preload_reached_speed) = push_trace_samples_until_speed(
                config,
                &mut samples,
                t0_ms,
                preload_down_ms.saturating_add(preload_ms as u128),
                sample_ms,
                target_speed,
                preload_target_tolerance,
                preload_target_forward_speed,
            );
            if !preload_target_reached {
                let _ = input.release_all();
                let after = current_live_state(config);
                let after_self = current_self_state(config);
                let result = format!(
                    "preload_target_speed_not_reached:target={target_speed:.6}:last_speed={}",
                    preload_reached_speed
                        .map(|speed| format!("{speed:.6}"))
                        .unwrap_or_else(|| "unknown".to_string())
                );
                append_control_evidence(
                    config,
                    "trace_hold",
                    &keys,
                    hold_ms,
                    focus,
                    &before,
                    &after,
                    &before_self,
                    &after_self,
                    &result,
                );
                return Err(result);
            }
        } else {
            push_trace_samples_until(
                config,
                &mut samples,
                t0_ms,
                "preload",
                preload_down_ms.saturating_add(preload_ms as u128),
                sample_ms,
            );
        }
    }
    let mut preload_released_before_hold = false;
    if release_preload_before_hold && !preload_keys.is_empty() {
        input.key_up_many(&preload_keys)?;
        preload_released_before_hold = true;
        if preload_release_settle_ms > 0 {
            let settle_end = tick_millis()
                .saturating_sub(t0_ms)
                .saturating_add(preload_release_settle_ms as u128);
            push_trace_samples_until(
                config,
                &mut samples,
                t0_ms,
                "preload_release",
                settle_end,
                sample_ms,
            );
        }
    }
    let key_down_ms = tick_millis().saturating_sub(t0_ms);
    if let Err(error) = input.key_down_many(&keys) {
        let _ = input.release_all();
        return Err(error);
    }
    push_trace_samples_until(
        config,
        &mut samples,
        t0_ms,
        "hold",
        key_down_ms.saturating_add(hold_ms as u128),
        sample_ms,
    );
    let switch_down_ms = if switch_keys.is_empty() {
        None
    } else {
        let mut persistent_keys = if preload_released_before_hold {
            Vec::new()
        } else {
            preload_keys.clone()
        };
        for key in &switch_keys {
            if !persistent_keys.contains(key) {
                persistent_keys.push(*key);
            }
        }
        let switch_release_keys = keys
            .iter()
            .copied()
            .filter(|key| !persistent_keys.contains(key))
            .collect::<Vec<_>>();
        let switch_press_keys = switch_keys
            .iter()
            .copied()
            .filter(|key| {
                !keys.contains(key) && (preload_released_before_hold || !preload_keys.contains(key))
            })
            .collect::<Vec<_>>();
        input.key_up_many(&switch_release_keys)?;
        input.key_down_many(&switch_press_keys)?;
        let switched_at = tick_millis().saturating_sub(t0_ms);
        push_trace_samples_until(
            config,
            &mut samples,
            t0_ms,
            "switch",
            switched_at.saturating_add(switch_ms as u128),
            sample_ms,
        );
        Some(switched_at)
    };
    let switch2_down_ms = if switch2_keys.is_empty() {
        None
    } else {
        let mut persistent_keys = if preload_released_before_hold {
            Vec::new()
        } else {
            preload_keys.clone()
        };
        for key in &switch2_keys {
            if !persistent_keys.contains(key) {
                persistent_keys.push(*key);
            }
        }
        let switch2_release_keys = switch_keys
            .iter()
            .copied()
            .filter(|key| !persistent_keys.contains(key))
            .collect::<Vec<_>>();
        let switch2_press_keys = switch2_keys
            .iter()
            .copied()
            .filter(|key| {
                !switch_keys.contains(key)
                    && (preload_released_before_hold || !preload_keys.contains(key))
            })
            .collect::<Vec<_>>();
        input.key_up_many(&switch2_release_keys)?;
        input.key_down_many(&switch2_press_keys)?;
        let switched_at = tick_millis().saturating_sub(t0_ms);
        push_trace_samples_until(
            config,
            &mut samples,
            t0_ms,
            "switch2",
            switched_at.saturating_add(switch2_ms as u128),
            sample_ms,
        );
        Some(switched_at)
    };
    let key_up_ms = tick_millis().saturating_sub(t0_ms);
    let mut release_keys = if preload_released_before_hold {
        Vec::new()
    } else {
        preload_keys.clone()
    };
    for key in &keys {
        if !release_keys.contains(key) {
            release_keys.push(*key);
        }
    }
    for key in &switch_keys {
        if !release_keys.contains(key) {
            release_keys.push(*key);
        }
    }
    for key in &switch2_keys {
        if !release_keys.contains(key) {
            release_keys.push(*key);
        }
    }
    let key_up_result = input.key_up_many(&release_keys);
    let release_result = input.release_all();
    key_up_result?;
    release_result?;
    push_trace_samples_until(
        config,
        &mut samples,
        t0_ms,
        "post",
        key_up_ms.saturating_add(post_ms as u128),
        sample_ms,
    );
    let after = current_live_state(config);
    let after_self = current_self_state(config);
    samples.push(compact_self_trace_sample("after", t0_ms, &after_self));
    append_control_evidence(
        config,
        "trace_hold",
        &keys,
        hold_ms,
        focus,
        &before,
        &after,
        &before_self,
        &after_self,
        "ok",
    );

    Ok(format!(
        concat!(
            "{{",
            "\"schema\":\"kr_control_trace_hold_v3\",",
            "\"backend\":\"{}\",",
            "\"focus\":{},",
            "\"keys\":\"{}\",",
            "\"preload_keys\":\"{}\",",
            "\"switch_keys\":\"{}\",",
            "\"switch_ms\":{},",
            "\"switch2_keys\":\"{}\",",
            "\"switch2_ms\":{},",
            "\"preload_ms\":{},",
            "\"preload_target_speed\":{},",
            "\"preload_target_tolerance\":{},",
            "\"preload_target_forward_speed\":{},",
            "\"preload_target_reached\":{},",
            "\"preload_reached_speed\":{},",
            "\"release_preload_before_hold\":{},",
            "\"preload_release_settle_ms\":{},",
            "\"hold_ms\":{},",
            "\"pre_ms\":{},",
            "\"post_ms\":{},",
            "\"sample_ms\":{},",
            "\"preload_down_ms\":{},",
            "\"key_down_ms\":{},",
            "\"switch_down_ms\":{},",
            "\"switch2_down_ms\":{},",
            "\"key_up_ms\":{},",
            "\"elapsed_ms\":{},",
            "\"sample_count\":{},",
            "\"before\":{},",
            "\"after\":{},",
            "\"samples\":[{}]",
            "}}"
        ),
        input_backend.name(),
        bool_json(focus),
        json_escape(&key_names(&keys)),
        json_escape(&key_names(&preload_keys)),
        json_escape(&key_names(&switch_keys)),
        switch_ms,
        json_escape(&key_names(&switch2_keys)),
        switch2_ms,
        preload_ms,
        preload_target_speed
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        preload_target_tolerance,
        bool_json(preload_target_forward_speed),
        bool_json(preload_target_reached),
        preload_reached_speed
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        bool_json(preload_released_before_hold),
        preload_release_settle_ms,
        hold_ms,
        pre_ms,
        post_ms,
        sample_ms,
        preload_down_ms,
        key_down_ms,
        switch_down_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        switch2_down_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        key_up_ms,
        tick_millis().saturating_sub(t0_ms),
        samples.len(),
        before.to_json(),
        after.to_json(),
        samples.join(",")
    ))
}

fn control_release_all(config: &DaemonConfig) -> Result<String, String> {
    let before = current_live_state(config);
    let before_self = current_self_state(config);
    let focus = focus_kart_rider_retry(2_000);
    execute(
        &[Command::ReleaseAll],
        &mut EventLog::from_optional_path(None)?,
    )?;
    let after = current_live_state(config);
    let after_self = current_self_state(config);
    append_control_evidence(
        config,
        "release_all",
        &[],
        0,
        focus,
        &before,
        &after,
        &before_self,
        &after_self,
        "ok",
    );
    Ok(format!(
        "{{\"focus\":{},\"before\":{},\"after\":{}}}",
        bool_json(focus),
        before.to_json(),
        after.to_json()
    ))
}

fn capture_start(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let existing_scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
    let existing_pid = existing_scenario
        .get("pid_at_inject")
        .cloned()
        .unwrap_or_default();
    let stop_requested = config.dynout.join("probe_stop.flag").exists();
    let stop_acknowledged = config.dynout.join("probe_capture_stopped.flag").exists();
    if !existing_pid.is_empty() && !stop_acknowledged {
        let process_status = live_process_status_json(config)?;
        let process = capture_process_view(config, &process_status);
        if process.alive {
            let state = if stop_requested { "stopping" } else { "active" };
            return Err(format!(
                "capture already {state} for pid={existing_pid}; use /capture/restart and wait for stop acknowledgement"
            ));
        }
    }
    let scenario = params
        .get("scenario")
        .map(String::as_str)
        .unwrap_or("capture");
    let expected_map = params
        .get("expected_map")
        .map(String::as_str)
        .unwrap_or("unknown");
    let expected_map_display_name = params
        .get("expected_map_display_name")
        .map(String::as_str)
        .unwrap_or("");
    let notes = params
        .get("notes")
        .map(String::as_str)
        .unwrap_or("http daemon capture");
    let timeout_seconds = params
        .get("timeout_seconds")
        .map(String::as_str)
        .unwrap_or("300");
    let profile = params
        .get("profile")
        .map(String::as_str)
        .unwrap_or("default");
    let wait_live_ms = params
        .get("wait_live_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let prepare = config
        .probe_root
        .join("scripts")
        .join("remote_capture_prepare.ps1");
    let profile_path = config
        .probe_root
        .join("profiles")
        .join(format!("{profile}.tsv"));
    let src_path = config.probe_root.join("dllmain.cpp");
    let ps = format!(
        concat!(
            "$ErrorActionPreference='Stop';",
            "$profileHash=(Get-FileHash -Algorithm SHA256 {}).Hash.ToLowerInvariant();",
            "$srcHash=if(Test-Path {}){{(Get-FileHash -Algorithm SHA256 {}).Hash.ToLowerInvariant()}}else{{'missing'}};",
            "& {} -Scenario {} -ExpectedMap {} -ExpectedMapDisplayName {} -Notes {} -TimeoutSeconds {} ",
            "-LocalGitCommit 'daemon-http' -LocalGitBranch 'daemon-http' -LocalGitDirty 'unknown' ",
            "-LocalProbeSourceSha256 $srcHash -ProfileName {} -ProfileSha256 $profileHash"
        ),
        ps_quote_path(&profile_path),
        ps_quote_path(&src_path),
        ps_quote_path(&src_path),
        ps_quote_path(&prepare),
        ps_quote(scenario),
        ps_quote(expected_map),
        ps_quote(expected_map_display_name),
        ps_quote(notes),
        timeout_seconds,
        ps_quote(profile),
    );
    run_powershell(&ps)?;
    let live_wait_json = if wait_live_ms > 0 {
        let state = wait_for_live(config, wait_live_ms, 250, |state| {
            state.fresh && state.truth_current
        })?;
        format!(
            ",\"live_wait\":{{\"ok\":true,\"timeout_ms\":{},\"state\":{}}}",
            wait_live_ms,
            state.to_json()
        )
    } else {
        String::new()
    };
    Ok(format!(
        "{{\"scenario\":\"{}\",\"expected_map\":\"{}\",\"state\":{}{} }}",
        json_escape(scenario),
        json_escape(expected_map),
        daemon_state_json(config),
        live_wait_json
    ))
}

fn capture_stop(config: &DaemonConfig, wait_ms: u64) -> Result<String, String> {
    fs::create_dir_all(&config.dynout)
        .map_err(|err| format!("create dynout {}: {err}", config.dynout.display()))?;
    File::create(config.dynout.join("probe_stop.flag"))
        .map_err(|err| format!("write probe_stop.flag: {err}"))?;
    // Hook restoration is owned by the in-process stop flag/ack handshake.
    // Releasing input is best-effort: a locked Windows input desktop rejects
    // SendInput, but that must not hide or abort a successful clean hook stop.
    let release_error = execute(
        &[Command::ReleaseAll],
        &mut EventLog::from_optional_path(None)?,
    )
    .err();
    let stopped_path = config.dynout.join("probe_capture_stopped.flag");
    let scenario = read_key_value_tsv(&config.dynout.join("probe_scenario.tsv"));
    let injected_pid = scenario.get("pid_at_inject").cloned().unwrap_or_default();
    let deadline_ms = tick_millis().saturating_add(wait_ms as u128);
    let mut acknowledged = stopped_path.exists();
    while wait_ms > 0 && !acknowledged && tick_millis() < deadline_ms {
        thread::sleep(Duration::from_millis(100));
        acknowledged = stopped_path.exists();
    }
    if wait_ms > 0 && !acknowledged {
        let release_detail = release_error
            .as_deref()
            .map(|value| format!("; release_all_error={value}"))
            .unwrap_or_default();
        return Err(format!(
            "probe did not acknowledge capture stop within {wait_ms} ms{release_detail}"
        ));
    }
    if acknowledged && !injected_pid.is_empty() {
        let process_status = live_process_status_json(config)?;
        let process = capture_process_view(config, &process_status);
        if !process.alive {
            return Err(format!(
                "capture process did not survive clean stop: injected_pid={} current_pid={} pid_mismatch={}",
                process.injected_pid, process.current_pid, process.pid_mismatch
            ));
        }
    }
    Ok(format!(
        "{{\"stop_flag\":true,\"acknowledged\":{},\"release_all_ok\":{},\"release_all_error\":\"{}\",\"state\":{}}}",
        bool_json(acknowledged),
        bool_json(release_error.is_none()),
        json_escape(release_error.as_deref().unwrap_or("")),
        daemon_state_json(config)
    ))
}

fn capture_package(
    config: &DaemonConfig,
    params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let scenario = params
        .get("scenario")
        .map(String::as_str)
        .unwrap_or("capture");
    let expected_map = params
        .get("expected_map")
        .map(String::as_str)
        .unwrap_or("unknown");
    let package = config
        .probe_root
        .join("scripts")
        .join("remote_capture_package.ps1");
    let ps = format!(
        "& {} -ScenarioSlug {} -MapSlug {}",
        ps_quote_path(&package),
        ps_quote(&slug(scenario)),
        ps_quote(&slug(expected_map))
    );
    let output = run_powershell(&ps)?;
    if let Some(package_path) = package_path_from_output(&output) {
        append_package_by_run(config, params, &package_path);
    }
    Ok(format!(
        "{{\"scenario\":\"{}\",\"expected_map\":\"{}\",\"output\":\"{}\"}}",
        json_escape(scenario),
        json_escape(expected_map),
        json_escape(&output)
    ))
}

fn package_path_from_output(output: &str) -> Option<PathBuf> {
    for line in output.replace('\r', "\n").lines() {
        let Some(rest) = line.strip_prefix("PACKAGE=") else {
            continue;
        };
        let value = rest.trim();
        if !value.is_empty() {
            return Some(PathBuf::from(value));
        }
    }
    None
}

fn append_package_by_run(config: &DaemonConfig, params: &BTreeMap<String, String>, package: &Path) {
    let Some(run_id) = params.get("run_id") else {
        return;
    };
    if run_id.trim().is_empty() {
        return;
    }
    if fs::create_dir_all(&config.dynout).is_err() {
        return;
    }
    let path = config.dynout.join("probe_package_by_run.tsv");
    let exists = path.exists();
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    if !exists {
        let _ = file
            .write_all(b"time_ms\trun_id\tscenario\texpected_map\tpackage_path\tpackage_name\n");
    }
    let package_name = package
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let _ = writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}",
        daemon_time_ms(),
        clean(run_id),
        clean(params.get("scenario").map(String::as_str).unwrap_or("")),
        clean(params.get("expected_map").map(String::as_str).unwrap_or("")),
        clean(&package.display().to_string()),
        clean(package_name)
    );
}

fn params_json(params: &BTreeMap<String, String>) -> String {
    params
        .iter()
        .map(|(key, value)| format!("\"{}\":\"{}\"", json_escape(key), json_escape(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn params_object_json(params: &BTreeMap<String, String>) -> String {
    format!("{{{}}}", params_json(params))
}

fn first_field(fields: &[(&BTreeMap<String, String>, &str)]) -> Option<String> {
    fields.iter().find_map(|(row, key)| {
        row.get(*key)
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
    })
}

fn first_field_from_row(row: &BTreeMap<String, String>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        row.get(*key)
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
    })
}

fn row_tick_ms(row: &BTreeMap<String, String>) -> Option<u64> {
    row.get("tick_ms")?.parse::<u64>().ok()
}

fn latest_row_tick_ms(rows: &[&BTreeMap<String, String>]) -> Option<u64> {
    rows.iter().filter_map(|row| row_tick_ms(row)).max()
}

fn fresh_row_near_tick(
    file: &FileStatus,
    row: &BTreeMap<String, String>,
    reference_tick_ms: Option<u64>,
    max_delta_ms: u64,
) -> Option<BTreeMap<String, String>> {
    if !file.fresh {
        return None;
    }
    let reference_tick_ms = reference_tick_ms?;
    let tick_ms = row_tick_ms(row)?;
    if tick_ms.abs_diff(reference_tick_ms) <= max_delta_ms {
        Some(row.clone())
    } else {
        None
    }
}

fn fresh_row_field(
    active_file: &FileStatus,
    active: &BTreeMap<String, String>,
    active_keys: &[&str],
    pose_file: &FileStatus,
    pose: &BTreeMap<String, String>,
    pose_keys: &[&str],
    owner_file: &FileStatus,
    owner: &BTreeMap<String, String>,
    owner_keys: &[&str],
    label: &str,
) -> (Option<String>, String) {
    if active_file.fresh {
        if let Some(value) = first_field_from_row(active, active_keys) {
            return (Some(value), "active".to_string());
        }
    }
    if pose_file.fresh {
        if let Some(value) = first_field_from_row(pose, pose_keys) {
            return (Some(value), "pose".to_string());
        }
    }
    if owner_file.fresh {
        if let Some(value) = first_field_from_row(owner, owner_keys) {
            return (Some(value), "owner".to_string());
        }
    }
    if active_file.exists {
        if let Some(value) = first_field_from_row(active, active_keys) {
            return (Some(value), format!("stale_active_{label}"));
        }
    }
    if pose_file.exists {
        if let Some(value) = first_field_from_row(pose, pose_keys) {
            return (Some(value), format!("stale_pose_{label}"));
        }
    }
    if owner_file.exists {
        if let Some(value) = first_field_from_row(owner, owner_keys) {
            return (Some(value), format!("stale_owner_{label}"));
        }
    }
    (None, "unknown".to_string())
}

fn matching_source_field(
    source: &str,
    active: &BTreeMap<String, String>,
    active_keys: &[&str],
    pose: &BTreeMap<String, String>,
    pose_keys: &[&str],
    owner: &BTreeMap<String, String>,
    owner_keys: &[&str],
) -> Option<String> {
    if source.starts_with("active") || source.starts_with("stale_active") {
        first_field_from_row(active, active_keys)
    } else if source.starts_with("pose") || source.starts_with("stale_pose") {
        first_field_from_row(pose, pose_keys)
    } else if source.starts_with("owner") || source.starts_with("stale_owner") {
        first_field_from_row(owner, owner_keys)
    } else {
        None
    }
}

fn owner_row_has_zero_pose_velocity(row: &BTreeMap<String, String>) -> bool {
    if row.is_empty() {
        return false;
    }
    let bad_pos = any_abs_out_of_range(row, &["pos_x", "pos_y", "pos_z"], 1_000_000.0);
    let bad_vel = any_abs_out_of_range(row, &["vel_x", "vel_y", "vel_z"], 100_000.0);
    let bad_basis = orientation_basis_is_bad(row);
    let bad_surface = row
        .get("surface_mode")
        .and_then(|value| value.parse::<u32>().ok())
        .map(|value| value > 1_000_000)
        .unwrap_or(false);
    if bad_pos || bad_vel || bad_basis || bad_surface {
        return true;
    }

    let pos_zero = all_near_zero(row, &["pos_x", "pos_y", "pos_z"], 0.000001);
    let vel_zero = all_near_zero(row, &["vel_x", "vel_y", "vel_z"], 0.000001);
    let basis_zero = all_near_zero(
        row,
        &[
            "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
        ],
        0.000001,
    );
    let mass_invalid_or_zero = row
        .get("mass_f32")
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| !value.is_finite() || value.abs() <= 0.000001)
        .unwrap_or(true);
    let speed_nonzero = first_field_from_row(row, &["initial_speed", "speed"])
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| value.abs() > 0.01)
        .unwrap_or(false);
    pos_zero && vel_zero && (speed_nonzero || (basis_zero && mass_invalid_or_zero))
}

fn any_abs_out_of_range(row: &BTreeMap<String, String>, keys: &[&str], limit: f64) -> bool {
    keys.iter().any(|key| {
        row.get(*key)
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| !value.is_finite() || value.abs() > limit)
            .unwrap_or(false)
    })
}

fn orientation_basis_is_bad(row: &BTreeMap<String, String>) -> bool {
    let keys = [
        "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
    ];
    let mut values = Vec::new();
    for key in keys {
        let Some(value) = row.get(key).and_then(|value| value.parse::<f64>().ok()) else {
            return false;
        };
        if !value.is_finite() {
            return true;
        }
        values.push(value);
    }
    for row_index in 0..3 {
        let base = row_index * 3;
        let len = (values[base] * values[base]
            + values[base + 1] * values[base + 1]
            + values[base + 2] * values[base + 2])
            .sqrt();
        if !len.is_finite() || (len > 0.000001 && (len - 1.0).abs() > 0.75) {
            return true;
        }
    }
    false
}

fn active_row_is_invalid(row: &BTreeMap<String, String>) -> bool {
    row.get("reason")
        .map(String::as_str)
        .map(|reason| reason.starts_with("active_invalid_sample"))
        .unwrap_or(false)
}

fn all_near_zero(row: &BTreeMap<String, String>, keys: &[&str], eps: f64) -> bool {
    keys.iter().all(|key| {
        row.get(*key)
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| value.is_finite() && value.abs() <= eps)
            .unwrap_or(false)
    })
}

fn vector_norm_values(x: Option<&str>, y: Option<&str>, z: Option<&str>) -> Option<String> {
    let vx = x?.parse::<f64>().ok()?;
    let vy = y?.parse::<f64>().ok()?;
    let vz = z?.parse::<f64>().ok()?;
    Some(format!("{:.9}", (vx * vx + vy * vy + vz * vz).sqrt()))
}

pub(crate) fn self_speed_value(
    active_fresh: bool,
    pose_fresh: bool,
    owner_fresh: bool,
    active: &BTreeMap<String, String>,
    pose: &BTreeMap<String, String>,
    owner: &BTreeMap<String, String>,
    vel_x: Option<&str>,
    vel_y: Option<&str>,
    vel_z: Option<&str>,
) -> Option<String> {
    vector_norm_values(vel_x, vel_y, vel_z)
        .or_else(|| {
            if active_fresh {
                first_field_from_row(active, &["speed"])
            } else {
                None
            }
        })
        .or_else(|| {
            if pose_fresh {
                first_field_from_row(pose, &["speed"])
            } else {
                None
            }
        })
        .or_else(|| {
            if owner_fresh {
                first_field_from_row(owner, &["speed"])
            } else {
                None
            }
        })
        .or_else(|| first_field_from_row(owner, &["initial_speed"]))
}

type MatrixProjection = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

fn matrix_velocity_projection(
    row: &BTreeMap<String, String>,
    vel_x: Option<&str>,
    vel_y: Option<&str>,
    vel_z: Option<&str>,
) -> MatrixProjection {
    let Some(vx) = vel_x.and_then(parse_f64) else {
        return (None, None, None, None, None, None);
    };
    let Some(vy) = vel_y.and_then(parse_f64) else {
        return (None, None, None, None, None, None);
    };
    let Some(vz) = vel_z.and_then(parse_f64) else {
        return (None, None, None, None, None, None);
    };
    let Some(m00) = row.get("post_m00").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m01) = row.get("post_m01").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m02) = row.get("post_m02").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m10) = row.get("post_m10").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m11) = row.get("post_m11").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m12) = row.get("post_m12").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m20) = row.get("post_m20").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m21) = row.get("post_m21").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    let Some(m22) = row.get("post_m22").and_then(|value| parse_f64(value)) else {
        return (None, None, None, None, None, None);
    };
    (
        normalized_dot((vx, vy, vz), (m00, m01, m02)),
        normalized_dot((vx, vy, vz), (m10, m11, m12)),
        normalized_dot((vx, vy, vz), (m20, m21, m22)),
        normalized_dot((vx, vy, vz), (m00, m10, m20)),
        normalized_dot((vx, vy, vz), (m01, m11, m21)),
        normalized_dot((vx, vy, vz), (m02, m12, m22)),
    )
}

fn normalized_dot(v: (f64, f64, f64), axis: (f64, f64, f64)) -> Option<String> {
    let norm = (axis.0 * axis.0 + axis.1 * axis.1 + axis.2 * axis.2).sqrt();
    if norm <= f64::EPSILON {
        return None;
    }
    Some(format!(
        "{:.9}",
        (v.0 * axis.0 + v.1 * axis.1 + v.2 * axis.2) / norm
    ))
}

fn parse_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok()
}

fn option_delta(before: Option<&str>, after: Option<&str>) -> Option<String> {
    let before = before?.parse::<f64>().ok()?;
    let after = after?.parse::<f64>().ok()?;
    Some(format!("{:.6}", after - before))
}

fn option_distance(
    ax: Option<&str>,
    ay: Option<&str>,
    az: Option<&str>,
    bx: Option<&str>,
    by: Option<&str>,
    bz: Option<&str>,
) -> Option<String> {
    let ax = ax?.parse::<f64>().ok()?;
    let ay = ay?.parse::<f64>().ok()?;
    let az = az?.parse::<f64>().ok()?;
    let bx = bx?.parse::<f64>().ok()?;
    let by = by?.parse::<f64>().ok()?;
    let bz = bz?.parse::<f64>().ok()?;
    Some(format!(
        "{:.6}",
        ((bx - ax).powi(2) + (by - ay).powi(2) + (bz - az).powi(2)).sqrt()
    ))
}

fn valid_u32_field(
    value: Option<String>,
    max_value: u32,
    field: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let value = value?;
    match value.parse::<u32>() {
        Ok(parsed) if parsed <= max_value => Some(value),
        _ => {
            warnings.push(format!("invalid_{field}:{value}"));
            None
        }
    }
}

fn valid_f64_abs_field(
    value: Option<String>,
    max_abs: f64,
    field: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let value = value?;
    match value.parse::<f64>() {
        Ok(parsed) if parsed.is_finite() && parsed.abs() <= max_abs => Some(value),
        _ => {
            warnings.push(format!("invalid_{field}:{value}"));
            None
        }
    }
}

fn json_number_or_null(value: Option<&str>) -> String {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return "null".to_string();
    };
    if value.parse::<f64>().is_ok() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

fn json_string_or_null(value: Option<&str>) -> String {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
}

fn string_array_json(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn ensure_experiment_run_id(params: &mut BTreeMap<String, String>) -> String {
    if let Some(run_id) = params.get("run_id") {
        if !run_id.trim().is_empty() {
            return run_id.clone();
        }
    }
    let run_id = format!("run_{}_{}", daemon_time_ms(), std::process::id());
    params.insert("run_id".to_string(), run_id.clone());
    run_id
}

fn write_experiment_manifest(
    config: &DaemonConfig,
    scenario: &str,
    expected_map: &str,
    params: &BTreeMap<String, String>,
    steps: &[String],
    status: &str,
) -> Result<(), String> {
    fs::create_dir_all(&config.dynout)
        .map_err(|err| format!("create dynout {}: {err}", config.dynout.display()))?;
    let path = config.dynout.join("probe_experiment.json");
    let run_id = params.get("run_id").map(String::as_str).unwrap_or("");
    let body = format!(
        concat!(
            "{{",
            "\"schema\":\"kr_probe_experiment_v1\",",
            "\"run_id\":\"{}\",",
            "\"status\":\"{}\",",
            "\"time_ms\":{},",
            "\"scenario\":\"{}\",",
            "\"expected_map\":\"{}\",",
            "\"daemon\":{{\"api\":\"kr-live-v1\",\"bind\":\"{}\",\"dynout\":\"{}\",\"probe_root\":\"{}\",\"package_root\":\"{}\"}},",
            "\"request_params\":{{{}}},",
            "\"steps\":[{}]",
            "}}\n"
        ),
        json_escape(run_id),
        json_escape(status),
        daemon_time_ms(),
        json_escape(scenario),
        json_escape(expected_map),
        json_escape(&config.bind),
        json_escape(&config.dynout.display().to_string()),
        json_escape(&config.probe_root.display().to_string()),
        json_escape(&config.package_root.display().to_string()),
        params_json(params),
        steps.join(",")
    );
    fs::write(&path, body).map_err(|err| format!("write {}: {err}", path.display()))
}

fn current_experiment_status(config: &DaemonConfig) -> String {
    let path = config.dynout.join("probe_experiment.json");
    let text = fs::read_to_string(&path).unwrap_or_default();
    json_extract_string_after(&text, "", "\"status\":\"").unwrap_or_default()
}

fn experiment_status_is_active(status: &str) -> bool {
    !status.is_empty()
        && !matches!(
            status,
            "done" | "failed" | "packaged" | "capture_stopped" | "return_ready_done" | "cancelled"
        )
}

fn experiment_manifest_blocks_start(fresh: bool, status: &str) -> bool {
    fresh && experiment_status_is_active(status)
}

fn experiment_cancel(config: &DaemonConfig, wait_ms: u64) -> Result<String, String> {
    EXPERIMENT_CANCEL_GENERATION.fetch_add(1, Ordering::SeqCst);
    fs::write(
        config.dynout.join("experiment_cancel.flag"),
        format!("{}\n", tick_millis()),
    )
    .map_err(|err| format!("write experiment cancel flag: {err}"))?;
    let manifest_path = config.dynout.join("probe_experiment.json");
    let text = fs::read_to_string(&manifest_path).unwrap_or_default();
    let run_id = json_extract_string_after(&text, "", "\"run_id\":\"")
        .unwrap_or_else(|| "cancelled".to_string());
    let scenario = json_extract_string_after(&text, "", "\"scenario\":\"")
        .unwrap_or_else(|| "cancelled".to_string());
    let expected_map = json_extract_string_after(&text, "", "\"expected_map\":\"")
        .unwrap_or_else(|| "unknown".to_string());
    let old_status = json_extract_string_after(&text, "", "\"status\":\"").unwrap_or_default();

    let stop_result = capture_stop(config, wait_ms)?;
    let mut params = BTreeMap::new();
    params.insert("run_id".to_string(), run_id.clone());
    params.insert("cancelled_old_status".to_string(), old_status.clone());
    let steps = vec![format!(
        "{{\"step\":\"experiment_cancel\",\"ok\":true,\"old_status\":\"{}\",\"capture_stop\":{}}}",
        json_escape(&old_status),
        stop_result
    )];
    write_experiment_manifest(
        config,
        &scenario,
        &expected_map,
        &params,
        &steps,
        "cancelled",
    )?;
    append_daemon_event(
        config,
        "experiment_cancel",
        &format!("run_id={run_id} old_status={old_status}"),
    );
    Ok(format!(
        "{{\"run_id\":\"{}\",\"old_status\":\"{}\",\"status\":\"cancelled\",\"manifest\":{}}}",
        json_escape(&run_id),
        json_escape(&old_status),
        daemon_experiment_json(config)
    ))
}

fn experiment_start_detached(
    config: &DaemonConfig,
    mut params: BTreeMap<String, String>,
    wait_return_ready: bool,
) -> Result<String, String> {
    let guard = try_begin_experiment()?;
    let manifest_file = file_status(&config.dynout.join("probe_experiment.json"), 5000);
    let current_status = current_experiment_status(config);
    if experiment_manifest_blocks_start(manifest_file.fresh, &current_status) {
        return Err(format!(
            "experiment already active: status={current_status}"
        ));
    }
    if wait_return_ready {
        params.insert("wait_return_ready".to_string(), "1".to_string());
    }
    let run_id = ensure_experiment_run_id(&mut params);
    let scenario = params
        .get("scenario")
        .cloned()
        .unwrap_or_else(|| "http_loop".to_string());
    let expected_map = params
        .get("expected_map")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let steps = vec!["{\"step\":\"detached_start\",\"ok\":true}".to_string()];
    write_experiment_manifest(
        config,
        &scenario,
        &expected_map,
        &params,
        &steps,
        "detached_queued",
    )?;
    append_daemon_event(
        config,
        "experiment_detached_start",
        &format!("scenario={scenario} map={expected_map}"),
    );

    let worker_config = config.clone();
    let worker_params = params.clone();
    let worker_scenario = scenario.clone();
    let worker_expected_map = expected_map.clone();
    thread::spawn(move || {
        let _guard = guard;
        if let Err(err) = experiment_loop_guarded(&worker_config, &worker_params, &_guard) {
            if err == "experiment_cancelled" {
                append_daemon_event(&worker_config, "experiment_detached_cancelled", &err);
                return;
            }
            let steps = vec![
                "{\"step\":\"detached_start\",\"ok\":true}".to_string(),
                format!(
                    "{{\"step\":\"detached_error\",\"ok\":false,\"error\":\"{}\"}}",
                    json_escape(&err)
                ),
            ];
            let _ = write_experiment_manifest(
                &worker_config,
                &worker_scenario,
                &worker_expected_map,
                &worker_params,
                &steps,
                "failed",
            );
            append_daemon_event(&worker_config, "experiment_detached_failed", &err);
        }
    });

    Ok(format!(
        "{{\"accepted\":true,\"run_id\":\"{}\",\"scenario\":\"{}\",\"expected_map\":\"{}\",\"status\":{}}}",
        json_escape(&run_id),
        json_escape(&scenario),
        json_escape(&expected_map),
        daemon_experiment_json(config)
    ))
}

fn experiment_loop(
    config: &DaemonConfig,
    request_params: &BTreeMap<String, String>,
) -> Result<String, String> {
    let guard = try_begin_experiment()?;
    experiment_loop_guarded(config, request_params, &guard)
}

fn experiment_loop_guarded(
    config: &DaemonConfig,
    request_params: &BTreeMap<String, String>,
    guard: &ExperimentRunGuard,
) -> Result<String, String> {
    match experiment_loop_inner(config, request_params, guard) {
        Ok(result) => Ok(result),
        Err(err) => {
            if err == "experiment_cancelled" || experiment_cancelled(guard) {
                append_daemon_event(config, "loop_cancelled", &err);
                let _ = execute(
                    &[Command::ReleaseAll],
                    &mut EventLog::from_optional_path(None)?,
                );
                return Err("experiment_cancelled".to_string());
            }
            if params_bool(request_params, "no_capture_stop") {
                append_daemon_event(config, "loop_failed_leave_capture_running", &err);
                let _ = execute(
                    &[Command::ReleaseAll],
                    &mut EventLog::from_optional_path(None)?,
                );
            } else {
                append_daemon_event(config, "loop_failed_stop_capture", &err);
                let wait_ms = request_params
                    .get("stop_wait_ms")
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(2000);
                let _ = capture_stop(config, wait_ms);
            }
            Err(err)
        }
    }
}

fn experiment_loop_inner(
    config: &DaemonConfig,
    request_params: &BTreeMap<String, String>,
    guard: &ExperimentRunGuard,
) -> Result<String, String> {
    let mut params = request_params.clone();
    let cancel_path = config.dynout.join("experiment_cancel.flag");
    if cancel_path.exists() {
        fs::remove_file(&cancel_path).map_err(|err| {
            format!(
                "remove stale experiment cancel flag {}: {err}",
                cancel_path.display()
            )
        })?;
    }
    params.insert("cancel_file".to_string(), cancel_path.display().to_string());
    let run_id = ensure_experiment_run_id(&mut params);
    let scenario = params
        .get("scenario")
        .map(String::as_str)
        .unwrap_or("http_loop");
    let expected_map = params
        .get("expected_map")
        .map(String::as_str)
        .unwrap_or("unknown");
    let start_ms = params
        .get("start_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(90);
    let ready_timeout_ms = params
        .get("ready_timeout_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(15000);
    let active_timeout_ms = params
        .get("active_timeout_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(45000);
    let post_drive_wait_ms = params
        .get("post_drive_wait_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1000);
    let return_timeout_ms = params
        .get("return_timeout_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120000);
    let return_timeout_policy = params
        .get("return_ready_timeout_policy")
        .map(String::as_str)
        .unwrap_or("fail");
    let stop_wait_ms = params
        .get("stop_wait_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2000);
    let mut package = params_bool(&params, "package");
    let no_capture_stop = params_bool(&params, "no_capture_stop");
    let drive = !params_bool(&params, "no_drive");
    let wait_return_ready = params_bool(&params, "wait_return_ready");
    let drive_settle_ms = params
        .get("drive_settle_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1000);
    let mut steps = Vec::new();

    append_daemon_event(
        config,
        "loop_start",
        &format!("scenario={scenario} map={expected_map}"),
    );
    let _ = write_experiment_manifest(config, scenario, expected_map, &params, &steps, "starting");
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    let start_result = capture_start(config, &params)?;
    steps.push(format!(
        "{{\"step\":\"capture_start\",\"ok\":true,\"result\":{}}}",
        start_result
    ));
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "capture_started",
    );
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;

    let initial_state = current_live_state(config);
    if initial_state.fresh
        && initial_state.truth_current
        && is_terminal_nonstart_phase(&initial_state.phase)
    {
        let err = format!("terminal_nonstart_phase:{}", initial_state.phase);
        append_daemon_event(config, "loop_terminal_nonstart_phase", &err);
        steps.push(format!(
            "{{\"step\":\"wait_ready_or_active\",\"ok\":false,\"error\":\"{}\",\"state\":{}}}",
            json_escape(&err),
            initial_state.to_json()
        ));
        let _ =
            write_experiment_manifest(config, scenario, expected_map, &params, &steps, "failed");
        return Err(format!(
            "{{\"steps\":[{}],\"error\":\"{}\"}}",
            steps.join(","),
            json_escape(&err)
        ));
    }

    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "waiting_ready_or_active",
    );
    let ready_state = wait_for_live(config, ready_timeout_ms, 250, |state| {
        state.fresh && (is_startable_phase(&state.phase) || state.control_allowed)
    })?;
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    steps.push(format!(
        "{{\"step\":\"wait_ready_or_active\",\"ok\":true,\"state\":{}}}",
        ready_state.to_json()
    ));
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "ready_or_active_seen",
    );

    let start_wait_ms = params
        .get("start_wait_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(active_timeout_ms.min(15000));
    let start_retry_interval_ms = params
        .get("start_retry_interval_ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(3000);
    let start_if_needed = control_start_if_needed(config, start_ms, start_wait_ms)?;
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    steps.push(format!(
        "{{\"step\":\"start_if_needed\",\"ok\":true,\"result\":{}}}",
        start_if_needed
    ));
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "start_if_needed_done",
    );

    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "waiting_game_active",
    );
    let (active_state, start_retry_count) = wait_for_game_active_with_start_retries(
        config,
        active_timeout_ms,
        250,
        start_ms,
        start_wait_ms,
        start_retry_interval_ms,
    )?;
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    steps.push(format!(
        "{{\"step\":\"wait_game_active\",\"ok\":true,\"message\":\"game_active with pose\",\"start_retries\":{},\"state\":{}}}",
        start_retry_count,
        active_state.to_json()
    ));
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "game_active_seen",
    );

    if drive {
        let _ =
            write_experiment_manifest(config, scenario, expected_map, &params, &steps, "driving");
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        let drive_before = current_live_state(config);
        let pre_drive_self_max_speed = params
            .get("pre_drive_max_speed")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.75);
        let pre_drive_self_timeout_ms = params
            .get("pre_drive_self_timeout_ms")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(15000);
        let drive_before_self = match wait_for_self_drive_gate(
            config,
            pre_drive_self_timeout_ms,
            100,
            pre_drive_self_max_speed,
        ) {
            Ok(state) => state,
            Err(err) => {
                append_daemon_event(config, "loop_drive_self_gate_failed", &err);
                steps.push(format!(
                    "{{\"step\":\"pre_drive_self_gate\",\"ok\":false,\"max_speed\":{},\"timeout_ms\":{},\"error\":\"{}\"}}",
                    pre_drive_self_max_speed,
                    pre_drive_self_timeout_ms,
                    json_escape(&err)
                ));
                let _ = write_experiment_manifest(
                    config,
                    scenario,
                    expected_map,
                    &params,
                    &steps,
                    "failed",
                );
                return Err(format!(
                    "{{\"steps\":[{}],\"error\":\"{}\"}}",
                    steps.join(","),
                    json_escape(&err)
                ));
            }
        };
        steps.push(format!(
            "{{\"step\":\"pre_drive_self_gate\",\"ok\":true,\"max_speed\":{},\"self\":{}}}",
            pre_drive_self_max_speed,
            drive_before_self.to_json()
        ));
        if has_drive_surface_target(&params)
            && params
                .get("pre_drive_kap_pose_gate")
                .or_else(|| params.get("pre_drive_kap_map_gate"))
                .map(|value| value != "0" && value != "false")
                .unwrap_or(true)
        {
            match pre_drive_kap_pose_gate(config, &params, &drive_before_self) {
                Ok(gate_json) => {
                    steps.push(format!(
                        "{{\"step\":\"pre_drive_kap_pose_gate\",\"ok\":true,\"result\":{}}}",
                        gate_json
                    ));
                }
                Err(err) => {
                    append_daemon_event(config, "loop_drive_kap_pose_gate_failed", &err);
                    steps.push(format!(
                        "{{\"step\":\"pre_drive_kap_pose_gate\",\"ok\":false,\"error\":\"{}\"}}",
                        json_escape(&err)
                    ));
                    let _ = write_experiment_manifest(
                        config,
                        scenario,
                        expected_map,
                        &params,
                        &steps,
                        "failed",
                    );
                    return Err(format!(
                        "{{\"steps\":[{}],\"error\":\"{}\"}}",
                        steps.join(","),
                        json_escape(&err)
                    ));
                }
            }
        }
        let _ = write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "pre_drive_self_ready",
        );
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        let focus = focus_kart_rider_retry(2_000);
        let drive_mode = params
            .get("drive_mode")
            .map(String::as_str)
            .unwrap_or("auto");
        let use_surface_drive = match drive_mode {
            "surface" => true,
            "basic" => false,
            "auto" | "" => has_drive_surface_target(&params),
            other => {
                let err = format!("unknown drive_mode: {other}");
                append_daemon_event(config, "loop_drive_failed", &err);
                steps.push(format!(
                    "{{\"step\":\"drive\",\"ok\":false,\"error\":\"{}\"}}",
                    json_escape(&err)
                ));
                let _ = write_experiment_manifest(
                    config,
                    scenario,
                    expected_map,
                    &params,
                    &steps,
                    "failed",
                );
                return Err(format!(
                    "{{\"steps\":[{}],\"error\":\"{}\"}}",
                    steps.join(","),
                    json_escape(&err)
                ));
            }
        };
        append_daemon_event(
            config,
            "loop_drive_start",
            &format!("focus={focus} mode={drive_mode} surface={use_surface_drive}"),
        );
        steps.push(format!(
            "{{\"step\":\"drive_start\",\"ok\":true,\"focus\":{},\"mode\":\"{}\",\"surface\":{},\"self\":{}}}",
            bool_json(focus),
            json_escape(drive_mode),
            bool_json(use_surface_drive),
            current_self_state(config).to_json()
        ));
        let _ = write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "drive_running",
        );
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        let drive_input_backend = params
            .get("drive_input_backend")
            .or_else(|| params.get("input_backend"))
            .map(String::as_str)
            .unwrap_or("sendinput");
        let drive_input_requires_focus =
            InputBackend::parse(drive_input_backend)?.requires_foreground();
        if use_surface_drive && !focus && drive_input_requires_focus {
            let warn = "focus false before surface drive; attempting foreground input and relying on motion evidence";
            append_daemon_event(config, "loop_drive_focus_warning", warn);
            steps.push(format!(
                "{{\"step\":\"drive_focus_warning\",\"ok\":true,\"focus\":false,\"input_backend\":\"{}\",\"warning\":\"{}\"}}",
                json_escape(drive_input_backend),
                json_escape(warn)
            ));
        }
        let drive_result = if use_surface_drive {
            let mut drive_params = params.clone();
            let control_log_path = config.dynout.join("probe_control.tsv");
            let control_log_start_rows = control_log_data_rows(&control_log_path);
            // KAP must follow the pose writer, not the owner-candidate scan.
            // probe_owner.tsv can contain several kart-like candidates at the
            // same tick and its last/highest-scored row may change after a
            // reset or local recovery. probe_pose.tsv is the continuously
            // integrated writer for the active physics object and therefore
            // remains the route-control authority.
            drive_params
                .entry("pose".to_string())
                .or_insert_with(|| config.dynout.join("probe_pose.tsv").display().to_string());
            drive_params
                .entry("collision".to_string())
                .or_insert_with(|| {
                    config
                        .dynout
                        .join("probe_collision_face.tsv")
                        .display()
                        .to_string()
                });
            drive_params
                .entry("log".to_string())
                .or_insert_with(|| control_log_path.display().to_string());
            if params.contains_key("drive_max_ms") && !drive_params.contains_key("max_ms") {
                if let Some(value) = params.get("drive_max_ms") {
                    drive_params.insert("max_ms".to_string(), value.clone());
                }
            }
            match daemon_drive_surface_args(config, &drive_params) {
                Ok(mut drive_args) => {
                    if params_bool(&drive_params, "seed_pose_from_self")
                        || !drive_params.contains_key("seed_pose_from_self")
                    {
                        let pose_path = PathBuf::from(
                            drive_params
                                .get("pose")
                                .ok_or("drive-surface missing pose for seed_pose_from_self")?,
                        );
                        if let Err(err) = seed_pose_tsv_from_self_if_needed(
                            config,
                            &pose_path,
                            &drive_before_self,
                        ) {
                            append_daemon_event(config, "loop_drive_pose_seed_failed", &err);
                        }
                    }
                    let result = run_drive_surface(&drive_args);
                    drive_args.clear();
                    result.map(|_| {
                        let (keys, duration_ms) =
                            surface_drive_control_delta(&control_log_path, control_log_start_rows);
                        ("drive_surface", keys, duration_ms)
                    })
                }
                Err(err) => Err(err),
            }
        } else {
            let keys = basic_drive_keys(&params)?;
            let duration_ms = basic_drive_duration_ms(&params);
            let log_path = config.dynout.join("probe_control.tsv");
            execute(
                &[Command::Hold {
                    keys: keys.clone(),
                    duration_ms,
                }],
                &mut EventLog::from_optional_path(Some(log_path))?,
            )
            .map(|_| ("drive_basic", keys, duration_ms))
        };
        let _ = execute(
            &[Command::ReleaseAll],
            &mut EventLog::from_optional_path(None)?,
        );
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        let drive_after = current_live_state(config);
        let drive_after_self = current_self_state(config);
        let drive_result = match drive_result {
            Ok(_) if drive_after.phase == "no_process" => Err(format!(
                "control_authority_lost_during_drive:phase={} reason={}",
                drive_after.phase, drive_after.control_reason
            )),
            other => other,
        };
        let (drive_settled, drive_settled_observed) = if drive_settle_ms > 0 {
            match wait_for_live(config, drive_settle_ms, 100, |state| {
                state.fresh
                    && state.truth_current
                    && !state.pose_summary.is_empty()
                    && state.pose_summary != drive_after.pose_summary
            }) {
                Ok(state) => (state, true),
                Err(_) => (current_live_state(config), false),
            }
        } else {
            (drive_after.clone(), false)
        };
        match drive_result {
            Ok((step_name, keys, duration_ms)) => {
                append_control_evidence(
                    config,
                    step_name,
                    &keys,
                    duration_ms,
                    focus,
                    &drive_before,
                    &drive_after,
                    &drive_before_self,
                    &drive_after_self,
                    "ok",
                );
                append_control_motion_evidence(
                    config,
                    step_name,
                    &keys,
                    duration_ms,
                    &drive_before,
                    &drive_after,
                    &drive_settled,
                    drive_settled_observed,
                );
                append_daemon_event(
                    config,
                    "loop_drive_done",
                    &format!(
                        "ok=1 mode={step_name} duration_ms={duration_ms} settled_observed={drive_settled_observed}"
                    ),
                );
                let distance =
                    pose_distance(&drive_before.pose_summary, &drive_settled.pose_summary)
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_default();
                steps.push(format!(
                    "{{\"step\":\"{}\",\"ok\":true,\"focus\":{},\"duration_ms\":{},\"settled_observed\":{},\"distance_before_settled\":\"{}\",\"settled_state\":{}}}",
                    step_name,
                    bool_json(focus),
                    duration_ms,
                    bool_json(drive_settled_observed),
                    json_escape(&distance),
                    drive_settled.to_json()
                ));
            }
            Err(err) => {
                append_control_evidence(
                    config,
                    "loop_drive",
                    &[],
                    0,
                    focus,
                    &drive_before,
                    &drive_after,
                    &drive_before_self,
                    &drive_after_self,
                    &err,
                );
                append_daemon_event(config, "loop_drive_failed", &err);
                steps.push(format!(
                    "{{\"step\":\"drive\",\"ok\":false,\"focus\":{},\"error\":\"{}\"}}",
                    bool_json(focus),
                    json_escape(&err)
                ));
                let _ = write_experiment_manifest(
                    config,
                    scenario,
                    expected_map,
                    &params,
                    &steps,
                    "failed",
                );
                return Err(format!(
                    "{{\"steps\":[{}],\"error\":\"{}\"}}",
                    steps.join(","),
                    json_escape(&err)
                ));
            }
        }
    } else {
        steps.push("{\"step\":\"drive_surface\",\"ok\":true,\"skipped\":true}".to_string());
    }
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "drive_done",
    );
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;

    if post_drive_wait_ms > 0 {
        thread::sleep(Duration::from_millis(post_drive_wait_ms));
    }
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    if wait_return_ready {
        let _ = write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "waiting_return_ready",
        );
        append_daemon_event(
            config,
            "loop_wait_return_ready",
            &format!("timeout_ms={return_timeout_ms}"),
        );
        let mut return_ready_observed = true;
        let return_state = match wait_for_live_or_cancel(
            config,
            return_timeout_ms,
            500,
            guard,
            |state| {
                state.fresh
                    && state.truth_current
                    && is_room_wait_phase(&state.phase)
                    && !state.control_allowed
            },
        ) {
            Ok(state) => state,
            Err(err) if err == "experiment_cancelled" => {
                check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
                unreachable!("cancelled experiment check must return")
            }
            Err(err) => {
                append_daemon_event(config, "loop_return_ready_failed", &err);
                if matches!(return_timeout_policy, "package" | "stop" | "continue") {
                    steps.push(format!(
                        "{{\"step\":\"wait_return_ready\",\"ok\":false,\"timeout_policy\":\"{}\",\"error\":\"{}\"}}",
                        json_escape(return_timeout_policy),
                        json_escape(&err)
                    ));
                    let _ = write_experiment_manifest(
                        config,
                        scenario,
                        expected_map,
                        &params,
                        &steps,
                        "return_ready_timeout",
                    );
                    append_daemon_event(
                        config,
                        "loop_return_ready_timeout_continue",
                        &format!("policy={return_timeout_policy}"),
                    );
                    if return_timeout_policy == "stop" {
                        package = false;
                    }
                    return_ready_observed = false;
                    current_live_state(config)
                } else {
                    steps.push(format!(
                        "{{\"step\":\"wait_return_ready\",\"ok\":false,\"error\":\"{}\"}}",
                        json_escape(&err)
                    ));
                    let _ = write_experiment_manifest(
                        config,
                        scenario,
                        expected_map,
                        &params,
                        &steps,
                        "failed",
                    );
                    return Err(format!(
                        "{{\"steps\":[{}],\"error\":\"{}\"}}",
                        steps.join(","),
                        json_escape(&err)
                    ));
                }
            }
        };
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        if return_ready_observed {
            append_daemon_event(
                config,
                "loop_return_ready",
                &format!("phase={}", return_state.phase),
            );
            steps.push(format!(
                "{{\"step\":\"wait_return_ready\",\"ok\":true,\"state\":{}}}",
                return_state.to_json()
            ));
        }
    } else {
        steps.push("{\"step\":\"wait_return_ready\",\"ok\":true,\"skipped\":true}".to_string());
    }
    let _ = write_experiment_manifest(
        config,
        scenario,
        expected_map,
        &params,
        &steps,
        "return_ready_done",
    );
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    if no_capture_stop {
        execute(
            &[Command::ReleaseAll],
            &mut EventLog::from_optional_path(None)?,
        )?;
        append_daemon_event(
            config,
            "loop_capture_stop_skipped",
            "reason=no_capture_stop",
        );
        steps.push("{\"step\":\"capture_stop\",\"ok\":true,\"skipped\":true,\"reason\":\"no_capture_stop\"}".to_string());
        let _ = write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "capture_left_running",
        );
    } else {
        let _ =
            write_experiment_manifest(config, scenario, expected_map, &params, &steps, "stopping");
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        let stop_result = capture_stop(config, stop_wait_ms)?;
        steps.push(format!(
            "{{\"step\":\"capture_stop\",\"ok\":true,\"result\":{}}}",
            stop_result
        ));
        let _ = write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "capture_stopped",
        );
    }

    if package {
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        write_experiment_manifest(
            config,
            scenario,
            expected_map,
            &params,
            &steps,
            "pre_package",
        )?;
        let package_result = capture_package(config, &params)?;
        check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
        steps.push(format!(
            "{{\"step\":\"capture_package\",\"ok\":true,\"result\":{}}}",
            package_result
        ));
        let _ =
            write_experiment_manifest(config, scenario, expected_map, &params, &steps, "packaged");
    }
    append_daemon_event(
        config,
        "loop_done",
        &format!("scenario={scenario} map={expected_map}"),
    );
    check_experiment_cancelled(config, scenario, expected_map, &params, &steps, guard)?;
    let _ = write_experiment_manifest(config, scenario, expected_map, &params, &steps, "done");
    Ok(format!(
        "{{\"run_id\":\"{}\",\"scenario\":\"{}\",\"expected_map\":\"{}\",\"steps\":[{}],\"final_state\":{}}}",
        json_escape(&run_id),
        json_escape(scenario),
        json_escape(expected_map),
        steps.join(","),
        daemon_state_json(config)
    ))
}

fn write_result_json(
    stream: &mut TcpStream,
    result: Result<String, String>,
    op: &str,
) -> Result<(), String> {
    match result {
        Ok(extra) => write_http_json(
            stream,
            200,
            &format!("{{\"ok\":true,\"op\":\"{}\",\"result\":{}}}", op, extra),
        ),
        Err(err) => write_http_json(
            stream,
            500,
            &format!(
                "{{\"ok\":false,\"op\":\"{}\",\"error\":\"{}\"}}",
                op,
                json_escape(&err)
            ),
        ),
    }
}

fn run_powershell(script: &str) -> Result<String, String> {
    let output = ProcessCommand::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output()
        .map_err(|err| format!("launch powershell: {err}"))?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        Ok(text)
    } else {
        Err(format!(
            "powershell exit {:?}: {text}",
            output.status.code()
        ))
    }
}

fn ps_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn ps_quote_path(path: &Path) -> String {
    ps_quote(&path.display().to_string())
}

fn slug(value: &str) -> String {
    let mut out = String::new();
    let mut last_us = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' {
            out.push(ch);
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn has_tsv_data_row(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    BufReader::new(file).lines().take(2).count() >= 2
}

fn split_query(target: &str) -> (&str, &str) {
    match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    }
}

fn parse_query(query: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for part in query.split('&') {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        out.insert(percent_decode(key), percent_decode(value));
    }
    out
}

#[allow(dead_code)]
fn parse_u64_query(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = u8::from_str_radix(&value[i + 1..i + 3], 16) {
                out.push(hex);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn url_encode(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

#[cfg(test)]
mod resource_projection_tests {
    use super::*;

    fn status(fresh: bool) -> FileStatus {
        FileStatus {
            exists: true,
            bytes: 100,
            age_ms: Some(1),
            fresh,
        }
    }

    fn row(values: &[(&str, &str)]) -> BTreeMap<String, String> {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    #[test]
    fn post_event_projects_structured_inventory() {
        let value = item_resource_items_json(
            &status(true),
            &row(&[
                ("phase", "1"),
                ("entry_count", "2"),
                ("item0", "6"),
                ("quantity0", "2"),
                ("item1", "31"),
                ("quantity1", "1"),
            ]),
        );
        assert_eq!(
            value,
            "[{\"item_id\":6,\"quantity\":2},{\"item_id\":31,\"quantity\":1}]"
        );
    }

    #[test]
    fn stale_or_pre_event_inventory_remains_unknown() {
        let post = row(&[("phase", "1"), ("entry_count", "0")]);
        assert_eq!(item_resource_items_json(&status(false), &post), "null");
        let pre = row(&[("phase", "0"), ("entry_count", "0")]);
        assert_eq!(item_resource_items_json(&status(true), &pre), "null");
    }

    #[test]
    fn fresh_post_empty_inventory_is_known_empty() {
        let post = row(&[("phase", "1"), ("entry_count", "0")]);
        assert_eq!(item_resource_items_json(&status(true), &post), "[]");
    }

    #[test]
    fn live_snapshot_self_fields_preserve_float_pose_and_runtime_selectors() {
        let fields = live_snapshot_self_fields(
            r#"{"owner":{"object":"0x02E554008"},"pose":{"present":1,"object":"0x2e554008","x":1010.5,"y":-2.25,"z":5.7e1,"yaw":-2.3e-5,"speed":9.25,"vel_x":8,"vel_y":1.5,"vel_z":-0.01},"physics":{"surface_mode":1,"b1c":0,"rail_state":3}}"#,
        );
        assert_eq!(
            fields.get("owner_this").map(String::as_str),
            Some("0x02E554008")
        );
        assert_eq!(
            fields.get("pose_this").map(String::as_str),
            Some("0x2e554008")
        );
        assert_eq!(fields.get("x").map(String::as_str), Some("1010.5"));
        assert_eq!(fields.get("z").map(String::as_str), Some("5.7e1"));
        assert_eq!(fields.get("yaw").map(String::as_str), Some("-2.3e-5"));
        assert_eq!(fields.get("rail_state").map(String::as_str), Some("3"));
        assert!(pointer_identity_eq("0x02E554008", "0x2e554008"));
        assert!(!pointer_identity_eq("0x02E554008", "0x2e554108"));
        assert!(!pointer_identity_eq("", ""));
    }

    #[test]
    fn absent_live_snapshot_pose_does_not_publish_partial_self_state() {
        assert!(live_snapshot_self_fields(
            r#"{"pose":{"present":0,"x":0,"y":0,"z":0},"physics":{"rail_state":0}}"#
        )
        .is_empty());
    }

    #[test]
    fn live_active_game_is_runtime_d3d_override() {
        assert_eq!(
            effective_d3d9_ready(false, true, true, true, "game_active"),
            (true, true)
        );
    }

    #[test]
    fn stale_state_cannot_override_d3d_probe_but_fresh_ready_can() {
        assert_eq!(
            effective_d3d9_ready(false, true, false, true, "game_active"),
            (false, false)
        );
        assert_eq!(
            effective_d3d9_ready(false, true, true, true, "ready_waiting_start"),
            (true, true)
        );
    }

    #[test]
    fn input_security_detects_uipi_integrity_mismatch() {
        assert_eq!(
            input_security_diagnosis(0x2000, 0x3000, "Default", true),
            ("daemon_integrity_below_game_uipi_blocks_sendinput", false)
        );
        assert_eq!(
            input_security_diagnosis(0x3000, 0x3000, "Default", true),
            ("integrity_levels_allow_sendinput", true)
        );
        assert_eq!(
            input_security_diagnosis(0x3000, 0x3000, "", false),
            ("no_active_input_desktop", false)
        );
    }

    #[test]
    fn empty_track_search_grid_is_rejected_before_clicking_confirm() {
        assert!(!search_result_grid_is_populated(0, 342));
        assert!(!search_result_grid_is_populated(3, 342));
        assert!(!search_result_grid_is_populated(17, 342));
    }

    #[test]
    fn populated_track_search_grid_is_accepted() {
        assert!(search_result_grid_is_populated(18, 342));
        assert!(search_result_grid_is_populated(137, 342));
    }

    #[test]
    fn only_fresh_active_experiment_manifest_blocks_start_or_preflight() {
        assert!(experiment_manifest_blocks_start(true, "starting"));
        assert!(experiment_manifest_blocks_start(true, "driving"));
        assert!(!experiment_manifest_blocks_start(false, "starting"));
        assert!(!experiment_manifest_blocks_start(true, "done"));
        assert!(!experiment_manifest_blocks_start(false, "done"));
    }

    #[test]
    fn surface_drive_forwards_experiment_cancel_file() {
        let config = DaemonConfig::default();
        let mut params = BTreeMap::new();
        params.insert("track".to_string(), r"D:\kr\track.1s".to_string());
        params.insert("kap_dir".to_string(), r"D:\kr\aiPath".to_string());
        params.insert(
            "cancel_file".to_string(),
            r"D:\kr\dynout\experiment_cancel.flag".to_string(),
        );
        let args = daemon_drive_surface_args(&config, &params).unwrap();
        let index = args
            .iter()
            .position(|value| value == "--cancel-file")
            .unwrap();
        assert_eq!(args[index + 1], r"D:\kr\dynout\experiment_cancel.flag");
    }

    #[test]
    fn live_map_prefers_runtime_snapshot_over_context_and_expected() {
        assert_eq!(
            resolve_live_map_identity(
                "fairy_I03",
                "",
                "china_I03",
                "",
                "declared_capture_expected_map",
                "abyss_I03",
                ""
            ),
            (
                "fairy_I03".to_string(),
                "".to_string(),
                "live_snapshot".to_string(),
                "runtime".to_string()
            )
        );
    }

    #[test]
    fn live_map_does_not_resolve_pid_validated_declared_context() {
        assert_eq!(
            resolve_live_map_identity(
                "",
                "",
                "fairy_I03",
                "童话世界 旋转木马",
                "declared_capture_expected_map",
                "unknown",
                ""
            ),
            (
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "unknown".to_string()
            )
        );
    }

    #[test]
    fn live_map_does_not_resolve_scenario_expected_map() {
        assert_eq!(
            resolve_live_map_identity("", "", "", "", "", "china_I03", "中国主题"),
            (
                "".to_string(),
                "".to_string(),
                "".to_string(),
                "unknown".to_string()
            )
        );
    }

    #[test]
    fn live_map_runtime_context_is_strong_after_probe_restart() {
        let result = resolve_live_map_identity(
            "",
            "",
            "fairy_I03",
            "",
            "runtime_file_open_CreateFileW",
            "unknown",
            "",
        );
        assert_eq!(result.0, "fairy_I03");
        assert_eq!(result.3, "runtime");
    }

    #[test]
    fn live_map_accepts_strong_pose_to_kap_runtime_context() {
        let result = resolve_live_map_identity(
            "",
            "",
            "factory_I02",
            "工厂 巨型齿轮",
            "runtime_pose_to_kap_strong",
            "unknown",
            "",
        );
        assert_eq!(result.0, "factory_I02");
        assert_eq!(result.1, "工厂 巨型齿轮");
        assert_eq!(result.2, "runtime_pose_to_kap_strong");
        assert_eq!(result.3, "runtime");
    }

    #[test]
    fn live_map_accepts_operator_confirmation_without_calling_it_runtime() {
        let result = resolve_live_map_identity(
            "",
            "",
            "fairy_I03",
            "童话世界 旋转木马",
            "operator_confirmed",
            "unknown",
            "",
        );
        assert_eq!(result.0, "fairy_I03");
        assert_eq!(result.1, "童话世界 旋转木马");
        assert_eq!(result.2, "operator_confirmed");
        assert_eq!(result.3, "operator_confirmed");
    }

    #[test]
    fn pose_seed_header_matches_current_probe_pose_epoch_schema() {
        let columns: Vec<_> = pose_tsv_header().trim_end().split('\t').collect();
        assert_eq!(columns[2], "physics_substep_id");
        assert_eq!(columns[3], "linear_write_epoch");
        assert_eq!(columns[4], "site_rva");
        assert_eq!(columns[17], "post_pos_x");
    }

    #[test]
    fn owner_matrix_forward_yaw_uses_negative_column_one() {
        let mut owner = BTreeMap::new();
        owner.insert("m01".to_string(), "1".to_string());
        owner.insert("m11".to_string(), "0".to_string());
        let yaw = owner_forward_yaw(&owner).unwrap();
        assert!((yaw.abs() - std::f32::consts::PI).abs() < 1e-5);
    }
}
