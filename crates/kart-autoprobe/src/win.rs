use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProcessIntegrityLevel {
    pub(crate) rid: u32,
    pub(crate) label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InputDesktopContext {
    pub(crate) window_station: String,
    pub(crate) daemon_desktop: String,
    pub(crate) input_desktop: String,
    pub(crate) game_desktop: String,
    pub(crate) foreground_present: bool,
    pub(crate) input_struct_bytes: usize,
}

#[cfg(not(windows))]
pub(crate) fn input_desktop_context() -> Result<InputDesktopContext, String> {
    Err("input desktop context is only available on Windows".to_string())
}

#[cfg(windows)]
pub(crate) fn input_desktop_context() -> Result<InputDesktopContext, String> {
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::StationsAndDesktops::{
        CloseDesktop, GetProcessWindowStation, GetThreadDesktop, GetUserObjectInformationW,
        OpenInputDesktop, DESKTOP_READOBJECTS, UOI_NAME,
    };
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FindWindowW, GetForegroundWindow, GetWindowThreadProcessId,
    };

    unsafe fn object_name(handle: HANDLE) -> String {
        if handle.is_null() {
            return String::new();
        }
        let mut needed = 0_u32;
        GetUserObjectInformationW(handle, UOI_NAME, std::ptr::null_mut(), 0, &mut needed);
        if needed < 2 {
            return String::new();
        }
        let mut value = vec![0_u16; (needed as usize + 1) / 2];
        if GetUserObjectInformationW(
            handle,
            UOI_NAME,
            value.as_mut_ptr().cast(),
            needed,
            &mut needed,
        ) == 0
        {
            return String::new();
        }
        let len = value
            .iter()
            .position(|unit| *unit == 0)
            .unwrap_or(value.len());
        String::from_utf16_lossy(&value[..len])
    }

    let title = "PopKart Client"
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    unsafe {
        let window_station = object_name(GetProcessWindowStation().cast());
        let daemon_desktop = object_name(GetThreadDesktop(GetCurrentThreadId()).cast());
        let input = OpenInputDesktop(0, 0, DESKTOP_READOBJECTS);
        let input_desktop = object_name(input.cast());
        if !input.is_null() {
            CloseDesktop(input);
        }
        let hwnd = FindWindowW(std::ptr::null(), title.as_ptr());
        let game_thread = if hwnd.is_null() {
            0
        } else {
            GetWindowThreadProcessId(hwnd, std::ptr::null_mut())
        };
        let game_desktop = if game_thread == 0 {
            String::new()
        } else {
            object_name(GetThreadDesktop(game_thread).cast())
        };
        Ok(InputDesktopContext {
            window_station,
            daemon_desktop,
            input_desktop,
            game_desktop,
            foreground_present: !GetForegroundWindow().is_null(),
            input_struct_bytes: std::mem::size_of::<INPUT>(),
        })
    }
}

#[cfg(not(windows))]
pub(crate) fn process_integrity_level(_pid: Option<u32>) -> Result<ProcessIntegrityLevel, String> {
    Err("process integrity is only available on Windows".to_string())
}

#[cfg(windows)]
pub(crate) fn process_integrity_level(pid: Option<u32>) -> Result<ProcessIntegrityLevel, String> {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
    use windows_sys::Win32::Security::{
        GetSidSubAuthority, GetSidSubAuthorityCount, GetTokenInformation, TokenIntegrityLevel,
        TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let (process, close_process): (HANDLE, bool) = match pid {
        Some(value) => {
            let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, value) };
            if handle.is_null() {
                return Err(format!("OpenProcess({value}) failed: {}", unsafe {
                    GetLastError()
                }));
            }
            (handle, true)
        }
        None => (unsafe { GetCurrentProcess() }, false),
    };
    let mut token: HANDLE = std::ptr::null_mut();
    let opened = unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) };
    if close_process {
        unsafe { CloseHandle(process) };
    }
    if opened == 0 || token.is_null() {
        return Err(format!("OpenProcessToken failed: {}", unsafe {
            GetLastError()
        }));
    }

    let mut needed = 0_u32;
    unsafe {
        GetTokenInformation(
            token,
            TokenIntegrityLevel,
            std::ptr::null_mut(),
            0,
            &mut needed,
        );
    }
    if needed < std::mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32 {
        unsafe { CloseHandle(token) };
        return Err(format!("invalid TokenIntegrityLevel size: {needed}"));
    }
    let word = std::mem::size_of::<usize>();
    let mut storage = vec![0_usize; (needed as usize + word - 1) / word];
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenIntegrityLevel,
            storage.as_mut_ptr().cast(),
            needed,
            &mut needed,
        )
    };
    unsafe { CloseHandle(token) };
    if ok == 0 {
        return Err(format!("GetTokenInformation failed: {}", unsafe {
            GetLastError()
        }));
    }
    let mandatory = unsafe { &*(storage.as_ptr().cast::<TOKEN_MANDATORY_LABEL>()) };
    let sid = mandatory.Label.Sid;
    if sid.is_null() {
        return Err("TokenIntegrityLevel returned a null SID".to_string());
    }
    let count = unsafe { *GetSidSubAuthorityCount(sid) } as u32;
    if count == 0 {
        return Err("TokenIntegrityLevel SID has no subauthorities".to_string());
    }
    let rid = unsafe { *GetSidSubAuthority(sid, count - 1) };
    let label = match rid {
        0..=0x0fff => "untrusted",
        0x1000..=0x1fff => "low",
        0x2000..=0x2fff => "medium",
        0x3000..=0x3fff => "high",
        0x4000..=0x4fff => "system",
        _ => "protected",
    };
    Ok(ProcessIntegrityLevel { rid, label })
}

pub(crate) struct InputDriver {
    backend: InputBackend,
    held: Vec<Key>,
}

impl InputDriver {
    pub(crate) fn new(backend: InputBackend) -> Self {
        Self {
            backend,
            held: Vec::new(),
        }
    }

    pub(crate) fn key_down_many(&mut self, keys: &[Key]) -> Result<(), String> {
        for key in keys {
            self.key_event(*key, true)?;
        }
        Ok(())
    }

    pub(crate) fn key_up_many(&mut self, keys: &[Key]) -> Result<(), String> {
        for key in keys.iter().rev() {
            self.key_event(*key, false)?;
        }
        Ok(())
    }

    pub(crate) fn release_all(&mut self) -> Result<(), String> {
        if self.backend == InputBackend::InternalSetters {
            self.held.clear();
            return write_internal_input_command(&self.held);
        }
        self.key_up_many(&[
            Key::W,
            Key::A,
            Key::S,
            Key::D,
            Key::V,
            Key::Up,
            Key::Down,
            Key::Left,
            Key::Right,
            Key::Space,
            Key::Shift,
            Key::Ctrl,
            Key::R,
            Key::Esc,
            Key::Tab,
            Key::Enter,
            Key::F5,
        ])
    }

    fn key_event(&mut self, key: Key, down: bool) -> Result<(), String> {
        match self.backend {
            InputBackend::SendInput => {
                platform_key_event(key, down)?;
                self.track_held(key, down);
                Ok(())
            }
            InputBackend::PostMessage => {
                post_key_event_to_kart_rider(key, down)?;
                self.track_held(key, down);
                Ok(())
            }
            InputBackend::InternalSetters => {
                if down && !internal_setters_supports(key) {
                    return Err(format!(
                        "internal-setters backend cannot press {}; only arrow keys are supported",
                        key.name()
                    ));
                }
                self.track_held(key, down);
                write_internal_input_command(&self.held)
            }
        }
    }

    fn track_held(&mut self, key: Key, down: bool) {
        if down {
            if !self.held.contains(&key) {
                self.held.push(key);
            }
        } else {
            self.held.retain(|held| *held != key);
        }
    }
}

impl Drop for InputDriver {
    fn drop(&mut self) {
        if !self.held.is_empty() {
            let _ = self.release_all();
        }
    }
}

fn internal_setters_supports(key: Key) -> bool {
    matches!(key, Key::Up | Key::Down | Key::Left | Key::Right)
}

fn write_internal_input_command(keys: &[Key]) -> Result<(), String> {
    let names = keys
        .iter()
        .filter_map(|key| internal_setters_supports(*key).then_some(key.name()))
        .collect::<Vec<_>>()
        .join("+");
    let body = format!("keys={names}\n");
    fs::write(r"D:\kr\dynout\probe_internal_input.cmd", body)
        .map_err(|err| format!("write internal input command: {err}"))
}

impl Key {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Key::W => "W",
            Key::A => "A",
            Key::S => "S",
            Key::D => "D",
            Key::V => "V",
            Key::Up => "Up",
            Key::Down => "Down",
            Key::Left => "Left",
            Key::Right => "Right",
            Key::Space => "Space",
            Key::Shift => "Shift",
            Key::Ctrl => "Ctrl",
            Key::R => "R",
            Key::Esc => "Esc",
            Key::Tab => "Tab",
            Key::Enter => "Enter",
            Key::F5 => "F5",
        }
    }

    #[cfg(windows)]
    fn vk_code(self) -> u16 {
        match self {
            Key::W => b'W' as u16,
            Key::A => b'A' as u16,
            Key::S => b'S' as u16,
            Key::D => b'D' as u16,
            Key::V => b'V' as u16,
            Key::Up => 0x26,
            Key::Down => 0x28,
            Key::Left => 0x25,
            Key::Right => 0x27,
            Key::Space => 0x20,
            Key::Shift => 0x10,
            Key::Ctrl => 0x11,
            Key::R => b'R' as u16,
            Key::Esc => 0x1B,
            Key::Tab => 0x09,
            Key::Enter => 0x0D,
            Key::F5 => 0x74,
        }
    }

    #[cfg(windows)]
    fn scan_code(self) -> u16 {
        match self {
            Key::Esc => 0x01,
            Key::W => 0x11,
            Key::A => 0x1e,
            Key::S => 0x1f,
            Key::D => 0x20,
            Key::V => 0x2f,
            Key::Up => 0x48,
            Key::Down => 0x50,
            Key::Left => 0x4b,
            Key::Right => 0x4d,
            Key::R => 0x13,
            Key::Ctrl => 0x1d,
            Key::Tab => 0x0f,
            Key::Enter => 0x1c,
            Key::Space => 0x39,
            Key::Shift => 0x2a,
            Key::F5 => 0x3f,
        }
    }

    #[cfg(windows)]
    fn is_extended(self) -> bool {
        matches!(self, Key::Up | Key::Down | Key::Left | Key::Right)
    }
}

#[cfg(not(windows))]
pub(crate) fn focus_kart_rider() -> bool {
    false
}

#[cfg(not(windows))]
pub(crate) fn post_key_to_kart_rider(_key: Key, _duration_ms: u64) -> bool {
    false
}

#[cfg(not(windows))]
pub(crate) fn launcher_post_key(_key: Key, _duration_ms: u64) -> Result<(), String> {
    Err("launcher window input is only available on Windows".to_string())
}

#[cfg(not(windows))]
pub(crate) fn clear_known_ui_blockers() -> Result<Vec<String>, String> {
    Err("ui blocker clearing is only available on Windows".to_string())
}

#[cfg(not(windows))]
fn post_key_event_to_kart_rider(_key: Key, _down: bool) -> Result<(), String> {
    Err("postmessage input is only available on Windows".to_string())
}

#[cfg(windows)]
pub(crate) fn focus_kart_rider() -> bool {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        keybd_event, SetActiveWindow, SetFocus, KEYEVENTF_KEYUP, VK_MENU,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, EnumWindows, FindWindowW, GetForegroundWindow, GetWindowTextLengthW,
        GetWindowTextW, GetWindowThreadProcessId, IsWindowVisible, SetForegroundWindow, ShowWindow,
        SW_RESTORE,
    };
    unsafe fn foreground_matches_target(hwnd: HWND) -> bool {
        let foreground = GetForegroundWindow();
        if foreground.is_null() {
            return false;
        }
        if foreground == hwnd {
            return true;
        }
        let mut target_pid = 0_u32;
        let mut foreground_pid = 0_u32;
        GetWindowThreadProcessId(hwnd, &mut target_pid);
        GetWindowThreadProcessId(foreground, &mut foreground_pid);
        target_pid != 0 && target_pid == foreground_pid
    }
    unsafe fn focus_hwnd(hwnd: HWND) -> bool {
        // SW_RESTORE both un-minimizes and shows; no need to branch on IsIconic.
        ShowWindow(hwnd, SW_RESTORE);
        keybd_event(VK_MENU as u8, 0, 0, 0);
        keybd_event(VK_MENU as u8, 0, KEYEVENTF_KEYUP, 0);
        BringWindowToTop(hwnd);
        if SetForegroundWindow(hwnd) != 0 && foreground_matches_target(hwnd) {
            return true;
        }

        let current_thread = GetCurrentThreadId();
        let target_thread = GetWindowThreadProcessId(hwnd, std::ptr::null_mut());
        let foreground = GetForegroundWindow();
        let foreground_thread = if foreground.is_null() {
            0
        } else {
            GetWindowThreadProcessId(foreground, std::ptr::null_mut())
        };
        let mut attached_target = false;
        let mut attached_foreground = false;
        if target_thread != 0 && target_thread != current_thread {
            attached_target = AttachThreadInput(current_thread, target_thread, 1) != 0;
        }
        if foreground_thread != 0
            && foreground_thread != current_thread
            && foreground_thread != target_thread
        {
            attached_foreground = AttachThreadInput(current_thread, foreground_thread, 1) != 0;
        }

        BringWindowToTop(hwnd);
        SetActiveWindow(hwnd);
        SetFocus(hwnd);
        let ok = SetForegroundWindow(hwnd) != 0 || foreground_matches_target(hwnd);

        if attached_foreground {
            AttachThreadInput(current_thread, foreground_thread, 0);
        }
        if attached_target {
            AttachThreadInput(current_thread, target_thread, 0);
        }
        ok || foreground_matches_target(hwnd)
    }

    for title in [
        "PopKart Client",
        "D3DProxyWindow",
        "KartRider",
        "跑跑卡丁车",
    ] {
        let wide = title
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let hwnd = unsafe { FindWindowW(std::ptr::null(), wide.as_ptr()) };
        if !hwnd.is_null() {
            unsafe {
                if focus_hwnd(hwnd) {
                    return true;
                }
            }
        }
    }
    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; (len as usize) + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if copied <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        if title.contains("PopKart")
            || title.contains("D3DProxyWindow")
            || title.contains("KartRider")
            || title.contains("跑跑")
        {
            let out = lparam as *mut HWND;
            if !out.is_null() {
                *out = hwnd;
            }
            return 0;
        }
        1
    }
    let mut hwnd: HWND = std::ptr::null_mut();
    unsafe {
        EnumWindows(Some(enum_window), &mut hwnd as *mut HWND as LPARAM);
        if !hwnd.is_null() && focus_hwnd(hwnd) {
            return true;
        }
    }
    false
}

#[cfg(windows)]
pub(crate) fn post_key_to_kart_rider(key: Key, duration_ms: u64) -> bool {
    if post_key_event_to_kart_rider(key, true).is_err() {
        return false;
    }
    thread::sleep(Duration::from_millis(duration_ms.max(20)));
    post_key_event_to_kart_rider(key, false).is_ok()
}

#[cfg(windows)]
pub(crate) fn launcher_post_key(key: Key, duration_ms: u64) -> Result<(), String> {
    if focus_launcher_window() && platform_key_event(key, true).is_ok() {
        thread::sleep(Duration::from_millis(duration_ms.max(20)));
        if platform_key_event(key, false).is_ok() {
            return Ok(());
        }
        // Do not leave the key held while falling back to PostMessage.
        let _ = platform_key_event(key, false);
    }
    launcher_post_key_event(key, true)?;
    thread::sleep(Duration::from_millis(duration_ms.max(20)));
    launcher_post_key_event(key, false)
}

#[cfg(windows)]
fn launcher_post_key_event(key: Key, down: bool) -> Result<(), String> {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible, PostMessageW,
        WM_KEYDOWN, WM_KEYUP,
    };

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; (len as usize) + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if copied <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        if title.contains("Launcher") || title.contains("Launcher.exe") {
            let out = lparam as *mut HWND;
            if !out.is_null() {
                *out = hwnd;
            }
            return 0;
        }
        1
    }

    let mut hwnd: HWND = std::ptr::null_mut();
    unsafe {
        EnumWindows(Some(enum_window), &mut hwnd as *mut HWND as LPARAM);
        if hwnd.is_null() {
            return Err("Launcher window not found for PostMessage input".to_string());
        }
        let vk = key.vk_code() as usize;
        let scan = key.scan_code() as isize;
        let extended = if key.is_extended() { 1isize << 24 } else { 0 };
        let transition = if down {
            0
        } else {
            (1isize << 30) | (1isize << 31)
        };
        let lparam = 1isize | (scan << 16) | extended | transition;
        let msg = if down { WM_KEYDOWN } else { WM_KEYUP };
        if PostMessageW(hwnd, msg, vk, lparam as LPARAM) == 0 {
            Err(format!(
                "PostMessage launcher {} failed for {}",
                if down { "keydown" } else { "keyup" },
                key.name()
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(windows)]
fn focus_launcher_window() -> bool {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{keybd_event, KEYEVENTF_KEYUP, VK_MENU};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, EnumWindows, GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW,
        GetWindowThreadProcessId, IsIconic, IsWindowVisible, SetForegroundWindow, ShowWindow,
        SW_RESTORE,
    };

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; (len as usize) + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if copied <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        if title.contains("Launcher") || title.contains("Launcher.exe") {
            let out = lparam as *mut HWND;
            if !out.is_null() {
                *out = hwnd;
            }
            return 0;
        }
        1
    }

    unsafe {
        let mut hwnd: HWND = std::ptr::null_mut();
        EnumWindows(Some(enum_window), &mut hwnd as *mut HWND as LPARAM);
        if hwnd.is_null() {
            return false;
        }
        if IsIconic(hwnd) != 0 {
            ShowWindow(hwnd, SW_RESTORE);
        }
        keybd_event(VK_MENU as u8, 0, 0, 0);
        keybd_event(VK_MENU as u8, 0, KEYEVENTF_KEYUP, 0);
        BringWindowToTop(hwnd);
        if SetForegroundWindow(hwnd) != 0 {
            return true;
        }
        let current_thread = GetCurrentThreadId();
        let target_thread = GetWindowThreadProcessId(hwnd, std::ptr::null_mut());
        let foreground = GetForegroundWindow();
        let foreground_thread = if foreground.is_null() {
            0
        } else {
            GetWindowThreadProcessId(foreground, std::ptr::null_mut())
        };
        let attached_target = target_thread != 0
            && target_thread != current_thread
            && AttachThreadInput(current_thread, target_thread, 1) != 0;
        let attached_foreground = foreground_thread != 0
            && foreground_thread != current_thread
            && foreground_thread != target_thread
            && AttachThreadInput(current_thread, foreground_thread, 1) != 0;
        BringWindowToTop(hwnd);
        let ok = SetForegroundWindow(hwnd) != 0 || GetForegroundWindow() == hwnd;
        if attached_foreground {
            AttachThreadInput(current_thread, foreground_thread, 0);
        }
        if attached_target {
            AttachThreadInput(current_thread, target_thread, 0);
        }
        ok || GetForegroundWindow() == hwnd
    }
}

#[cfg(windows)]
pub(crate) fn clear_known_ui_blockers() -> Result<Vec<String>, String> {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible, PostMessageW, WM_CLOSE,
        WM_KEYDOWN, WM_KEYUP,
    };

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; (len as usize) + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if copied <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        if title.contains("Direct3D Failure") {
            let out = lparam as *mut HWND;
            if !out.is_null() {
                *out = hwnd;
            }
            return 0;
        }
        1
    }

    let mut hwnd: HWND = std::ptr::null_mut();
    unsafe {
        EnumWindows(Some(enum_window), &mut hwnd as *mut HWND as LPARAM);
        if hwnd.is_null() {
            return Ok(Vec::new());
        }
        let enter = Key::Enter;
        let vk = enter.vk_code() as usize;
        let scan = enter.scan_code() as isize;
        let down_lparam = 1isize | (scan << 16);
        let up_lparam = 1isize | (scan << 16) | (1isize << 30) | (1isize << 31);
        let mut actions = Vec::new();
        if PostMessageW(hwnd, WM_KEYDOWN, vk, down_lparam as LPARAM) != 0
            && PostMessageW(hwnd, WM_KEYUP, vk, up_lparam as LPARAM) != 0
        {
            actions.push("direct3d_failure_enter".to_string());
        }
        thread::sleep(Duration::from_millis(300));
        if PostMessageW(hwnd, WM_CLOSE, 0, 0) != 0 {
            actions.push("direct3d_failure_wm_close".to_string());
        }
        Ok(actions)
    }
}

#[cfg(windows)]
fn post_key_event_to_kart_rider(key: Key, down: bool) -> Result<(), String> {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible, PostMessageW,
        WM_KEYDOWN, WM_KEYUP,
    };

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0_u16; (len as usize) + 1];
        let copied = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if copied <= 0 {
            return 1;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        if title.contains("PopKart")
            || title.contains("D3DProxyWindow")
            || title.contains("KartRider")
            || title.contains("跑跑")
        {
            let out = lparam as *mut HWND;
            if !out.is_null() {
                *out = hwnd;
            }
            return 0;
        }
        1
    }

    let mut hwnd: HWND = std::ptr::null_mut();
    unsafe {
        EnumWindows(Some(enum_window), &mut hwnd as *mut HWND as LPARAM);
        if hwnd.is_null() {
            return Err("KartRider window not found for PostMessage input".to_string());
        }
        let vk = key.vk_code() as usize;
        let scan = key.scan_code() as isize;
        let extended = if key.is_extended() { 1isize << 24 } else { 0 };
        let transition = if down {
            0
        } else {
            (1isize << 30) | (1isize << 31)
        };
        let lparam = 1isize | (scan << 16) | extended | transition;
        let msg = if down { WM_KEYDOWN } else { WM_KEYUP };
        if PostMessageW(hwnd, msg, vk, lparam as LPARAM) == 0 {
            Err(format!(
                "PostMessage {} failed for {}",
                if down { "keydown" } else { "keyup" },
                key.name()
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(not(windows))]
fn platform_key_event(key: Key, down: bool) -> Result<(), String> {
    eprintln!(
        "dry-run input: {} {}",
        if down { "down" } else { "up" },
        key.name()
    );
    Ok(())
}

#[cfg(windows)]
fn platform_key_event(key: Key, down: bool) -> Result<(), String> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
        KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE,
    };

    let scan = key.scan_code();
    let mut flags = KEYEVENTF_SCANCODE;
    if key.is_extended() {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if !down {
        flags |= KEYEVENTF_KEYUP;
    }
    let input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: 0,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };

    let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
    if sent == 1 {
        Ok(())
    } else {
        // SendInput is blocked (UIPI / elevated foreground window). keybd_event
        // is the same injection path with the same restrictions, so there is no
        // usable fallback; fail loudly instead of pretending the key was sent.
        let last_error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        Err(format!(
            "SendInput blocked for {} {} (sent=0, last_error={last_error})",
            key.name(),
            if down { "down" } else { "up" }
        ))
    }
}
