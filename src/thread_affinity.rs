//! Optional host thread affinity for emulator worker threads.

use std::sync::OnceLock;

use crate::config::PerfConfig;

static PERF: OnceLock<PerfConfig> = OnceLock::new();

/// Called once from `Machine::new` with the loaded `[perf]` section.
pub fn init(perf: PerfConfig) {
    let _ = PERF.set(perf);
}

#[derive(Clone, Copy, Debug)]
pub enum PerfRole {
    MipsCpu,
    Rex3Processor,
    Rex3Refresh,
}

/// Pin the current thread when `[perf] thread_affinity` is enabled.
pub fn pin_current(role: PerfRole) {
    let Some(perf) = PERF.get() else { return; };
    if !perf.thread_affinity {
        return;
    }
    let core = match role {
        PerfRole::MipsCpu => perf.cpu_core,
        PerfRole::Rex3Processor => perf.rex3_core,
        PerfRole::Rex3Refresh => perf.refresh_core,
    };
    let Some(core) = core else { return; };
    if apply_core(core) {
        eprintln!("iris: pinned {:?} thread to core {}", role, core);
    }
}

#[cfg(windows)]
fn apply_core(core: u32) -> bool {
    let mask = 1usize << core.min(63);
    unsafe {
        windows_sys::Win32::System::Threading::SetThreadAffinityMask(
            windows_sys::Win32::System::Threading::GetCurrentThread(),
            mask,
        ) != 0
    }
}

#[cfg(target_os = "linux")]
fn apply_core(core: u32) -> bool {
    use std::mem::MaybeUninit;
    let mut set: libc::cpu_set_t = unsafe { MaybeUninit::zeroed().assume_init() };
    unsafe {
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core as usize, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) == 0
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn apply_core(_core: u32) -> bool {
    false
}

#[cfg(not(any(windows, unix)))]
fn apply_core(_core: u32) -> bool {
    false
}

/// One-line summary for `perf snapshot`.
pub fn status_line() -> String {
    let Some(perf) = PERF.get() else {
        return "thread_affinity: (not initialized)".to_string();
    };
    if !perf.thread_affinity {
        return "thread_affinity: off".to_string();
    }
    format!(
        "thread_affinity: on  cpu={:?} rex3={:?} refresh={:?}",
        perf.cpu_core, perf.rex3_core, perf.refresh_core
    )
}
