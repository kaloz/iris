//! RAM bank helpers shared by the Memory tab, menus, and status readouts.

/// Quick preset totals (MB) for the Memory menu and new-machine dialog.
pub const RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512];

pub fn active_banks(banks: &[u32; 4]) -> usize {
    banks.iter().filter(|&&s| s > 0).count()
}

pub fn ram_summary(banks: &[u32; 4]) -> String {
    let total: u32 = banks.iter().sum();
    let n = active_banks(banks);
    format!("{total} MB ({n} bank{})", if n == 1 { "" } else { "s" })
}
