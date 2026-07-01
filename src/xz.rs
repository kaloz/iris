/// Indy XZ / Elan (GR3) graphics — preview stub
///
/// The Indy XZ option replaces Newport (REX3) with an Express-class GR3 board:
/// HQ2 command engine, four GE7 geometry engines, RE3 raster engine, VC1/XMAP5
/// display path. IRIX uses the `gfx` (Express) driver stack, not Newport.
///
/// **Preview only:** register storage and probe-friendly ID/status defaults.
/// No command FIFO draining, geometry, raster, or framebuffer. Selecting
/// `graphics.board = "xz"` disables the Newport window/compositor path.
///
/// GIO mapping matches the Newport XL layout used by MAME (`newport.cpp` mem_map):
/// the CPU register window sits at offset `+0x0F0000` within the 4 MB gfx slot
/// (`0x1F000000`–`0x1F3FFFFF` on IP24).
///
/// References:
///   `docs/indy-xz-elan.md`
///   MAME `src/devices/bus/gio64/newport.cpp` (GIO slot layout)
///   SGI Indy Technical Report ch.5 (XZ architecture overview)

use parking_lot::Mutex;
use std::io::Write as IoWrite;

use crate::devlog::LogModule;
use crate::snapshot::{get_field, hex_u32, toml_u32};
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, Device, Saveable};

// ─── GIO64 mapping (Indy IP24 gfx slot) ─────────────────────────────────────

/// Physical base of the 4 MB GIO gfx aperture.
pub const XZ_GIO_BASE: u32 = 0x1F00_0000;
pub const XZ_GIO_SIZE: u32 = 0x0040_0000;

/// HQ2 / CPU register window (same offset within the slot as REX3 on Newport).
pub const XZ_REG_BASE: u32 = 0x1F0F_0000;
pub const XZ_REG_SIZE: u32 = 0x2000;

// ─── Register offsets (relative to XZ_REG_BASE) ─────────────────────────────
//
// HQ2 is a large gate array; public sources disagree on the fine-grained map.
// Offsets below follow the IRIX/Linux Express driver grouping (board ID, status,
// interrupt, command FIFO port) documented in `docs/indy-xz-elan.md`. Treat
// unlisted offsets as reserved — reads return 0, writes are logged and ignored.

pub mod reg {
    /// Board / architecture ID (read-only). Low byte encodes GR generation.
    pub const BOARD_ID:     u32 = 0x0000;
    /// HQ2 revision (read-only).
    pub const REVISION:     u32 = 0x0004;
    /// Command-engine status: FIFO level, engine busy (preview: always idle).
    pub const STATUS:       u32 = 0x0008;
    /// Interrupt status (write-1-to-clear bits in hardware; stub stores written val).
    pub const INTR_STATUS:  u32 = 0x000C;
    /// Interrupt enable mask.
    pub const INTR_ENABLE:  u32 = 0x0010;
    /// Command FIFO write port (writes accepted, not executed).
    pub const FIFO_WRITE:   u32 = 0x0018;
    /// FIFO read/debug port (hardware-dependent; stub returns 0).
    pub const FIFO_READ:    u32 = 0x001C;
    /// Reset / soft-reset control.
    pub const RESET:        u32 = 0x0020;

    pub const BOARD_ID_VAL: u32 = 0x0003_0001; // GR3 family, XZ variant (research placeholder)
    pub const REVISION_VAL: u32 = 0x0000_0021; // HQ2.1 per Indy `gfxinfo` reports

    /// STATUS: FIFO empty (bit 0), GE idle (bit 1), RE idle (bit 2).
    pub const STATUS_FIFO_EMPTY: u32 = 1 << 0;
    pub const STATUS_GE_IDLE:    u32 = 1 << 1;
    pub const STATUS_RE_IDLE:    u32 = 1 << 2;
    pub const STATUS_RESET_VAL:  u32 =
        STATUS_FIFO_EMPTY | STATUS_GE_IDLE | STATUS_RE_IDLE;
}

struct XzState {
    intr_status: u32,
    intr_enable: u32,
    fifo_depth:  u32, // bytes accepted (diagnostic counter only)
}

impl Default for XzState {
    fn default() -> Self {
        Self { intr_status: 0, intr_enable: 0, fifo_depth: 0 }
    }
}

#[derive(Clone)]
pub struct Xz {
    state: std::sync::Arc<Mutex<XzState>>,
}

impl Xz {
    pub fn new() -> Self {
        Self { state: std::sync::Arc::new(Mutex::new(XzState::default())) }
    }

    pub fn power_on(&self) {
        *self.state.lock() = XzState::default();
    }

    fn in_regs(addr: u32) -> bool {
        (addr & 0xFFFF_E000) == XZ_REG_BASE
    }

    fn reg_offset(addr: u32) -> u32 {
        addr & (XZ_REG_SIZE - 1)
    }

    fn read_reg(&self, off: u32) -> u32 {
        let st = self.state.lock();
        match off {
            reg::BOARD_ID    => reg::BOARD_ID_VAL,
            reg::REVISION    => reg::REVISION_VAL,
            reg::STATUS      => reg::STATUS_RESET_VAL,
            reg::INTR_STATUS => st.intr_status,
            reg::INTR_ENABLE => st.intr_enable,
            reg::FIFO_READ   => 0,
            _                => 0,
        }
    }

    fn write_reg(&self, off: u32, val: u32) {
        let mut st = self.state.lock();
        match off {
            reg::INTR_STATUS => st.intr_status &= !val,
            reg::INTR_ENABLE => st.intr_enable = val,
            reg::FIFO_WRITE  => {
                st.fifo_depth = st.fifo_depth.saturating_add(4);
                dlog_dev!(LogModule::Rex3, "XZ: FIFO write {:08x} (stub, depth={})", val, st.fifo_depth);
            }
            reg::RESET => {
                *st = XzState::default();
            }
            _ => {
                dlog_dev!(LogModule::Rex3, "XZ: write reg {:04x} = {:08x} (ignored)", off, val);
            }
        }
    }
}

impl Device for Xz {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("xz".into(), "XZ/Elan preview stub (status)".into())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd != "xz" {
            return Err(format!("unknown xz command: {cmd}"));
        }
        if !args.is_empty() {
            return Err("usage: xz".into());
        }
        let st = self.state.lock();
        writeln!(
            writer,
            "XZ/Elan preview stub @ {:#010x}..{:#010x}\n  fifo_bytes_accepted={}\n  intr_status={:#010x} intr_enable={:#010x}",
            XZ_REG_BASE,
            XZ_REG_BASE + XZ_REG_SIZE,
            st.fifo_depth,
            st.intr_status,
            st.intr_enable,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl BusDevice for Xz {
    fn read32(&self, addr: u32) -> BusRead32 {
        if !Self::in_regs(addr) {
            return BusRead32::ok(0);
        }
        BusRead32::ok(self.read_reg(Self::reg_offset(addr)))
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        if Self::in_regs(addr) {
            self.write_reg(Self::reg_offset(addr), val);
        }
        BUS_OK
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        BusRead8::ok(self.read32(addr & !3).data as u8)
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        let shift = (addr & 3) * 8;
        let cur = self.read32(addr & !3).data;
        self.write32(addr & !3, (cur & !(0xFF << shift)) | ((val as u32) << shift));
        BUS_OK
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        BusRead16::ok(self.read32(addr & !3).data as u16)
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        let shift = if (addr & 2) != 0 { 16 } else { 0 };
        let cur = self.read32(addr & !3).data;
        self.write32(addr & !3, (cur & !(0xFFFF << shift)) | ((val as u32) << shift));
        BUS_OK
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        let lo = self.read32(addr).data as u64;
        let hi = self.read32(addr.wrapping_add(4)).data as u64;
        BusRead64::ok((hi << 32) | lo)
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        self.write32(addr, val as u32);
        self.write32(addr.wrapping_add(4), (val >> 32) as u32);
        BUS_OK
    }
}

impl Saveable for Xz {
    fn save_state(&self) -> toml::Value {
        let st = self.state.lock();
        let mut tbl = toml::map::Map::new();
        tbl.insert("intr_status".into(), hex_u32(st.intr_status));
        tbl.insert("intr_enable".into(), hex_u32(st.intr_enable));
        tbl.insert("fifo_depth".into(), toml::Value::Integer(st.fifo_depth as i64));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut st = self.state.lock();
        if let Some(x) = get_field(v, "intr_status") {
            st.intr_status = toml_u32(x).unwrap_or(st.intr_status);
        }
        if let Some(x) = get_field(v, "intr_enable") {
            st.intr_enable = toml_u32(x).unwrap_or(st.intr_enable);
        }
        if let Some(x) = get_field(v, "fifo_depth") {
            if let Some(n) = x.as_integer() {
                st.fifo_depth = n as u32;
            }
        }
        Ok(())
    }
}
