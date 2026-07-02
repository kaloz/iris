/// IMPACT / MGRAS graphics — preview stub (Indigo2 IP22)
///
/// Post-1995 IMPACT boards use the MGRAS ASIC set (geometry engine, raster
/// engine, TRAM controllers) spread across one to three GIO64 slots depending on
/// the option (Solid / High / Maximum).
///
/// **Preview only:** per-slot register files with probe-friendly board IDs and
/// idle status. No TRAM, no DMA, no GL/command processing.
///
/// GIO slot bases (physical, IP22):
///   gfx  — 0x1F000000 (4 MB)
///   exp0 — 0x1F400000 (2 MB)
///   exp1 — 0x1F600000 (4 MB)
///
/// See `docs/impact-mgras-research.md` for hardware notes and implementation status.

use parking_lot::Mutex;
use std::io::Write as IoWrite;

use crate::config::{ImpactSection, ImpactSlot};
use crate::devlog::LogModule;
use crate::snapshot::{get_field, hex_u32, toml_u32};
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, Device, Saveable};

// ─── GIO slot geometry ───────────────────────────────────────────────────────

pub const MGRAS_SLOT_GFX_BASE:  u32 = 0x1F00_0000;
pub const MGRAS_SLOT_GFX_SIZE:  u32 = 0x0040_0000;
pub const MGRAS_SLOT_EXP0_BASE: u32 = 0x1F40_0000;
pub const MGRAS_SLOT_EXP0_SIZE: u32 = 0x0020_0000;
pub const MGRAS_SLOT_EXP1_BASE: u32 = 0x1F60_0000;
pub const MGRAS_SLOT_EXP1_SIZE: u32 = 0x0040_0000;

/// MGRAS CPU register window within each populated slot (research placeholder;
/// mirrors the Newport/XZ `+0x0F0000` pattern until a verified map lands).
pub const MGRAS_REG_OFF:  u32 = 0x000F_0000;
pub const MGRAS_REG_SIZE: u32 = 0x2000;

// ─── Register offsets (relative to slot base + MGRAS_REG_OFF) ───────────────

pub mod reg {
    use crate::config::ImpactSlot;

    pub const BOARD_ID:    u32 = 0x0000;
    pub const REVISION:    u32 = 0x0004;
    pub const STATUS:      u32 = 0x0008;
    pub const INTR_STATUS: u32 = 0x000C;
    pub const INTR_ENABLE: u32 = 0x0010;
    pub const FIFO_WRITE:  u32 = 0x0018;
    pub const SLOT_ROLE:   u32 = 0x0024; // which GE/TRAM slice (multi-slot boards)

    pub const STATUS_IDLE: u32 = 0x0000_0007; // FIFO empty + engines idle (preview)

    pub fn board_id_for(slot: ImpactSlot) -> u32 {
        match slot {
            ImpactSlot::None  => 0,
            ImpactSlot::Solid => 0x004D_4752, // "MGR" + Solid class tag (ASCII-ish)
            ImpactSlot::High  => 0x004D_4748, // High IMPACT
            ImpactSlot::Max   => 0x004D_474D, // Maximum IMPACT
        }
    }

    pub fn revision_for(slot: ImpactSlot) -> u32 {
        match slot {
            ImpactSlot::None  => 0,
            ImpactSlot::Solid => 0x0000_0100,
            ImpactSlot::High  => 0x0000_0200,
            ImpactSlot::Max   => 0x0000_0300,
        }
    }
}

#[derive(Clone, Copy)]
struct SlotMap {
    kind: ImpactSlot,
    base: u32,
    size: u32,
}

struct SlotState {
    intr_status: u32,
    intr_enable: u32,
    fifo_depth:  u32,
}

struct MgrasState {
    slots: [Option<SlotState>; 3],
}

#[derive(Clone)]
pub struct Mgras {
    map:   [SlotMap; 3],
    state: std::sync::Arc<Mutex<MgrasState>>,
}

impl Mgras {
    pub fn new(cfg: &ImpactSection) -> Self {
        let kinds = [cfg.gfx, cfg.exp0, cfg.exp1];
        let bases = [MGRAS_SLOT_GFX_BASE, MGRAS_SLOT_EXP0_BASE, MGRAS_SLOT_EXP1_BASE];
        let sizes = [MGRAS_SLOT_GFX_SIZE, MGRAS_SLOT_EXP0_SIZE, MGRAS_SLOT_EXP1_SIZE];
        let mut map = [SlotMap { kind: ImpactSlot::None, base: 0, size: 0 }; 3];
        for i in 0..3 {
            map[i] = SlotMap { kind: kinds[i], base: bases[i], size: sizes[i] };
        }
        let mut slots = [None, None, None];
        for (i, m) in map.iter().enumerate() {
            if m.kind != ImpactSlot::None {
                slots[i] = Some(SlotState { intr_status: 0, intr_enable: 0, fifo_depth: 0 });
            }
        }
        Self {
            map,
            state: std::sync::Arc::new(Mutex::new(MgrasState { slots })),
        }
    }

    pub fn power_on(&self) {
        let mut st = self.state.lock();
        for (i, m) in self.map.iter().enumerate() {
            st.slots[i] = if m.kind != ImpactSlot::None {
                Some(SlotState { intr_status: 0, intr_enable: 0, fifo_depth: 0 })
            } else {
                None
            };
        }
    }

    pub fn any_slot(&self) -> bool {
        self.map.iter().any(|m| m.kind != ImpactSlot::None)
    }

    fn locate(&self, addr: u32) -> Option<(usize, u32)> {
        for (i, m) in self.map.iter().enumerate() {
            if m.kind == ImpactSlot::None {
                continue;
            }
            let reg_base = m.base.wrapping_add(MGRAS_REG_OFF);
            if (addr & 0xFFFF_E000) == reg_base {
                return Some((i, addr & (MGRAS_REG_SIZE - 1)));
            }
            // Rest of the slot aperture: reads as 0, writes ignored.
            if addr >= m.base && addr < m.base.wrapping_add(m.size) {
                return None;
            }
        }
        None
    }

    fn read_reg(&self, slot_idx: usize, off: u32) -> u32 {
        let kind = self.map[slot_idx].kind;
        let st = self.state.lock();
        let Some(s) = &st.slots[slot_idx] else { return 0 };
        match off {
            reg::BOARD_ID    => reg::board_id_for(kind),
            reg::REVISION    => reg::revision_for(kind),
            reg::STATUS      => reg::STATUS_IDLE,
            reg::INTR_STATUS => s.intr_status,
            reg::INTR_ENABLE => s.intr_enable,
            reg::SLOT_ROLE   => slot_idx as u32,
            _                => 0,
        }
    }

    fn write_reg(&self, slot_idx: usize, off: u32, val: u32) {
        let mut st = self.state.lock();
        let Some(s) = &mut st.slots[slot_idx] else { return };
        match off {
            reg::INTR_STATUS => s.intr_status &= !val,
            reg::INTR_ENABLE => s.intr_enable = val,
            reg::FIFO_WRITE  => {
                s.fifo_depth = s.fifo_depth.saturating_add(4);
                dlog_dev!(
                    LogModule::Rex3,
                    "MGRAS slot{}: FIFO write {:08x} (stub, depth={})",
                    slot_idx,
                    val,
                    s.fifo_depth
                );
            }
            _ => {
                dlog_dev!(
                    LogModule::Rex3,
                    "MGRAS slot{}: write reg {:04x} = {:08x} (ignored)",
                    slot_idx,
                    off,
                    val
                );
            }
        }
    }
}

impl Device for Mgras {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("mgras".into(), "IMPACT/MGRAS preview stub (status)".into()),
            ("impact".into(), "IMPACT inventory summary (hinv-style preview)".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd != "mgras" && cmd != "impact" {
            return Err(format!("unknown mgras command: {cmd}"));
        }
        if !args.is_empty() {
            return Err(format!("usage: {cmd}"));
        }
        let st = self.state.lock();
        let names = ["gfx", "exp0", "exp1"];
        if cmd == "impact" {
            writeln!(writer, "Graphics inventory (IMPACT preview — driver attach not implemented):").map_err(|e| e.to_string())?;
        }
        for (i, m) in self.map.iter().enumerate() {
            if m.kind == ImpactSlot::None {
                if cmd == "impact" {
                    continue;
                }
                writeln!(writer, "  {}: (empty)", names[i]).map_err(|e| e.to_string())?;
                continue;
            }
            let depth = st.slots[i].as_ref().map(|s| s.fifo_depth).unwrap_or(0);
            let label = match m.kind {
                ImpactSlot::Solid => "IMPACT Solid",
                ImpactSlot::High  => "IMPACT High",
                ImpactSlot::Max   => "IMPACT Maximum",
                ImpactSlot::None  => unreachable!(),
            };
            if cmd == "impact" {
                writeln!(
                    writer,
                    "  Graphics board {}: {}  (GIO @ {:#010x})",
                    i,
                    label,
                    m.base.wrapping_add(MGRAS_REG_OFF),
                )
                .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    writer,
                    "  {}: {:?} @ {:#010x} fifo_bytes={}",
                    names[i],
                    m.kind,
                    m.base.wrapping_add(MGRAS_REG_OFF),
                    depth,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        if cmd == "impact" && !self.map.iter().any(|m| m.kind != ImpactSlot::None) {
            writeln!(writer, "  (no IMPACT boards configured in [impact])").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

impl BusDevice for Mgras {
    fn read32(&self, addr: u32) -> BusRead32 {
        match self.locate(addr) {
            Some((idx, off)) => BusRead32::ok(self.read_reg(idx, off)),
            None => BusRead32::ok(0),
        }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        if let Some((idx, off)) = self.locate(addr) {
            self.write_reg(idx, off, val);
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

impl Saveable for Mgras {
    fn save_state(&self) -> toml::Value {
        let st = self.state.lock();
        let mut slots = toml::map::Map::new();
        let names = ["gfx", "exp0", "exp1"];
        for (i, name) in names.iter().enumerate() {
            if self.map[i].kind == ImpactSlot::None {
                continue;
            }
            let Some(s) = st.slots[i].as_ref() else { continue };
            let mut slot_tbl = toml::map::Map::new();
            slot_tbl.insert("intr_status".into(), hex_u32(s.intr_status));
            slot_tbl.insert("intr_enable".into(), hex_u32(s.intr_enable));
            slot_tbl.insert("fifo_depth".into(), toml::Value::Integer(s.fifo_depth as i64));
            slots.insert((*name).into(), toml::Value::Table(slot_tbl));
        }
        toml::Value::Table(slots)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let Some(tbl) = v.as_table() else { return Ok(()) };
        let names = ["gfx", "exp0", "exp1"];
        let mut st = self.state.lock();
        for (i, name) in names.iter().enumerate() {
            let Some(slot_v) = tbl.get(*name) else { continue };
            let Some(s) = st.slots[i].as_mut() else { continue };
            if let Some(x) = get_field(slot_v, "intr_status") {
                s.intr_status = toml_u32(x).unwrap_or(s.intr_status);
            }
            if let Some(x) = get_field(slot_v, "intr_enable") {
                s.intr_enable = toml_u32(x).unwrap_or(s.intr_enable);
            }
            if let Some(x) = get_field(slot_v, "fifo_depth") {
                if let Some(n) = x.as_integer() {
                    s.fifo_depth = n as u32;
                }
            }
        }
        Ok(())
    }
}
