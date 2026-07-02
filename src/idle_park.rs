//! Kernel idle-loop detection and in-place CPU thread parking.
//!
//! Shared by the interpreter run loop (`mips_exec.rs`) and the JIT dispatch
//! loop (`jit/dispatch.rs`). See `rules/perf/idle-pause-work.md`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::mips_core::{CAUSE_IP_MASK, CAUSE_IP7, STATUS_IM_MASK};
use crate::mips_core::MipsCore;

const IDLE_RING: usize = 32;
const SLICE_NS: u64 = 1_000_000;
const MIN_SLICE_NS: u64 = 50_000;

/// Tracks recent architectural-state hashes to detect polling idle loops.
#[derive(Default)]
pub struct IdleParkState {
    ring: [u64; IDLE_RING],
    ring_len: usize,
    ring_pos: usize,
}

impl IdleParkState {
    /// Hash PC + GPRs (excluding k0/k1 scratch registers).
    fn hash_state(core: &MipsCore) -> u64 {
        let mut h = core.pc;
        for (i, &g) in core.gpr.iter().enumerate() {
            if i == 26 || i == 27 {
                continue;
            }
            h = h.rotate_left(7) ^ g;
        }
        h
    }

    /// Update idle ring. Returns true when the current state repeated (safe to park).
    pub fn update(&mut self, core: &MipsCore) -> bool {
        let ie = core.interrupts_enabled();
        let pending = core.interrupts.load(Ordering::Relaxed) as u32;
        let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
        let im = core.cp0_status & STATUS_IM_MASK;
        let interrupt_ready = (ip & im) != 0;

        if !(ie && !interrupt_ready) {
            self.ring_len = 0;
            self.ring_pos = 0;
            return false;
        }

        let h = Self::hash_state(core);
        if self.ring[..self.ring_len].contains(&h) {
            return true;
        }
        self.ring[self.ring_pos] = h;
        self.ring_pos = (self.ring_pos + 1) % IDLE_RING;
        if self.ring_len < IDLE_RING {
            self.ring_len += 1;
        }
        false
    }

    /// Park in ≤1 ms slices until an interrupt is due or pending.
    pub fn park(&self, core: &mut MipsCore, running: &AtomicBool) {
        // hw_per_ns is measured directly from the last Compare interval's real
        // elapsed time (see mips_core.rs write_cp0), unlike compare_delta_slow
        // which is only a fuzzy-matched bucket label that *assumes* 100 Hz on
        // its first sample. Using the bucket instead of the measured rate here
        // silently mis-paces cp0_count advancement whenever the guest's actual
        // first Compare interval isn't really 10ms (rules/perf/idle-pause-work.md).
        let hw_per_ns = core.hw_per_ns;
        let cs = core.count_step;
        if hw_per_ns == 0 || cs == 0 {
            return;
        }

        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let cnt = core.cp0_count;
            let cmp = core.cp0_compare;
            let diff = cmp.wrapping_sub(cnt);
            if (diff >> 63) != 0 || (diff >> 32) == 0 {
                core.cp0_count = cmp;
                core.cp0_cause |= CAUSE_IP7;
                break;
            }
            let pending = core.interrupts.load(Ordering::Relaxed) as u32;
            let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
            let im = core.cp0_status & STATUS_IM_MASK;
            if (ip & im) != 0 {
                break;
            }

            let rem_hw = diff >> 32;
            let ns_to_tick = ((rem_hw as u128) << 32) / hw_per_ns as u128;
            let slice_ns = (ns_to_tick.min(SLICE_NS as u128)) as u64;
            if slice_ns < MIN_SLICE_NS {
                core.cp0_count = cmp;
                core.cp0_cause |= CAUSE_IP7;
                break;
            }

            let t0 = Instant::now();
            std::thread::sleep(Duration::from_nanos(slice_ns));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            let adv_hw = ((elapsed_ns as u128 * hw_per_ns as u128) >> 32) as u64;
            core.cp0_count = core.cp0_count.wrapping_add(adv_hw << 32);
            let adv_instrs = (((adv_hw as u128) << 32) / cs as u128) as u64;
            core.local_cycles = core.local_cycles.wrapping_add(adv_instrs);
        }
    }
}

pub fn idle_park_enabled() -> bool {
    std::env::var_os("IRIS_NO_IDLE").is_none()
}
