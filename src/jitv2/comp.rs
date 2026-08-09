//! JIT v2 compiler-thread logic. See `rules/jitv2/jit-v2-design.md`.
//!
//! `handle_request` is the compile-from-snapshot protocol (§6.5): read the
//! page's 4KB snapshot off the bus, bounded-walk it from the requested
//! offset, hand the walked region to `Codegen::compile_region`, and publish
//! the result (or sticky-deny the offset if codegen declined it). Bounded to
//! `MAX_INSTRS_PER_COMPILE` *head* instructions — started at 1 (the smallest
//! possible working JIT) and growing incrementally from there, not a
//! redesign each time (the walk bound is just `max_instrs` on
//! `Analyzer::walk_bounded`). A branch/jump's mandatory delay slot is never
//! charged against this budget (`Analyzer::visit_slot` — a slot can never be
//! omitted, so it was never a truncation candidate) — including a nested
//! delay-slot chain (branch-in-delay-slot, "unusual but legal" on real
//! hardware), which keeps extending for free until it terminates or runs
//! off the page, at which point the walk declines the whole region rather
//! than compiling a partial chain.
//!
//! Corpus collection (raw page dump to `jitv2_corpus/`, used to develop the
//! analyzer/codegen offline against real captured pages) is preserved behind
//! the `jitv2_corpus_dump` feature — with it off, `handle_request` never
//! touches the filesystem.

use std::sync::Arc;

use crate::jitv2::analyzer::Analyzer;
use crate::jitv2::codegen::Codegen;
use crate::jitv2::{CompileRequest, ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::traits::BusDevice;

/// Instruction budget for a compile-from-arrival region (see module doc):
/// head instructions only — a branch/jump's mandatory delay slot (or nested
/// slot-chain) is free and always included regardless of this number
/// (`Analyzer::visit_slot`). Raised incrementally from 1 (the original
/// "smallest possible working JIT" milestone) through 2/3/4/8, all booted
/// clean — a live `j2 status` histogram at 8 showed real regions landing
/// anywhere from 1 to 16 instructions (the tail past the nominal budget
/// comes from a branch's mandatory delay slot counting toward the total but
/// not the budget itself), clustering 8-11, with only a handful ever
/// reaching 16. `usize::MAX` here is deliberate, not "as large as
/// possible": any value at or above `ENTRIES_PER_PAGE` (1024 words/page) is
/// already equivalent to no budget at all, since the walk was always going
/// to decline rather than compile a region longer than fits on one
/// physical page anyway (module doc: "runs off the page... declines the
/// whole region") — the page boundary, not this constant, is the real
/// ceiling from here on (same as `Analyzer::walk`'s own unbounded case).
/// Both `Analyzer::walk_bounded` and `Codegen::compile_region` were already
/// written generically against `max_instrs` (fallthrough-edge wiring for a
/// multi-instruction straight-line region already exists, per
/// `compile_region`'s Pass 2), so this is still a config change, not a
/// redesign — just the last one for this knob.
const MAX_INSTRS_PER_COMPILE: usize = usize::MAX;

/// Minimum walked-region instruction count a compile must reach to actually
/// get compiled — regions shorter than this are sticky-denylisted instead,
/// same treatment as any other decline (§6.4). Below this floor, the
/// per-compile fixed overhead (Cranelift IR building + one arena allocation
/// + one entry_table publish) very likely costs more than the region will
/// ever save over just interpreting it: a true single-instruction region is
/// the worst case for this tradeoff and, per `MAX_INSTRS_PER_COMPILE`'s own
/// doc comment, real regions cluster 8-11 instructions anyway — a
/// one-instruction region is far more often either a rare cold path or an
/// analyzer/codegen edge case than a genuinely hot single-instruction loop
/// body worth paying compile cost for. `j2 min-instrs [N]` tunes this at
/// runtime. Applies identically to `handle_request` and
/// `handle_request_deferred` — both consult `min_instrs_to_compile()` right
/// after a successful, non-empty walk.
///
/// Defaults to 1 (no filtering) under `developer` — diagnostics builds want
/// to see and measure every compile the analyzer/codegen would otherwise
/// attempt, not have some silently skipped by a production-tuned floor — and
/// to 2 otherwise, since real-world usage wants a *little* filtering by
/// default rather than requiring a manual `j2 min-instrs` on every run.
static MIN_INSTRS_TO_COMPILE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(
    if cfg!(feature = "developer") { 1 } else { 2 }
);

pub fn set_min_instrs_to_compile(n: usize) {
    MIN_INSTRS_TO_COMPILE.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

pub fn min_instrs_to_compile() -> usize {
    MIN_INSTRS_TO_COMPILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Handle a single compile request end to end (§6.5): snapshot the page,
/// walk from `req.offset`, codegen, and publish. No-ops (declining silently)
/// if the page's snapshot read fails, the walk finds the entry offset
/// excluded, or codegen has no emitter for something in the walked region —
/// the last two sticky-denylist the offset so arrival stops re-requesting a
/// compile that will only be declined again (§6.4).
///
/// Returns `true` iff `codegen`'s arena ran out of memory on this call
/// (`Codegen::last_compile_ran_out_of_memory`) — the caller (`worker_loop`
/// or the inline dispatch path) owns the `Jitv2`/CPU-pause machinery this
/// function doesn't have access to, so it's responsible for actually
/// flushing when this is `true`. `false` for every other outcome
/// (success, a real codegen decline, or a transient bus-read failure).
pub fn handle_request(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };
    let offset = req.offset as usize;

    // Clears this offset's ENTRY_SCHEDULED flag on every return path,
    // including early returns and panics — `exec_decoded`'s dispatch gate
    // (mips_exec.rs) sets this bit via `try_schedule` before sending a
    // request, specifically so a hot PC that keeps re-satisfying the gate's
    // trigger conditions on every dispatch (e.g. a loop back-edge landing on
    // the same still-uncompiled word every iteration) doesn't flood the
    // compile queue with duplicate requests for the exact same offset while
    // the first one is still in flight. Whatever this function decides
    // (publish, denylist, or "retry later" on a bus read failure), the bit
    // must come back down afterward or that offset can never be
    // (re-)requested again — a scope guard rather than clearing at each of
    // the several return points below so a future added early-return can't
    // forget it.
    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage, offset: usize }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(self.offset); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page, offset };

    if page.is_denylisted(offset) || page.is_entry_valid(offset) {
        return false; // already decided (rejected or already has a fresh artifact)
    }

    let phys_base = page.pfn * PAGE_SIZE;
    // gen_snap is read *before* the copy (§6.5 step 2) so a mutation that
    // lands between this read and the snapshot copy below is never missed:
    // worst case it's captured in the copy too, and publish's re-check
    // catches anything after.
    let gen_snap = page.current_gen();

    let mut words = [0u32; ENTRIES_PER_PAGE];
    for (i, w) in words.iter_mut().enumerate() {
        let r = bus.read32(phys_base + (i as u32) * 4);
        if !r.is_ok() {
            return false; // page not readable right now; the offset stays un-denylisted and can retry later
        }
        *w = r.data;
    }

    #[cfg(feature = "jitv2_corpus_dump")]
    dump_corpus_snapshot(page, req.offset, &words);

    let (instrs, non_empty) = analyzer.walk_bounded(&words, req.offset, phys_base, MAX_INSTRS_PER_COMPILE);
    if !non_empty {
        page.denylist(offset); // entry offset itself is excluded (§6.4)
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::EntryExcluded);
        return false;
    }

    let instr_count = crate::jitv2::analyzer::instrs_linear(instrs).count();
    if instr_count < min_instrs_to_compile() {
        // Too short to be worth the fixed per-compile cost — see
        // MIN_INSTRS_TO_COMPILE's own doc comment. Sticky-denylisted like
        // any other decline (§6.4): the region's instruction count can't
        // change without the page itself mutating (a gen bump, which clears
        // ENTRY_DENYLISTED alongside it), so there's nothing to gain from
        // re-evaluating this offset on a later arrival against the same
        // unchanged bytes.
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::TooShort);
        return false;
    }

    let mut instrs_owned = *instrs;
    // skip_entry_preamble=true: this compile is always reached from
    // mips_exec.rs's step() dispatch loop, which already ran the equivalent
    // IP7/pending-interrupt checks for req.offset's exact PC immediately
    // before ever arriving here — see compile_region's doc comment.
    let func = codegen.compile_region(&mut instrs_owned, req.offset, req.compiled_for_fr1, true);
    match func {
        Some(jit_fn) => {
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            page.publish(offset, jit_fn as *const (), gen_snap, instr_count, code_size);
            #[cfg(feature = "developer")]
            stats.compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            // Not a real decline — the offset is perfectly compilable, the
            // shared arena is just out of room right now (see
            // `Codegen::last_compile_ran_out_of_memory`'s doc comment).
            // Denylisting it would be wrong (permanently, until an
            // unrelated future flush happens to clear ENTRY_DENYLISTED too);
            // instead, leave it un-denylisted so it retries naturally on
            // its own next arrival, and tell the caller to flush — it has
            // the `Jitv2`/CPU-pause machinery this function doesn't.
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            page.denylist(offset); // some visited instruction has no emitter yet, or a real Cranelift verifier rejection
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// One compiled-but-not-yet-finalized artifact, accumulated by
/// `handle_request_deferred` and consumed by `flush_pending_batch` — the
/// batched counterpart to `handle_request`'s immediate `page.publish(...)`
/// call. Carries everything `page.publish` needs except the `JitFn` pointer
/// itself, since that pointer doesn't exist yet (the whole point of
/// deferring: it's only valid after the batch's one `finalize_batch` call).
///
/// # Safety
/// `page` must outlive this entry, same requirement as `CompileRequest::page`
/// (see that field's own doc comment) — the caller (`worker_loop`) only ever
/// holds these for the brief window between a `consumer.pop()` and the next
/// flush, well within a single compile-request's already-established
/// lifetime guarantee.
pub struct PendingPublish {
    page: *mut crate::jitv2::PhysicalCodePage,
    offset: usize,
    func_id: cranelift_module::FuncId,
    gen_snap: u64,
    instr_count: usize,
    code_size: u32,
}

unsafe impl Send for PendingPublish {}

/// Deferred counterpart to `handle_request`: identical compile-from-snapshot
/// protocol (walk, codegen), but stops short of finalizing/publishing —
/// instead of a `page.publish(...)` call, a successful compile is appended
/// to `pending` as a `PendingPublish` for the caller to finalize later in a
/// batch (`Codegen::finalize_batch` + `flush_pending_batch`). Every other
/// outcome (excluded entry, bus-read failure, OOM, real codegen decline) is
/// handled exactly like `handle_request` — those don't produce a `FuncId` to
/// defer at all, so there's nothing batching-specific about them.
///
/// Returns the same `bool` as `handle_request` (arena-OOM signal) for the
/// same reason.
pub fn handle_request_deferred(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    pending: &mut Vec<PendingPublish>,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };
    let offset = req.offset as usize;

    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage, offset: usize }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(self.offset); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page, offset };

    if page.is_denylisted(offset) || page.is_entry_valid(offset) {
        return false;
    }

    let phys_base = page.pfn * PAGE_SIZE;
    let gen_snap = page.current_gen();

    let mut words = [0u32; ENTRIES_PER_PAGE];
    for (i, w) in words.iter_mut().enumerate() {
        let r = bus.read32(phys_base + (i as u32) * 4);
        if !r.is_ok() {
            return false;
        }
        *w = r.data;
    }

    #[cfg(feature = "jitv2_corpus_dump")]
    dump_corpus_snapshot(page, req.offset, &words);

    let (instrs, non_empty) = analyzer.walk_bounded(&words, req.offset, phys_base, MAX_INSTRS_PER_COMPILE);
    if !non_empty {
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::EntryExcluded);
        return false;
    }

    let instr_count = crate::jitv2::analyzer::instrs_linear(instrs).count();
    if instr_count < min_instrs_to_compile() {
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::TooShort);
        return false;
    }

    let mut instrs_owned = *instrs;
    let func_id = codegen.compile_region_uncommitted(&mut instrs_owned, req.offset, req.compiled_for_fr1, true);
    match func_id {
        Some(func_id) => {
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            pending.push(PendingPublish { page: req.page, offset, func_id, gen_snap, instr_count, code_size });
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            page.denylist(offset);
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// Finalize and publish every `PendingPublish` accumulated so far — the
/// deferred counterpart to `handle_request`'s immediate `page.publish(...)`.
/// `codegen.finalize_batch` does the one `finalize_definitions()` call this
/// whole batch has been building up to; each pending entry's own `gen_snap`
/// re-check inside `page.publish` (same as the immediate path) safely drops
/// any entry whose page mutated since it was compiled, exactly as it always
/// has — batching doesn't change that safety property, only when the
/// finalize/publish happens. Clears `pending` on return regardless of
/// per-entry outcome; a no-op (and no `finalize_definitions()` call at all,
/// via `finalize_batch`'s own empty-slice short circuit) when `pending` is
/// already empty.
pub fn flush_pending_batch(codegen: &mut Codegen, pending: &mut Vec<PendingPublish>) {
    if pending.is_empty() {
        return;
    }
    let func_ids: Vec<cranelift_module::FuncId> = pending.iter().map(|p| p.func_id).collect();
    let jit_fns = codegen.finalize_batch(&func_ids);
    for (entry, jit_fn) in pending.drain(..).zip(jit_fns) {
        let page = unsafe { &*entry.page };
        page.publish(entry.offset, jit_fn as *const (), entry.gen_snap, entry.instr_count, entry.code_size);
    }
}

#[cfg(feature = "jitv2_corpus_dump")]
pub const CORPUS_DIR: &str = "jitv2_corpus";

#[cfg(feature = "jitv2_corpus_dump")]
fn dump_corpus_snapshot(page: &crate::jitv2::PhysicalCodePage, offset: u16, words: &[u32; ENTRIES_PER_PAGE]) {
    use std::io::Write;

    if !page.mark_saved(offset as usize) {
        return; // already dumped for this (pfn, offset)
    }
    let out_dir = std::path::Path::new(CORPUS_DIR);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("jitv2 corpus: failed to create {}: {}", CORPUS_DIR, e);
        return;
    }
    let path = out_dir.join(format!("pfn_{:08x}_off_{:04x}.bin", page.pfn, offset));
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        };
        f.write_all(bytes)
    };
    if let Err(e) = write() {
        eprintln!("jitv2 corpus: failed to write snapshot for pfn={:#010x} offset={:#06x}: {}", page.pfn, offset, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitv2::PhysicalCodePage;
    use crate::mips_isa::{OP_ADDIU, OP_SPECIAL};
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_ERR};
    use std::sync::atomic::AtomicU64;

    /// Test-only wrapper hiding `handle_request`'s `developer`-only `stats`
    /// parameter — every test below just wants "run a request," not the
    /// stats side effects, so a fresh throwaway `JitStats` under `developer`
    /// keeps every call site identical regardless of which build this runs
    /// under (`--features jitv2` alone must build and pass exactly like
    /// `--features jitv2,developer` does).
    fn handle_request_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request(req, bus, analyzer, codegen, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request(req, bus, analyzer, codegen)
        }
    }

    /// Every word decodes as `ADDIU r1, r0, 1` (opcode 0x09, rs=0, rt=1,
    /// imm=1) — a plain Sequential instruction with a real codegen emitter,
    /// so `handle_request` can exercise the whole walk+compile+publish path
    /// without needing a real memory device.
    struct AddiuDevice;
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok((OP_ADDIU << 26) | (1 << 16) | 1)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    /// Every word decodes as `SYSCALL` (§4.4 excluded instruction) — used to
    /// exercise `handle_request`'s denylist exit path deterministically.
    struct SyscallDevice;
    impl BusDevice for SyscallDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok((OP_SPECIAL << 26) | crate::mips_isa::FUNCT_SYSCALL)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    struct ErrDevice;
    impl BusDevice for ErrDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    #[test]
    fn handle_request_publishes_a_compilable_instruction() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(page.is_entry_valid(4), "a plain ADDIU must compile and publish");
        assert!(!page.is_denylisted(4));
    }

    #[test]
    fn handle_request_skips_already_valid_entry() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_entry_valid(0));

        // Bump gen so a naive re-compile would behave differently if it ran;
        // since is_entry_valid is already true, handle_request must bail
        // before touching the bus again.
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_entry_valid(0), "gen bump alone must not un-publish without going through it");
    }

    #[test]
    fn handle_request_leaves_offset_undecided_on_bus_error() {
        let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(!page.is_entry_valid(0));
        assert!(!page.is_denylisted(0), "a transient read failure must not become a sticky rejection");
    }

    /// Regression test: `exec_decoded`'s dispatch gate sets `ENTRY_SCHEDULED`
    /// (via `try_schedule`) before sending a `CompileRequest` to stop a hot
    /// PC from flooding the queue with duplicate requests for the same
    /// offset. `handle_request` must clear that flag again once it decides —
    /// on every exit path, not just the success path — or the offset can
    /// never be scheduled again after this one request, even once it's
    /// fully resolved. Covers all three decision outcomes: publish success,
    /// sticky denylist, and "bus not readable, retry later."
    #[test]
    fn handle_request_clears_scheduled_bit_on_every_outcome() {
        let counter = AtomicU64::new(0);

        // Outcome 1: publishes successfully.
        {
            let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            assert!(page.try_schedule(4));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_entry_valid(4));
            assert!(page.try_schedule(4), "scheduled bit must be cleared after a successful publish, so a future recompile request isn't blocked");
        }

        // Outcome 2: sticky-denylisted (SYSCALL is an excluded instruction,
        // §4.4 — entering directly on one yields a zero-instruction region).
        {
            let bus: Arc<dyn BusDevice> = Arc::new(SyscallDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            assert!(page.try_schedule(0));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_denylisted(0));
            assert!(page.try_schedule(0), "scheduled bit must be cleared after a denylist decision");
        }

        // Outcome 3: bus read fails, offset left undecided ("retry later").
        {
            let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
            let mut page = PhysicalCodePage::new(0, std::ptr::null());
            assert!(page.try_schedule(0));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(!page.is_entry_valid(0));
            assert!(!page.is_denylisted(0));
            assert!(page.try_schedule(0), "scheduled bit must be cleared even on a transient bus-read failure, so the offset can be retried");
        }
    }

    /// Same `stats`-hiding wrapper as `handle_request_for_test`, for the
    /// deferred path.
    fn handle_request_deferred_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen, pending: &mut Vec<PendingPublish>) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request_deferred(req, bus, analyzer, codegen, pending, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request_deferred(req, bus, analyzer, codegen, pending)
        }
    }

    #[test]
    fn handle_request_deferred_accumulates_pending_without_publishing() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending = Vec::new();
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert_eq!(pending.len(), 1, "a successful deferred compile must accumulate exactly one PendingPublish");
        assert!(!page.is_entry_valid(4), "must not be published until flush_pending_batch runs");
    }

    #[test]
    fn flush_pending_batch_publishes_every_accumulated_entry() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page_a = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let mut page_b = PhysicalCodePage::new(1, &counter as *const AtomicU64);
        let req_a = CompileRequest { page: &mut page_a as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };
        let req_b = CompileRequest { page: &mut page_b as *mut PhysicalCodePage, offset: 8, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending = Vec::new();
        handle_request_deferred_for_test(&req_a, &bus, &mut analyzer, &mut codegen, &mut pending);
        handle_request_deferred_for_test(&req_b, &bus, &mut analyzer, &mut codegen, &mut pending);
        assert_eq!(pending.len(), 2);

        flush_pending_batch(&mut codegen, &mut pending);

        assert!(pending.is_empty(), "flush_pending_batch must drain pending on return");
        assert!(page_a.is_entry_valid(4), "first batched entry must be published");
        assert!(page_b.is_entry_valid(8), "second batched entry must be published");
    }

    #[test]
    fn flush_pending_batch_on_empty_pending_is_a_harmless_noop() {
        let mut codegen = Codegen::new();
        let mut pending: Vec<PendingPublish> = Vec::new();
        flush_pending_batch(&mut codegen, &mut pending); // must not panic
        assert!(pending.is_empty());
    }

    #[test]
    fn handle_request_deferred_denylists_excluded_entry_without_accumulating() {
        let bus: Arc<dyn BusDevice> = Arc::new(SyscallDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending = Vec::new();
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert!(pending.is_empty(), "an excluded entry produces no FuncId to defer");
        assert!(page.is_denylisted(0));
    }

    #[test]
    fn handle_request_deferred_clears_scheduled_bit() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(page.try_schedule(4));
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending = Vec::new();
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert!(page.try_schedule(4), "scheduled bit must be cleared even though publish itself is deferred");
    }

    #[test]
    fn flush_pending_batch_skips_an_entry_whose_page_mutated_since_compile() {
        // Same gen_snap staleness check `page.publish` already applies on
        // the immediate path — batching must not weaken it: an entry
        // compiled against gen N, flushed after the page bumped to gen N+1,
        // must be silently dropped rather than published against stale
        // content.
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending = Vec::new();
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);
        assert_eq!(pending.len(), 1);

        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed); // page "mutates"

        flush_pending_batch(&mut codegen, &mut pending);
        assert!(!page.is_entry_valid(4), "a batch entry compiled against a stale gen must not be published");
    }
}
