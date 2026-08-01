//! JIT v2 core data structures.
//!
//! See `rules/jitv2/jit-v2-design.md` for the full design. This module holds the
//! per-page metadata the rest of the engine builds on (§2.4):
//!
//! - [`PhysicalCodePage`]: the mips executor's view of "what's compiled here" for
//!   the physical page it's currently executing out of.
//! - Generation counters live in the owning `BusDevice` (one per page for RAM,
//!   a single shared never-bumped counter for ROM — `BusDevice::gen_ptr`,
//!   `src/mem.rs`, `src/prom.rs`) and are read through a raw pointer here so the
//!   hot path avoids an indirect call through the device trait object.
//!
//! Threading model: the mips exec thread owns `PhysicalCodePage` management
//! (arrival, promotion — §6.1) and pushes compile requests to the compile thread
//! over an SPSC fifo (§6.4); the compile thread publishes finished artifacts back
//! into the page's `entry_table`/`entry_bits` (§6.1.3), which is why requests carry
//! a mutable pointer. Only the compile-request queue itself is added in this pass
//! — the compile thread and publish path land with codegen (Phase 2).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::mips_core::MipsCore;
use crate::mips_exec::ExecStatus;
use crate::traits::{BusDevice, Device};

/// Physical frame number. Physical addresses are keyed by PFN, never by VA (§2.1).
pub type Pfn = u32;

/// Minimum interpreter dispatches an offset must accumulate before
/// `exec_decoded`'s dispatch gate (`mips_exec.rs`) will send its first
/// `CompileRequest` — `j2 min-calls [N]` tunes this at runtime. A nonzero
/// value trades a few extra interpreted dispatches of a genuinely-cold
/// offset for never paying compile cost on paths that only ever run once or
/// twice — real hot loops clear any small threshold within a handful of
/// iterations, so this mostly filters out one-shot/rare code, not anything
/// actually hot. See `PhysicalCodePage::count_dispatch_and_check_threshold`
/// for the counter mechanism (reuses the per-entry `gen` slot, unused before
/// first publish). Callers that must stay deterministic/immediate (tests,
/// `jitv2_inline_compile`, lockstep/verification harnesses — none of which
/// go through `exec_decoded`'s real gate at all, `jitv2_lockstep` least of
/// all since it's a separate code path entirely) are exempt by construction,
/// not by passing a special value: they never call
/// `count_dispatch_and_check_threshold`, so this setting never applies to
/// them regardless of its value.
///
/// Defaults to 0 ("always ready," the original behavior) under `developer` —
/// diagnostics builds want every compilable offset compiled immediately, not
/// filtered by a production-tuned call-count floor — and to 4 otherwise.
static MIN_CALLS_BEFORE_COMPILE: AtomicU64 = AtomicU64::new(if cfg!(feature = "developer") { 0 } else { 4 });

pub fn set_min_calls_before_compile(n: u64) {
    MIN_CALLS_BEFORE_COMPILE.store(n, Ordering::Relaxed);
}

pub fn min_calls_before_compile() -> u64 {
    MIN_CALLS_BEFORE_COMPILE.load(Ordering::Relaxed)
}

/// Page size for JIT v2 (§2.4) — matches the MIPS TLB/cache page granularity
/// used throughout the codebase. Canonical home for this constant; `mem.rs`
/// re-exports it as `JITV2_PAGE_SIZE` for its own generation-counter indexing.
pub const PAGE_SIZE: u32 = 4096;

/// Number of possible entry offsets per page: one per 4-byte-aligned word
/// (MIPS instructions are always word-aligned), i.e. `PAGE_SIZE / 4` (§2.4:
/// `entry_bits` 16×u64 = 1024 bits, `entry_table` 1024 entries).
pub const ENTRIES_PER_PAGE: usize = (PAGE_SIZE / 4) as usize;

/// u64 words needed for a 1-bit-per-entry bitmap over `ENTRIES_PER_PAGE` offsets.
pub const BITMAP_WORDS: usize = ENTRIES_PER_PAGE / 64;

/// Compiled-function ABI (§6.1.2's "handler ABI", simplified for this storage
/// pass — no `DecodedInstr`/state-struct plumbing yet, just direct MipsCore
/// access): takes a pointer to the executor's `MipsCore` and returns the same
/// `ExecStatus` every interpreter handler returns. `vbase` derivation, the
/// two mirrored checks (§3.2), and exit-stub materialization all live inside
/// the compiled body once codegen exists — this signature is just the call
/// boundary.
pub type JitFn = unsafe extern "C" fn(*mut MipsCore) -> ExecStatus;

/// Single entry in a page's compiled-function table (§2.4 `entry_table`).
/// AoS layout (one `JitEntry` per offset) rather than the design doc's literal
/// SoA (`entry_bits` bitmap + separate `entry_table` pointer array): `gen` is
/// consulted together with `func` at every dispatch (staleness check against
/// the page's current generation, §4.1/§6.5), so keeping them in the same
/// cache line avoids a second, unrelated array touch on the hot path. The
/// separate `valid_bits`/`denylist_bits` arrays in `PhysicalCodePage` remain
/// SoA because each is scanned/tested independently of the others (bitmap
/// probe on un-promoted arrivals vs. sticky-rejection check, §6.1/§6.4).
pub struct JitEntry {
    /// Compiled function pointer for this offset, or null if unpublished.
    /// Validity is owned entirely by `PhysicalCodePage::valid_bits` (§6.1.2's
    /// "the remove check IS this load" — a raw pointer, not `Option`, keeps
    /// that the single source of truth instead of letting callers branch on
    /// `func.is_some()` as a second, potentially-stale answer to the same
    /// question). Callers must check the valid bit before calling this.
    pub func: *const (),
    /// Generation this entry was compiled against (§6.5 `gen_snap`). An entry
    /// is valid iff `gen == page.current_gen()` — mismatch means the page
    /// mutated since compilation and the entry must be treated as stale
    /// (downgrade to interpreter, §6.1.2) regardless of what `valid_bits` says.
    pub gen: AtomicU64,
    /// Dev-only diagnostics for `j2 pcp`: how many instructions this
    /// entry's compiled region covers (set once at publish time — not
    /// atomic since it's write-once-then-read, same lifecycle as `func`)
    /// and how many times `exec_decoded`'s dispatch gate has actually
    /// called this entry's `func` (incremented on every dispatch, hence
    /// atomic — the exec thread is the only writer today, but this is read
    /// concurrently from the monitor console thread via `j2 pcp`). Added to
    /// help diagnose "lockstep boots fine but normal dispatch stalls
    /// somewhere" class bugs: a hot loop stuck calling the same
    /// under-sized/wrong region over and over shows up immediately as one
    /// offset's `call_count` growing without bound while PC-visible
    /// progress stalls, without needing to add ad-hoc eprintln!s each time.
    #[cfg(feature = "developer")]
    pub instr_count: u16,
    /// Dev-only diagnostic (`j2 pcp`/`j2 stats`): size in bytes of this
    /// entry's compiled machine code, read from Cranelift's own
    /// `CompiledCode::buffer` right after `define_function` — see
    /// `Codegen::compile_region`'s doc comment. Summed across every
    /// published entry (`Jitv2::code_bytes_used`) as the best available
    /// proxy for the shared `Codegen`'s Cranelift memory-arena size, since
    /// `cranelift_jit::Memory` exposes no size/usage API of its own
    /// (`pub(crate)`, not reachable from outside the crate).
    #[cfg(feature = "developer")]
    pub code_size: u32,
    #[cfg(feature = "developer")]
    pub call_count: AtomicU64,
}

impl Default for JitEntry {
    fn default() -> Self {
        Self {
            func: std::ptr::null(),
            gen: AtomicU64::new(0),
            #[cfg(feature = "developer")]
            instr_count: 0,
            #[cfg(feature = "developer")]
            code_size: 0,
            #[cfg(feature = "developer")]
            call_count: AtomicU64::new(0),
        }
    }
}

// Safety: `func`, when non-null, points to finalized JIT-compiled code owned
// by the compile-thread arena, which outlives every PhysicalCodePage entry
// referencing it until the next mega_flush (mirrors PhysicalCodePage's own
// Send/Sync rationale for `gen`).
unsafe impl Send for JitEntry {}
unsafe impl Sync for JitEntry {}

/// Process-wide JIT event counters, displayed by `j2 status` only under the
/// `developer` feature — plain `AtomicU64`s rather than per-`Jitv2` state
/// protected by a lock, since these are touched from both the compile
/// thread (`comp::handle_request`, on every request, inline or threaded)
/// and the CPU thread (`jit_kill_entry`, on every FR-mismatch bail —
/// `mips_exec.rs`) and none of them need to be read together atomically
/// with anything else; a lock here would just be hot-path contention for no
/// correctness benefit. The struct and its `Arc<JitStats>` field on `Jitv2`
/// exist unconditionally (not `#[cfg(feature = "developer")]`) so nothing
/// on the path from `Jitv2::new`/`CompileQueue::start` down to
/// `handle_request` needs a second, feature-gated copy of itself just to
/// thread one more `Arc` through — only the actual `.fetch_add()` call
/// sites and the `j2 status` display are gated. Survives `mega_flush`
/// deliberately (counts are lifetime totals, not "since last reset" —
/// `Codegen::function_count`'s own since-reset counter already covers that
/// angle).
/// Why a compile request was declined — `failed_compiles`'s single counter
/// doesn't distinguish these, but they're different enough situations to
/// want separately (`j2 status`'s "rejections by reason" breakdown):
/// a codegen gap you could go implement an emitter for reads very
/// differently from a Cranelift verifier bug, which reads differently again
/// from "the analyzer and codegen's emitter tables disagree" (should be
/// structurally impossible after `opcode_support::has_emitter` unified them —
/// a nonzero count here means they've drifted again).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum RejectReason {
    /// `walk_bounded` reported the entry offset itself unreachable —
    /// architecturally excluded (COP0, CACHE, LL/SC, ...) or a codegen gap
    /// at the very first instruction, `analyzer::classify` returned
    /// `Excluded` either way (`comp::handle_request`'s `!non_empty` arm).
    EntryExcluded,
    /// `compile_region`'s upfront rejection loop found a visited instruction
    /// with no emitter in any of the four lookup tables. Per
    /// `opcode_support::has_emitter` being the single source of truth both
    /// `analyzer::classify` and this loop consult, this should be
    /// unreachable in practice — every instruction the analyzer walked past
    /// as `Sequential`/`Branch`/`Jump`/`RegJump` already passed the same
    /// check. Tracked anyway as a canary: a nonzero count here means the
    /// tables have drifted apart again, the exact bug class
    /// `opcode_support.rs` was built to close structurally.
    AnalyzerCodegenDisagreement,
    /// Cranelift's `define_function` returned `ModuleError::Compilation` —
    /// this module emitted IR the verifier rejected, a real bug in some
    /// emitter, not an unsupported-instruction-shape decline (those are all
    /// caught by the upfront loop before Cranelift is ever invoked).
    CraneliftVerifierError,
    /// The walked region's instruction count fell below
    /// `comp::min_instrs_to_compile()` (`j2 min-instrs`) — not a codegen gap
    /// at all, just not judged worth the fixed per-compile cost. See
    /// `comp::MIN_INSTRS_TO_COMPILE`'s own doc comment.
    TooShort,
}
#[cfg(feature = "developer")]
pub const REJECT_REASON_COUNT: usize = 4;
#[cfg(feature = "developer")]
impl RejectReason {
    pub fn index(self) -> usize {
        match self {
            RejectReason::EntryExcluded => 0,
            RejectReason::AnalyzerCodegenDisagreement => 1,
            RejectReason::CraneliftVerifierError => 2,
            RejectReason::TooShort => 3,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::EntryExcluded => "entry excluded (unsupported first instruction / architectural exclusion)",
            RejectReason::AnalyzerCodegenDisagreement => "analyzer/codegen disagreement (should be unreachable — see doc comment)",
            RejectReason::CraneliftVerifierError => "Cranelift verifier error (real emitter bug)",
            RejectReason::TooShort => "region too short (below j2 min-instrs threshold)",
        }
    }
}

#[derive(Default)]
pub struct JitStats {
    /// Successful `compile_region` calls that got published
    /// (`comp::handle_request`'s `Some(jit_fn)` arm).
    pub compiles: AtomicU64,
    /// `compile_region` declines or `walk_bounded` exclusions
    /// (`comp::handle_request`'s two `page.denylist(offset)` sites) — codegen
    /// gaps and analyzer-excluded entry offsets alike, not distinguished
    /// further (both end up sticky-denylisted the same way). See
    /// `reject_reasons` for the breakdown by cause.
    pub failed_compiles: AtomicU64,
    /// Per-`RejectReason` breakdown of `failed_compiles`, indexed by
    /// `RejectReason::index()`. Excludes the arena-out-of-memory outcome
    /// (`comp::handle_request`'s `last_compile_ran_out_of_memory` arm) —
    /// that one isn't a rejection of the *region*, it retries on its own,
    /// so it doesn't belong in a "why did this instruction never compile"
    /// breakdown; it's counted in `failed_compiles` but not here.
    #[cfg(feature = "developer")]
    pub reject_reasons: [AtomicU64; REJECT_REASON_COUNT],
    /// `jit_kill_entry` calls — a compiled unit's FR-mode-mismatch guard
    /// bailing and un-publishing its own entry (`emit_fpu_entry_guard`'s CU1/FR
    /// mismatch design, `MipsCore::kill_entry_fn`'s doc comment).
    pub kill_entry_calls: AtomicU64,
    /// Total `CompileQueue::send` calls (both accepted and dropped) —
    /// `j2 status`'s FIFO-fullness section denominator. `developer`-only:
    /// unlike `compiles`/`failed_compiles`/`kill_entry_calls` (rare events,
    /// off the hot path), `send` runs on every dispatch-gate arrival that
    /// misses the JIT cache — real per-instruction-adjacent traffic, so the
    /// extra atomic touch these three fields cost is worth avoiding outside
    /// a diagnostics build.
    #[cfg(feature = "developer")]
    pub compile_queue_dispatches: AtomicU64,
    /// `CompileQueue::send` calls that dropped the request because the ring
    /// buffer (`COMPILE_QUEUE_CAPACITY`) was already full — see `send`'s own
    /// doc comment for why a drop here is a normal, non-fatal outcome (the
    /// hot page just re-triggers the request on a later arrival), not an
    /// error; this counter exists to tell whether that's happening often
    /// enough to be a real bottleneck, not a rare edge case.
    #[cfg(feature = "developer")]
    pub compile_queue_full: AtomicU64,
    /// Running sum of `Producer::slots()`-derived occupancy
    /// (`capacity - free_slots`) sampled at every `send` call — divide by
    /// `compile_queue_dispatches` for the mean queue depth the compile
    /// thread is actually running at. A cumulative sum rather than a
    /// separate min/max/histogram: cheap (one more atomic add on an
    /// already-atomic-touching path), and mean depth is the number that
    /// actually answers "is the queue usually near-empty (compile thread
    /// keeping up) or usually near-full (compile thread falling behind)."
    #[cfg(feature = "developer")]
    pub compile_queue_depth_sum: AtomicU64,
    /// `flush_pending_batch` calls triggered by `Codegen::provider_crossed_page`
    /// (a batch that filled its host-page segment) — `j2 stats`'s batching
    /// section, split from `batch_flushes_queue_drain` so it's possible to
    /// tell whether batches are actually reaching page-fill in practice or
    /// are mostly getting cut short by the queue draining first (a sign
    /// `MAX_INSTRS_PER_COMPILE`-sized regions or a slow request rate are
    /// preventing batches from ever growing large).
    #[cfg(feature = "developer")]
    pub batch_flushes_page_cross: AtomicU64,
    /// `flush_pending_batch` calls triggered by the compile queue draining
    /// empty (`worker_loop`'s `Err(_)` arm) — see `batch_flushes_page_cross`.
    #[cfg(feature = "developer")]
    pub batch_flushes_queue_drain: AtomicU64,
    /// High-water mark of `pending.len()` just before any `flush_pending_batch`
    /// call actually finalized something (i.e. sampled right before the
    /// batch drains, not after) — `j2 stats`'s "biggest batch we ever
    /// packed" figure. A `Relaxed` compare-and-swap loop, not a simple
    /// `fetch_max` (stabilized in std but this codebase's MSRV predates it
    /// for `AtomicU64` — matches the CAS-loop pattern already used elsewhere
    /// in this codebase for high-water-mark tracking).
    #[cfg(feature = "developer")]
    pub batch_max_pending: AtomicU64,
}

/// Why `flush_pending_batch` was called — see `JitStats::batch_flushes_page_cross`/
/// `batch_flushes_queue_drain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum BatchFlushReason {
    PageCross,
    QueueDrain,
}

#[cfg(feature = "developer")]
impl JitStats {
    /// Bump both `failed_compiles` and the per-reason breakdown together —
    /// the two call sites (`comp::handle_request`'s two denylist arms,
    /// `Codegen::compile_region`'s two `None` returns) always want to record
    /// both, and a shared helper means they can't drift out of sync (one
    /// incremented without the other).
    pub fn record_reject(&self, reason: RejectReason) {
        self.failed_compiles.fetch_add(1, Ordering::Relaxed);
        self.reject_reasons[reason.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Record one `flush_pending_batch` call: bump the trigger-specific
    /// counter and update the high-water mark from `pending_len` (the batch
    /// size just before this flush drained it).
    pub fn record_batch_flush(&self, pending_len: usize, reason: BatchFlushReason) {
        match reason {
            BatchFlushReason::PageCross => self.batch_flushes_page_cross.fetch_add(1, Ordering::Relaxed),
            BatchFlushReason::QueueDrain => self.batch_flushes_queue_drain.fetch_add(1, Ordering::Relaxed),
        };
        let pending_len = pending_len as u64;
        let mut cur = self.batch_max_pending.load(Ordering::Relaxed);
        while pending_len > cur {
            match self.batch_max_pending.compare_exchange_weak(cur, pending_len, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Default page-pool capacity for `Jitv2::new()` as embedded in `MipsExecutor`.
/// Sizing is a Phase 0 measurement per the design doc (§9, "Max live entries per
/// epoch"); `mega_flush` absorbs it being wrong in either direction. Now that
/// the whole pool is a single array allocated once at this capacity (see
/// `Jitv2::new`'s doc comment — `PhysicalCodePage` is no longer boxed inside,
/// so a larger capacity is a real memory cost, not just reserved address
/// space), 4096 is a deliberately modest working-set size rather than a
/// generous upper bound — `mega_flush`'s cost of getting this wrong low
/// (a pool-exhaustion flush) is cheap relative to permanently carrying a much
/// larger array.
pub const JITV2_INITIAL_PAGE_CAPACITY: usize = 4096;

/// Flush threshold for the shared `Codegen`'s Cranelift arena, in bytes
/// actually reserved (`Codegen::packing_stats()`'s `reserved` — real
/// host-page-rounded arena footprint, not the function-count proxy this
/// constant used before batching landed). `cranelift_jit::Memory` never
/// frees on drop/replace (`Codegen::reset`'s own doc comment), so nothing
/// else bounds arena growth — a long-enough-running compile (real IRIX boot,
/// not just PROM) will otherwise exhaust the whole `Codegen::ARENA_RESERVE_SIZE`
/// reservation.
///
/// Function count stopped being a good proxy for arena growth once
/// deferred-finalize batching (`j2 batch`) started letting many small
/// functions pack into a shared host-page segment instead of each getting
/// its own — the byte size actually reserved is now directly measurable
/// (`PagedArenaState`), so there's no reason to keep estimating it from a
/// count. 128MiB leaves comfortable headroom under `Codegen::ARENA_RESERVE_SIZE`
/// (512MiB) for the batch that happens to be in flight when this trips (a
/// batch isn't finalized/counted until it flushes, so the real reservation
/// can run slightly ahead of this threshold between checks) while still
/// flushing well before the arena's own exhaustion error could ever fire —
/// that error path (`comp::handle_request`'s exhaustion match arm) stays as
/// a belt-and-suspenders backstop, not the primary trigger.
pub const CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES: u64 = 128 * 1024 * 1024;

/// A compile request pushed from the mips exec thread to the compile thread
/// over the SPSC fifo (§6.4). Carries the page pointer rather than a snapshotted
/// generation: the compile thread reads `gen` itself at snapshot time (§6.5 step
/// 2, `gen_snap = gen`) and re-reads it at publish time — the generation at
/// queue time is never consulted, only current-at-compile and current-at-publish.
/// The pointer is mutable because publish (§6.1.3) writes into the page's
/// `entry_table`/`entry_bits`/`artifact_list`.
///
/// # Safety
/// `page` must outlive the request — pages live for the lifetime of their
/// owning device (see [`PhysicalCodePage`]'s Send/Sync safety note).
#[derive(Debug)]
pub struct CompileRequest {
    pub page: *mut PhysicalCodePage,
    pub offset: u16,
    /// Live `STATUS_FR` bit at enqueue time, threaded through because the
    /// compile thread has no `MipsCore` to read it from itself — codegen's
    /// FPR-access emitters are FR-mode-specific and must match whatever mode
    /// the executor will actually be in when it calls the compiled function
    /// (same value `exec_decoded` used to run the interpreter fallback for
    /// this same arrival).
    pub compiled_for_fr1: bool,
}

unsafe impl Send for CompileRequest {}

/// Per-physical-page code cache metadata, as tracked by the mips executor
/// (§2.4). One instance per physical RAM/ROM page that has ever been a JIT
/// compilation target; the executor holds a pointer to the page it is
/// currently executing out of.
///
/// Does not yet own `queued_bits`/`artifact_list` (§2.4) — those land with
/// the compile-thread/dispatcher work. `valid_bits`, `denylist_bits`, and
/// `entries` (this pass) are the `entry_bits`/`entry_table` pair from the
/// design doc, laid out AoS per-entry (see `JitEntry`) rather than the
/// document's literal SoA split.
pub struct PhysicalCodePage {
    pub pfn: Pfn,
    /// Pointer to this page's generation counter, obtained from the owning
    /// `BusDevice` via `gen_ptr` (§2.4, §7). RAM devices return one counter
    /// per page; ROM devices point every page at a single counter that is
    /// initialized to 0 and never bumped, since ROM content is immutable.
    /// Null if the page's device doesn't back JIT-compilable memory (MMIO).
    gen: *const AtomicU64,
    /// One bit per entry offset (word-aligned, §2.4): set iff `entries[i]`
    /// holds a published, dispatchable function. Authoritative for dispatch
    /// (probed on un-promoted arrivals, §6.1) and the kill path — release-set
    /// by publish, cleared by kill. This bit is the single source of truth
    /// for "is this entry live"; `entries[i].func` being non-null is not
    /// itself checked (kill nulls the bit but may leave `func` populated
    /// until the slot is reused — see `JitEntry::func`'s doc comment).
    pub valid_bits: [AtomicU64; BITMAP_WORDS],
    /// One bit per entry offset: set iff this offset was permanently refused
    /// by the compiler (too-short region below the yield threshold, excluded
    /// first instruction, 0xFFC/slot hazard, etc — §6.4 "sticky rejection").
    /// Consulted by arrival/queueing to stop re-requesting a compile that will
    /// only be declined again; cleared on a gen bump (re-classify against new
    /// bytes) alongside `valid_bits`.
    pub denylist_bits: [AtomicU64; BITMAP_WORDS],
    /// One bit per entry offset: set iff a `CompileRequest` for this offset
    /// has been sent to the async compile-thread queue and hasn't been
    /// decided yet (published or denylisted). `exec_decoded`'s dispatch gate
    /// consults this before building/sending another request for the same
    /// offset — without it, every dispatch of a not-yet-compiled offset that
    /// keeps re-satisfying the gate's trigger conditions (e.g. a hot loop
    /// back-edge landing on the same still-uncompiled word every iteration,
    /// or `jit_trigger` now also being set by JIT-to-JIT jump exits — see
    /// `MipsCore::jit_trigger`) sends a fresh, redundant `CompileRequest`
    /// for the exact same (page, offset) every single time, flooding the
    /// compile-thread's queue with requests that all do the same work.
    /// `try_schedule` is the only setter (a compare-exchange, so only the
    /// first caller for a given offset actually wins and sends); cleared by
    /// `clear_scheduled` once `handle_request` (`comp.rs`) has decided the
    /// offset one way or the other, so a later legitimate re-request (e.g.
    /// after a gen bump invalidates a stale artifact) isn't permanently
    /// blocked. Irrelevant to the synchronous `jitv2_inline_compile` path —
    /// that path can't re-enter before `comp::handle_request` returns, so
    /// there's no queue to flood.
    pub scheduled_bits: [AtomicU64; BITMAP_WORDS],
    /// Per-offset compiled-function slots (§2.4 `entry_table`). Inline, not
    /// boxed: `Jitv2::pages` is a single array allocated once, up front, at
    /// full capacity (`Jitv2::new`'s own doc comment) — every
    /// `PhysicalCodePage` is constructed exactly once and never moved again
    /// (no more `Vec::push` growing the pool one page at a time), so the
    /// "avoid copying the 1024-entry table on move" concern a `Box` used to
    /// exist for doesn't apply anymore; the indirection would just be extra
    /// pointer-chasing on every entry access for no benefit.
    pub entries: [JitEntry; ENTRIES_PER_PAGE],
    /// Scaffolding for corpus collection only (`jitv2/comp.rs`) — NOT part of
    /// the design doc's per-page metadata (§2.4). One bit per entry offset:
    /// set once that (pfn, offset) pair's page snapshot has been dumped to
    /// `jitv2_corpus/`, so a hot page revisited many times only gets saved
    /// once. Safe to delete once the real compiler (reachability walk +
    /// codegen) replaces the dump-to-disk stub in the worker loop.
    pub saved_bits: [AtomicU64; BITMAP_WORDS],
}

// Safety: `gen` points into the owning BusDevice's storage, which outlives
// every PhysicalCodePage derived from it (devices are held in Arcs/statics
// for the lifetime of the machine). The pointee is only ever read or
// atomically incremented, never moved or freed while a PhysicalCodePage
// referencing it exists.
unsafe impl Send for PhysicalCodePage {}
unsafe impl Sync for PhysicalCodePage {}

impl PhysicalCodePage {
    /// Construct an as-yet-unclaimed page descriptor: `pfn = 0`, `gen =
    /// null` (the same "not compilable" sentinel `is_compilable()` already
    /// checks everywhere), every bitmap zeroed, every entry at its default
    /// (unpublished) state. Used both to build `Jitv2::pages`' full-capacity
    /// array up front (every slot starts unclaimed) and, functionally
    /// identically, by [`Self::claim`]/[`Self::reset_in_place`] to return a
    /// slot to this same state in place without reallocating anything.
    pub fn new(pfn: Pfn, gen: *const AtomicU64) -> Self {
        Self {
            pfn,
            gen,
            valid_bits: std::array::from_fn(|_| AtomicU64::new(0)),
            denylist_bits: std::array::from_fn(|_| AtomicU64::new(0)),
            scheduled_bits: std::array::from_fn(|_| AtomicU64::new(0)),
            entries: std::array::from_fn(|_| JitEntry::default()),
            saved_bits: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Zero every bitmap and reset every entry to its default (unpublished)
    /// state, in place — no reallocation, `entries`'s 1024 slots are written
    /// through, not replaced. Called only from [`Self::reset_to_unclaimed`]
    /// (`Jitv2::mega_flush`'s per-slot reset) — a fresh, never-claimed slot
    /// is already zeroed by `PhysicalCodePage::new` and doesn't need this
    /// (see [`Self::claim`]'s doc comment for why re-running it there on
    /// every ordinary page arrival would be wasted, hot-path work).
    ///
    /// Explicitly zeroing every `entries[i].gen` here (not just `func`/
    /// `valid_bits`) is the important part: that field doubles as
    /// `PhysicalCodePage::count_dispatch_and_check_threshold`'s pre-publish
    /// call counter (`j2 min-calls`) when the entry has never been
    /// published — without this, a slot reused after a flush would inherit
    /// whatever count was sitting there from its previous physical page's
    /// occupancy, silently skewing the new page's own warm-up window.
    fn reset_entries_and_bitmaps(&mut self) {
        for word in self.valid_bits.iter() { word.store(0, std::sync::atomic::Ordering::Relaxed); }
        for word in self.denylist_bits.iter() { word.store(0, std::sync::atomic::Ordering::Relaxed); }
        for word in self.scheduled_bits.iter() { word.store(0, std::sync::atomic::Ordering::Relaxed); }
        for word in self.saved_bits.iter() { word.store(0, std::sync::atomic::Ordering::Relaxed); }
        for entry in self.entries.iter_mut() {
            entry.func = std::ptr::null();
            entry.gen.store(0, std::sync::atomic::Ordering::Relaxed);
            #[cfg(feature = "developer")]
            {
                entry.instr_count = 0;
                entry.code_size = 0;
                entry.call_count.store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Bump-allocate this (already-constructed, already-clean) slot into
    /// service for a newly-arrived physical page — the in-place counterpart
    /// to what used to be a fresh `PhysicalCodePage::new(pfn, gen)` +
    /// `Vec::push`. Deliberately does **not** reset bitmaps/entries itself —
    /// `page_for` calls this on every distinct-PFN arrival, which happens
    /// constantly during real execution (every newly-touched physical page,
    /// not just at startup), so re-zeroing all 1024 entries here on the hot
    /// path would be wasted work almost every time: a slot only ever reaches
    /// `claim` in one of two states, both already clean — freshly
    /// constructed (`PhysicalCodePage::new` zeroes everything) or freshly
    /// reset by `mega_flush` (`reset_to_unclaimed`, called on exactly the
    /// slots it's returning to the unclaimed pool, before any of them can be
    /// claimed again). The `debug_assert!` below exists specifically to
    /// catch a violation of that invariant (e.g. a future caller reusing a
    /// slot without going through `mega_flush` first) in a diagnostics
    /// build, rather than paying a real-build cost to re-verify it on every
    /// single claim.
    pub fn claim(&mut self, pfn: Pfn, gen: *const AtomicU64) {
        debug_assert!(!self.is_compilable() && self.pfn == 0,
            "claim() called on a slot that wasn't clean (pfn={}, is_compilable={}) — every path that reuses a slot must reset it first (see reset_to_unclaimed)",
            self.pfn, self.is_compilable());
        debug_assert!((0..ENTRIES_PER_PAGE).all(|i| !self.is_published(i)),
            "claim() called on a slot with a still-published entry — mega_flush must reset_to_unclaimed before this slot can be reused");
        self.pfn = pfn;
        self.gen = gen;
    }

    /// Return this slot to the fully-unclaimed state (`pfn = 0`, `gen =
    /// null`) — `Jitv2::mega_flush`'s in-place counterpart to what used to
    /// be `Vec::clear()` dropping every `PhysicalCodePage` outright.
    pub fn reset_to_unclaimed(&mut self) {
        self.reset_entries_and_bitmaps();
        self.pfn = 0;
        self.gen = std::ptr::null();
    }

    /// Whether this page's backing device supports JIT generation tracking
    /// at all (i.e. `gen_ptr` returned non-null).
    #[inline]
    pub fn is_compilable(&self) -> bool {
        !self.gen.is_null()
    }

    /// Current generation count for this page. Panics in debug builds if the
    /// page isn't compilable (see [`Self::is_compilable`]) — callers on the
    /// hot path are expected to have already branched on that.
    #[inline]
    pub fn current_gen(&self) -> u64 {
        debug_assert!(self.is_compilable());
        // Relaxed: publish-time (§6.5) and mutation-time (§7) orderings are
        // established by the fetch_or/re-read pair at those call sites, not here.
        unsafe { (*self.gen).load(std::sync::atomic::Ordering::Relaxed) }
    }

    /// Whether `entries[offset_word]`'s valid bit is set — i.e. some compile
    /// has published a function here, without regard to whether it's still
    /// fresh against the page's current gen. Callers that need "is this
    /// dispatchable right now" want [`Self::is_entry_valid`]; this is for the
    /// dispatch-trigger gate (§6.1.2's `entry_bits[pfn].test(offset)`), which
    /// probes the bit first and only then decides exec-vs-recompile from gen.
    #[inline]
    pub fn is_published(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.valid_bits[word].load(std::sync::atomic::Ordering::Acquire) & (1 << bit) != 0
    }

    /// Whether `entries[offset_word]` is both published (`valid_bits`) and
    /// still fresh (its recorded gen matches the page's current gen, §6.5).
    /// A set valid bit whose gen has drifted is stale — the caller should
    /// treat it as unpublished (downgrade to interpreter, §6.1.2) rather than
    /// dispatch it.
    ///
    /// The `gen` load is Acquire, not Relaxed: it's the real synchronization
    /// point for `entries[offset_word].func` (see `publish`'s doc comment) —
    /// `valid_bits` alone doesn't provide fresh ordering across a recompile
    /// of an already-published entry, since the bit's *value* doesn't change
    /// on that path. Callers must not read `func` after this returns `true`
    /// without going through this same `gen` load (i.e. don't cache
    /// `is_published`'s result and reuse it — always call `is_entry_valid`
    /// fresh right before trusting `func`), or the ordering guarantee is
    /// lost.
    #[inline]
    pub fn is_entry_valid(&self, offset_word: usize) -> bool {
        self.is_published(offset_word)
            && self.entries[offset_word].gen.load(std::sync::atomic::Ordering::Acquire) == self.current_gen()
    }

    /// Whether `offset_word` has been sticky-rejected by the compiler (§6.4).
    #[inline]
    pub fn is_denylisted(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.denylist_bits[word].load(std::sync::atomic::Ordering::Relaxed) & (1 << bit) != 0
    }

    /// Count one interpreter dispatch of `offset_word` *before* it has ever
    /// been compiled, returning `true` once `threshold` dispatches have been
    /// observed (the caller should send a `CompileRequest` now) and `false`
    /// otherwise (not hot enough yet — stay on the interpreter this time).
    ///
    /// Reuses `entries[offset_word].gen` as the counter storage: that field
    /// has no meaning before an entry is ever published (`is_entry_valid`
    /// only ever reads it after first checking `is_published`, per that
    /// method's own doc comment — a pre-publish `gen` value is simply never
    /// consulted by anything), so borrowing it here for an unrelated purpose
    /// during that same pre-publish window is safe: the two uses never
    /// overlap in time for a given entry, and `publish()` unconditionally
    /// overwrites it with the entry's real `gen_snap` the moment it does get
    /// compiled, erasing whatever count was here. No new field needed.
    ///
    /// `threshold == 0` means "always ready" (returns `true` on the very
    /// first call, matching today's behavior with the gate absent entirely)
    /// — the caller (`exec_decoded`) is expected to pass 0 for every
    /// dispatch path that must stay deterministic/immediate (tests,
    /// `jitv2_inline_compile`'s synchronous "run it now" contract,
    /// verification harnesses), and the real tunable value
    /// (`j2 min-calls`) otherwise.
    #[inline]
    pub fn count_dispatch_and_check_threshold(&self, offset_word: usize, threshold: u64) -> bool {
        if threshold == 0 {
            return true;
        }
        let prev = self.entries[offset_word].gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        prev + 1 >= threshold
    }

    /// Sticky-reject `offset_word` (§6.4): the compiler declined this offset
    /// (no emitter for some visited instruction, or `walk` found it excluded
    /// outright) and arrival/queueing should stop re-requesting a compile
    /// that will only be declined again. Cleared on a gen bump alongside
    /// `valid_bits` — not implemented yet (no gen-triggered reclassification
    /// exists until invalidation lands, §7).
    #[inline]
    pub fn denylist(&self, offset_word: usize) {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.denylist_bits[word].fetch_or(1 << bit, std::sync::atomic::Ordering::Relaxed);
    }

    /// Un-publish `offset_word`'s entry — clears the valid bit only, NOT
    /// denylist: the artifact itself isn't wrong in general, just stale for
    /// *this* dispatch (`emit_fpu_entry_guard`'s FR-mismatch case — the
    /// region was compiled for the wrong FR mode, so every future dispatch
    /// through the normal gate would hit this exact same guard failure
    /// again, forever, without ever getting a chance at a fresh compile for
    /// the FR mode that's actually live). Unlike `denylist`, a later
    /// dispatch is expected and welcome to recompile this offset — most
    /// likely for the *other* FR mode, but `try_schedule`'s normal miss path
    /// handles that exactly like any other never-yet-compiled offset.
    /// `entries[offset_word].func` itself is deliberately left in place
    /// (same "may be stale until slot reuse" contract as `JitEntry::func`'s
    /// own doc comment) — nothing between clearing this bit and the next
    /// `publish` ever reads `func` without first re-checking `is_entry_valid`.
    #[inline]
    pub fn kill(&self, offset_word: usize) {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.valid_bits[word].fetch_and(!(1 << bit), std::sync::atomic::Ordering::Release);
    }

    /// Test-and-set `offset_word`'s scheduled bit: returns `true` (and sets
    /// the bit) only if it was previously clear, i.e. only the first caller
    /// for a given offset should actually build and send a `CompileRequest`
    /// — every other concurrent/subsequent caller sees `false` and skips it,
    /// since a request for this offset is already in flight. `fetch_or`
    /// alone (like `denylist`'s) isn't enough here: unlike sticky rejection,
    /// where every caller doing the same idempotent write is fine, this bit
    /// exists specifically to distinguish "I am the one who should send the
    /// request" from "someone else already did" — `fetch_or`'s return value
    /// (the *previous* bits) gives exactly that distinction for free, no
    /// separate compare-exchange needed.
    #[inline]
    pub fn try_schedule(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        let prev = self.scheduled_bits[word].fetch_or(1 << bit, std::sync::atomic::Ordering::Relaxed);
        prev & (1 << bit) == 0
    }

    /// Clear `offset_word`'s scheduled bit — called once `handle_request`
    /// (`comp.rs`) has decided the offset one way or the other (published or
    /// denylisted), so a future legitimate re-request for this offset (e.g.
    /// after a gen bump invalidates a stale artifact) isn't permanently
    /// blocked by a stale scheduled bit from a request that already finished.
    #[inline]
    pub fn clear_scheduled(&self, offset_word: usize) {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.scheduled_bits[word].fetch_and(!(1 << bit), std::sync::atomic::Ordering::Relaxed);
    }

    /// Publish a freshly compiled function at `offset_word` (§6.5 step 4):
    /// write `func` first, then Release-store `gen` — in that order, always.
    /// `gen_snap` is the page generation read *before* the compile started
    /// (§6.5 step 2); if the page has mutated since (current gen no longer
    /// matches), the publish is aborted before the bit is ever (re)set, so a
    /// racing writer that invalidated this compile during codegen can't have
    /// its mutation silently shadowed by a stale artifact. Returns `true` if
    /// the entry was actually published.
    ///
    /// The `func`-then-`gen` order (and `gen`'s Release/Acquire ordering,
    /// paired with `is_entry_valid`'s Acquire load) is load-bearing, not
    /// cosmetic — it's what makes *recompiling* an already-published entry
    /// safe, which `valid_bits` alone cannot do. §2.5's "no recompile of
    /// existing artifacts" design intent means `handle_request`
    /// (`comp.rs`) never calls this on an entry that's currently
    /// `is_entry_valid` — but an entry whose gen has drifted stale (page
    /// mutated, bit still 1) *does* get recompiled in place. On that path,
    /// `valid_bits`' Release/Acquire pairing provides no fresh
    /// synchronization at all: the bit doesn't change value (it was already
    /// 1), so a dispatcher's Acquire-load of it can be satisfied by a stale
    /// cached observation from the *original* publish, with no
    /// happens-before relationship to this recompile's writes whatsoever.
    /// Without `gen` itself carrying the ordering, a dispatcher could
    /// observe the *new* `gen` (matching `current_gen()`, so `is_entry_valid`
    /// reports true) paired with the *old* `func` pointer — silently calling
    /// a compiled function for a page state it was never actually compiled
    /// against. Making `gen` the synchronization point (write `func` before
    /// it, Release it, Acquire-load it before ever trusting `func`) closes
    /// that window using the field that already exists for exactly this
    /// "has this artifact been superseded" purpose, rather than adding a
    /// second one.
    /// `instr_count`: dev-only diagnostic (`JitEntry::instr_count`, `j2 pcp`)
    /// — the number of instructions the just-compiled region covers.
    /// `code_size`: dev-only diagnostic (`JitEntry::code_size`) — compiled
    /// machine code size in bytes. Both ignored (but still accepted, to keep
    /// this signature stable across feature combinations) when the
    /// `developer` feature is off.
    pub fn publish(&self, offset_word: usize, func: *const (), gen_snap: u64, #[allow(unused_variables)] instr_count: usize, #[allow(unused_variables)] code_size: u32) -> bool {
        if gen_snap != self.current_gen() {
            return false; // page already mutated past gen_snap; discard rather than publish a stale artifact
        }
        // Safety: `func` is a raw pointer write on a `JitEntry` shared behind
        // `&self` — sound because no concurrent reader trusts `func` without
        // first Acquire-loading `gen` below and observing it equal to
        // current_gen() (see this function's doc comment for why that's the
        // synchronization point, not `valid_bits`), and no other writer
        // targets the same offset concurrently (the compile thread is
        // single-threaded and processes requests in order).
        unsafe {
            let entries = self.entries.as_ptr() as *mut JitEntry;
            (*entries.add(offset_word)).func = func;
            #[cfg(feature = "developer")]
            {
                (*entries.add(offset_word)).instr_count = instr_count as u16;
                (*entries.add(offset_word)).code_size = code_size;
            }
        }
        self.entries[offset_word].gen.store(gen_snap, std::sync::atomic::Ordering::Release);
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        // fetch_or on an already-1 bit (the recompile-of-a-stale-entry case)
        // is a no-op on the bit's value but still executes as a Release op —
        // harmless to keep doing unconditionally here since it costs nothing
        // extra and keeps first-publish's existing Acquire/Release-on-bit
        // contract intact for `is_published`'s own callers.
        self.valid_bits[word].fetch_or(1 << bit, std::sync::atomic::Ordering::Release);
        true
    }

    /// Corpus-collection scaffolding (`saved_bits`, see its field doc):
    /// whether this offset's page snapshot has already been dumped to disk.
    #[inline]
    pub fn is_saved(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.saved_bits[word].load(std::sync::atomic::Ordering::Relaxed) & (1 << bit) != 0
    }

    /// Corpus-collection scaffolding: mark `offset_word` as saved. Returns
    /// `true` if this call is the one that set the bit (i.e. the caller
    /// should actually write the file) — using `fetch_or`'s previous value
    /// means concurrent duplicate work is impossible even if this is ever
    /// called from more than one thread for the same page.
    #[inline]
    pub fn mark_saved(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        let prev = self.saved_bits[word].fetch_or(1 << bit, std::sync::atomic::Ordering::Relaxed);
        prev & (1 << bit) == 0
    }
}

/// Slot index into [`Jitv2::pages`].
pub type PageSlot = u32;

/// JIT v2 engine state embedded in the mips executor.
///
/// Owns the [`PhysicalCodePage`] pool (§2.4): a single array, allocated once
/// at full `capacity` in `Jitv2::new` (every slot pre-built as unclaimed —
/// see `PhysicalCodePage::new`'s doc comment), never resized or reallocated
/// afterward. Pages are handed out on demand (first arrival at a given PFN)
/// by bump-claiming the next unclaimed slot (`PhysicalCodePage::claim`) — no
/// per-page allocation or move at claim time, matching §6.3's arena-allocator
/// model ("Bump-only allocation, reset only at `flush_all()`" — D6.2 lock-in
/// 1, generalized here to the page pool itself since PCPs, like wrapper
/// slots, only ever grow monotonically between flushes). This is stronger
/// than the old `Vec::push`-based growth: that also never reallocated once
/// capacity was reserved, but each `push` still moved a freshly-constructed
/// `PhysicalCodePage` value into place — with `entries` now inline (not
/// boxed), that move would copy the whole 1024-entry table. Claiming a
/// pre-existing slot in place avoids that entirely: nothing is ever copied
/// after `Jitv2::new` builds the array.
///
/// Lookup from `pfn` to pool slot goes through `pfn_to_slot`. This is a
/// HashMap for now — simplest thing that works. If page-switch lookup shows
/// up hot in profiling, the design doc's dense pfn-indexed alternative
/// (§2.4) is the fallback; not built preemptively.
pub struct Jitv2 {
    /// The full-capacity page pool, allocated once — see this struct's own
    /// doc comment. Indices are stable for the pool's entire lifetime,
    /// including across `mega_flush` (`mega_flush` resets slots in place,
    /// it never shrinks or reallocates this array), so pointers into it —
    /// e.g. the executor's current-PCP pointer, `CompileRequest::page` —
    /// stay valid for the process's whole lifetime, not just "until the next
    /// flush" as before. A `CompileRequest`/PCP pointer surviving a flush it
    /// raced against and landing back in a slot that's since been reclaimed
    /// for a different physical page is still a real hazard (unchanged from
    /// before — `worker_loop`'s `drain_pending` and the various
    /// gen-mismatch/`is_entry_valid` checks are what actually guard against
    /// stale content, not pointer validity) — array-lifetime stability alone
    /// doesn't imply content freshness.
    pages: Box<[PhysicalCodePage]>,
    /// Number of slots in `pages` that have been claimed (bump-allocated,
    /// `PhysicalCodePage::claim`) since construction or the last
    /// `mega_flush` — the pool's actual live extent; `pages[next_free..]`
    /// are still in the unclaimed state. Replaces the old `Vec::len()`
    /// (which used to double as this count, back when the pool grew via
    /// `push`).
    next_free: usize,
    /// pfn -> index into `pages`. Consulted only on a page switch (fetch
    /// lands on a different PFN than the currently-tracked one) — not on
    /// every fetch.
    pfn_to_slot: HashMap<Pfn, PageSlot>,
    /// Pool capacity, fixed at construction (== `pages.len()`). Claiming past
    /// this triggers `mega_flush` (the "ran out of PCPs" resource-exhaustion
    /// trigger).
    capacity: usize,
    /// Compile-request fifo + worker thread (§6.4/§9). Constructed alongside
    /// the pool; `start()`/`stop()` are separate calls so the executor
    /// controls the compile thread's lifetime independently (mirrors
    /// `MipsCpu`'s own start/stop, e.g. no compile thread while paused).
    pub compile_queue: CompileQueue,
    /// The one `Codegen` (and therefore its one Cranelift `JITModule`/memory
    /// arena) shared between `jitv2_inline_compile` and the async compile
    /// thread — the two modes are mutually exclusive at any given moment (a
    /// monitor command, `j2 inline on|off`, is the only way to switch), so
    /// there is no reason for each to carry its own separate arena that then
    /// each needs its own independent flush bookkeeping. `None` exactly when
    /// the compile-thread worker currently owns it (moved out by
    /// `CompileQueue::start`, handed back by `stop`); `Some` otherwise —
    /// whenever inline mode is what's actually running, or the queue simply
    /// isn't started. `flush` operates on whichever `Some` it finds; callers
    /// on the inline path (`exec_decoded`'s dispatch gate,
    /// `jitv2_track_pcp`'s pool-exhaustion handler) take it via
    /// `codegen.lock()` for the duration of one compile/reset, same
    /// exclusion discipline the compile thread already has by construction
    /// (only it ever touches its own moved-out copy).
    pub codegen: Mutex<Option<crate::jitv2::codegen::Codegen>>,
    /// Event counters, read only under `j2 status` (dev-only display — see
    /// `JitStats`'s own doc comment for why the *fields themselves* still
    /// exist and get threaded through unconditionally: it's cheaper to carry
    /// one always-present `Arc` clone than to duplicate every function on
    /// its path behind `#[cfg(feature = "developer")]`). `Arc`, not embedded
    /// by value, so `CompileQueue::start` can clone a handle into the worker
    /// thread once (same pattern as `function_count`/`cpu`/`jitv2` below)
    /// instead of every `handle_request` call locking `Jitv2` just to touch
    /// a counter.
    pub stats: Arc<JitStats>,
}

/// Per-instruction-count bucket of a `Jitv2::code_size_by_instr_count()`
/// scan: how many published entries landed at this instruction count, and
/// the raw (not host-page-rounded — that rounding only matters for the
/// arena-usage estimate, `code_bytes_used`; here we want the real
/// per-function code size Cranelift actually emitted) `code_size` bytes
/// across them, as count/sum/min/max — enough to report both an average and
/// a spread without keeping every individual size around.
#[cfg(feature = "developer")]
#[derive(Debug, Clone, Copy)]
pub struct CodeSizeBucket {
    pub count: u32,
    pub sum_bytes: u64,
    pub min_bytes: u32,
    pub max_bytes: u32,
}

impl Jitv2 {
    /// Allocate the full page pool up front, `capacity` slots, every one
    /// starting unclaimed (`PhysicalCodePage::new(0, null)`) — see `Jitv2`'s
    /// own doc comment for why this is a single one-shot allocation rather
    /// than lazy `push`-based growth. Sizing is a Phase 0 measurement per the
    /// design doc (§9); start with whatever the caller passes and let
    /// `mega_flush` absorb a too-small guess rather than trying to size this
    /// perfectly up front — getting it wrong now just means an earlier
    /// flush, not a correctness problem. Does not start the compile thread —
    /// call `compile_queue.start()`.
    pub fn new(capacity: usize) -> Self {
        Self {
            pages: (0..capacity).map(|_| PhysicalCodePage::new(0, std::ptr::null())).collect(),
            next_free: 0,
            pfn_to_slot: HashMap::new(),
            capacity,
            compile_queue: CompileQueue::new(),
            codegen: Mutex::new(Some(crate::jitv2::codegen::Codegen::new())),
            stats: Arc::new(JitStats::default()),
        }
    }

    /// Look up the pool slot for `pfn`, claiming the next unclaimed slot
    /// in place (`PhysicalCodePage::claim`, `gen_ptr(phys_addr)` on the bus)
    /// if this is the first arrival at this page. Returns `None` if the pool
    /// is exhausted — the caller (mips exec thread) is responsible for
    /// running `mega_flush` and retrying.
    ///
    /// `phys_addr` must be the physical address whose containing page is
    /// `pfn` (i.e. `pfn == phys_addr / PAGE_SIZE`) — passed separately
    /// rather than reconstructed here because callers already have it from
    /// translation and multiplying back out is wasted work on the hot path.
    pub fn page_for(&mut self, pfn: Pfn, phys_addr: u32, bus: &dyn BusDevice) -> Option<PageSlot> {
        if let Some(&slot) = self.pfn_to_slot.get(&pfn) {
            return Some(slot);
        }
        if self.next_free >= self.capacity {
            return None;
        }
        let gen = bus.gen_ptr(phys_addr);
        let slot = self.next_free as PageSlot;
        self.pages[slot as usize].claim(pfn, gen);
        self.next_free += 1;
        self.pfn_to_slot.insert(pfn, slot);
        Some(slot)
    }

    /// Raw pointer to the page at `slot`. Valid for the process's entire
    /// lifetime (see the `pages` field doc — the array itself never moves or
    /// resizes, including across `mega_flush`) — used to set the executor's
    /// current-PCP pointer without holding a borrow of `self`.
    #[inline]
    pub fn page_ptr(&mut self, slot: PageSlot) -> *mut PhysicalCodePage {
        &mut self.pages[slot as usize] as *mut PhysicalCodePage
    }

    /// Number of pool slots currently claimed (since construction or the
    /// last `mega_flush`). Exit-time diagnostic — see `MipsCpu::stop`.
    #[inline]
    pub fn pages_used(&self) -> usize {
        self.next_free
    }

    /// Pool capacity, as passed to `new()`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sum, across every published entry in every pooled page, of each
    /// entry's `JitEntry::code_size` **rounded up to
    /// `Codegen::HOST_PAGE_SIZE`** — dev-only diagnostic (`j2 stats`), the
    /// best available proxy for the shared `Codegen`'s actual Cranelift
    /// memory-arena usage (see `JitEntry::code_size`'s doc comment for why
    /// nothing more direct is available). Rounding matters: `code_size` is
    /// raw compiled-machine-code bytes (~215 bytes/function observed for a
    /// single-instruction region), but `ArenaMemoryProvider` gives every
    /// function its own segment, always rounded up to a full host page
    /// regardless of actual size (`CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s doc
    /// comment — confirmed live: the arena exhausted at exactly
    /// `ARENA_RESERVE_SIZE / HOST_PAGE_SIZE` functions, not the ~2.4M a
    /// byte-size-only estimate would predict) — summing raw `code_size`
    /// alone would under-report real arena consumption by roughly 19x at
    /// that average function size, which is exactly the gap that made the
    /// original OOM investigation confusing (small `code_bytes_used`
    /// numbers right up until the arena actually exhausted). O(pages ×
    /// ENTRIES_PER_PAGE); fine for an on-demand monitor command, not called
    /// from any hot path.
    #[cfg(feature = "developer")]
    pub fn code_bytes_used(&self) -> u64 {
        let page_size = crate::jitv2::codegen::Codegen::HOST_PAGE_SIZE;
        self.pages.iter()
            .map(|page| {
                (0..ENTRIES_PER_PAGE)
                    .filter(|&off| page.is_published(off))
                    .map(|off| {
                        let raw = page.entries[off].code_size as u64;
                        raw.div_ceil(page_size) * page_size
                    })
                    .sum::<u64>()
            })
            .sum()
    }

    /// Histogram of `JitEntry::instr_count` across every published entry in
    /// every pooled page, paired with each bucket's code-size distribution —
    /// dev-only diagnostic (`j2 status`), a full scan like `code_bytes_used`
    /// (O(pages × ENTRIES_PER_PAGE), fine for an on-demand monitor command).
    /// Indexed by instruction count directly (`result[n]` = stats for
    /// published entries whose region covers exactly `n` instructions);
    /// `n == 0` is always absent (a published entry's region always covers
    /// at least its own head instruction). Lets you see the real
    /// distribution against `MAX_INSTRS_PER_COMPILE` (`comp.rs`) — regions
    /// land at fewer instructions than the budget whenever a branch/excluded
    /// instruction/page boundary cuts the walk short, so this is ground
    /// truth, not just "the budget was N" — and, per bucket, whether code
    /// size scales roughly linearly with instruction count or has wide
    /// per-region variance (e.g. FPU regions paying the CU1/FR guard's fixed
    /// overhead regardless of instruction count).
    #[cfg(feature = "developer")]
    pub fn code_size_by_instr_count(&self) -> Vec<Option<CodeSizeBucket>> {
        let mut hist: Vec<Option<CodeSizeBucket>> = Vec::new();
        for page in self.pages.iter() {
            for off in 0..ENTRIES_PER_PAGE {
                if !page.is_published(off) { continue; }
                let entry = &page.entries[off];
                let n = entry.instr_count as usize;
                let size = entry.code_size;
                if n >= hist.len() { hist.resize(n + 1, None); }
                match &mut hist[n] {
                    Some(bucket) => {
                        bucket.count += 1;
                        bucket.sum_bytes += size as u64;
                        bucket.min_bytes = bucket.min_bytes.min(size);
                        bucket.max_bytes = bucket.max_bytes.max(size);
                    }
                    slot @ None => {
                        *slot = Some(CodeSizeBucket { count: 1, sum_bytes: size as u64, min_bytes: size, max_bytes: size });
                    }
                }
            }
        }
        hist
    }

    /// Reset the JIT to its initial state: drop every compiled artifact and
    /// every tracked page, and reset the pool allocator. The MAME-style
    /// "flush the world" response to running out of any bump-allocated JIT
    /// resource — page pool slots here; code arena and wrapper slots join
    /// this call as those pieces land (§6.3's `flush_all()`, of which this
    /// is the first caller: arena-full, `restore`, `rollback` all route
    /// through one routine).
    ///
    /// Does not yet demote promoted decode-entry handlers or null
    /// entry_table slots (§6.1.3, §6.3) — there are none to demote until
    /// the dispatcher/compiler land. Once they exist, this is where that
    /// walk goes, on the executor thread, before the reset loop below.
    ///
    /// Resets every claimed slot in place (`PhysicalCodePage::reset_to_unclaimed`)
    /// rather than the old `pages.clear()` — the array itself is never
    /// dropped/reallocated (see `Jitv2`'s own doc comment), so returning
    /// every slot to its unclaimed state, including zeroing every entry's
    /// `gen` (doubling as the pre-publish call counter — see
    /// `reset_entries_and_bitmaps`'s doc comment for why that specifically
    /// matters here, not just for `func`/`valid_bits`), is what "flush"
    /// means now.
    ///
    /// Private: real callers want [`Self::flush`], which wraps this with the
    /// compile-queue pause every caller of this used to have to remember
    /// (see that method's doc comment for why bundling it here, rather than
    /// leaving it the caller's responsibility, is what makes this whole
    /// operation self-contained now that `Jitv2` owns its own compile-queue
    /// lifecycle independently of `MipsCpu::stop()`/`start()`).
    fn mega_flush(&mut self) {
        for page in self.pages[..self.next_free].iter_mut() {
            page.reset_to_unclaimed();
        }
        self.next_free = 0;
        self.pfn_to_slot.clear();
    }

    /// Self-contained page-pool + compiled-code-arena flush, called FROM the
    /// CPU thread (`jitv2_track_pcp`'s pool-exhaustion handler). The caller
    /// is already "as good as stopped" — it's the one executing this,
    /// synchronously, not racing itself — so this never touches the CPU at
    /// all, only the *compile* queue (the other side): pauses it if it's
    /// running (every in-flight/queued `CompileRequest` points into
    /// `self.pages`, which `mega_flush` is about to clear — the worker must
    /// be fully joined and drained first, or it could dereference a page
    /// mid-drop out from under it; `stop()` also hands back whatever
    /// `Codegen` the worker was using), clears the pool, resets the
    /// `Codegen` (frees the Cranelift memory arena — `Codegen::function_count`'s
    /// doc comment for why nothing else ever does), and hands the reset
    /// `Codegen` back to wherever it came from — restarting the compile
    /// queue if (and only if) it was the one running, or back into
    /// `self.codegen` (idle slot) if inline dispatch owned it instead. Which
    /// of those it was is exactly what `compile_queue.stop()`'s `Option`
    /// tells us; blindly restarting the queue regardless used to silently
    /// steal the `Codegen` away from inline dispatch (see the code's own
    /// comment for the failure mode this caused).
    ///
    /// The caller (`jitv2_track_pcp`) is still responsible for its own
    /// `nanotlb_invalidate()`/`self.pcp = null` afterward — this type has no
    /// executor access to do that itself. See [`Self::flush_from_jit_thread`]
    /// for the mirror-image case (compile thread detects its own growth,
    /// must pause the *CPU* instead).
    ///
    /// # Safety
    /// Same contract as `Codegen::reset` — no `JitFn` this `Codegen` ever
    /// produced may still be reachable/callable after this returns.
    /// Guaranteed here: every `PhysicalCodePage` that could reference such a
    /// function is cleared by `mega_flush` in the same operation, and the
    /// compile queue is fully stopped (joined) before that clear runs.
    pub unsafe fn flush_from_cpu_thread(&mut self, bus: Arc<dyn BusDevice>) {
        // `compile_queue.stop()` returns `Some` only if the async worker was
        // actually running (i.e. threaded/`j2 inline off` mode) — that's
        // also the only case this should restart it afterward. When it
        // returns `None`, the codegen was already idle in `self.codegen`
        // (inline/`j2 inline on` mode, the default), and it must go back
        // there, NOT to the compile queue — unconditionally restarting the
        // queue here regardless of which mode was actually active used to
        // silently steal the codegen out from under inline dispatch: every
        // inline compile after the first pool-exhaustion flush would find
        // `self.codegen` empty and silently no-op (mips_exec.rs's `if let
        // Some(codegen) = codegen.as_mut()` guard swallows it with no
        // error), while `j2 inline` still reported "on" the whole time.
        let (was_threaded, mut codegen) = match self.compile_queue.stop() {
            Some(codegen) => (true, codegen),
            None => (false, self.codegen.get_mut().take().expect(
                "flush_from_cpu_thread: no Codegen available from either the stopped compile queue or the idle slot"
            )),
        };
        // `stop()` joins the worker (provably not popping anymore) and hands
        // back its Consumer as-is — anything still queued in it points into
        // the pool `mega_flush` is about to clear. Drain before clearing, not
        // after, same reasoning (and the same live-confirmed crash) as
        // `CompileQueue::worker_loop`'s own pre-flush drain.
        self.compile_queue.drain_pending_queue();
        eprintln!(
            "jitv2: mega_flush (from cpu thread) — {} / {} pages used, {} functions compiled",
            self.pages_used(), self.capacity(), codegen.function_count(),
        );
        self.mega_flush();
        unsafe { codegen.reset(); }
        if was_threaded {
            let stats = self.stats.clone();
            self.compile_queue.start(bus, codegen, stats);
        } else {
            *self.codegen.get_mut() = Some(codegen);
        }
    }

    /// Mirror image of [`Self::flush_from_cpu_thread`], called FROM the
    /// compile thread (`CompileQueue::worker_loop`, when its own `Codegen`
    /// growth crosses `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`). The compile thread
    /// can't pause itself (`CompileQueue::stop()` joins the very thread
    /// that would be calling it — a self-join deadlock), so this pauses the
    /// *CPU* instead: `cpu.stop()` fully joins the CPU's OS thread and
    /// establishes `pcp == null` as a stop-time invariant (`MipsCpu::stop`'s
    /// own doc comment) before returning, which is what makes it safe for
    /// this to clear the page pool directly despite §6.1.3's usual
    /// CPU-thread-only contract — the CPU is provably not running for the
    /// whole operation. `codegen` is the caller's own instance (already
    /// owned by the worker, never taken from `self.codegen` — unlike
    /// `flush_from_cpu_thread`, there's no ambiguity about which `Codegen`
    /// is "the" one here).
    ///
    /// # Safety
    /// Same contract as `Codegen::reset`. Guaranteed here by `cpu.stop()`
    /// having fully joined before `mega_flush` runs.
    ///
    /// Must be called with `self` NOT already locked by the caller:
    /// `cpu.stop()` locks the executor and, through it, this same
    /// `Mutex<Jitv2>` again (to print its own page-pool stats — see
    /// `MipsCpu::stop`'s doc comment) — calling this while already holding
    /// `Jitv2`'s lock (e.g. via `jit.lock().flush_from_jit_thread(...)`)
    /// self-deadlocks the compile thread on its own non-reentrant lock.
    /// Callers must take the lock only for the `mega_flush`/`reset` portion,
    /// not across `cpu.stop()`/`cpu.start()` — see `CompileQueue::worker_loop`
    /// for the correct call shape.
    pub unsafe fn flush_from_jit_thread(&mut self, cpu: &dyn Device, codegen: &mut crate::jitv2::codegen::Codegen) {
        eprintln!(
            "jitv2: mega_flush (from jit thread) — {} / {} pages used, {} functions compiled",
            self.pages_used(), self.capacity(), codegen.function_count(),
        );
        self.mega_flush();
        unsafe { codegen.reset(); }
    }
}

/// Depth of the compile-request SPSC ring (§6.4 "bounded queue; drop on full —
/// hot pages re-trigger"). A starting guess, like `JITV2_INITIAL_PAGE_CAPACITY`
/// — doubled from 1024 after a live `j2 status` reading showed the compile
/// thread genuinely falling behind at that size (20.9% of dispatches
/// dropped for a full queue, average depth at dispatch 248.6/1024, out of
/// 1,166,218 total dispatches during one session) rather than the queue
/// mostly sitting near-empty.
pub const COMPILE_QUEUE_CAPACITY: usize = 2048;

/// The compile thread and its inbound SPSC fifo (§6.4/§9 Phase 1: "mips exec
/// thread pushes work via spsc fifo, jit thread compiles").
///
/// Owns the `rtrb::Producer` end; the worker thread owns the `Consumer` end
/// for its lifetime. Restartable: `stop()` gets the consumer back from the
/// exiting worker so a later `start()` can resume with the same queue — this
/// is what lets `mega_flush` stop-drain-restart the thread around a page-pool
/// reset instead of the queue being single-use (see `mega_flush`'s call site
/// in `mips_exec.rs`, `jitv2_track_pcp`, for why that's required: every
/// `CompileRequest::page` in flight or still queued points into the `Vec`
/// `mega_flush` is about to clear, so the worker must be fully joined, and
/// its queue drained, before that clear happens).
pub struct CompileQueue {
    producer: rtrb::Producer<CompileRequest>,
    /// Consumer end, held here whenever the worker isn't running (between
    /// `new()`/`stop()` and the next `start()`). `start()` takes it to move
    /// into the worker thread; `None` while the thread is running.
    consumer: Option<rtrb::Consumer<CompileRequest>>,
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<(rtrb::Consumer<CompileRequest>, crate::jitv2::codegen::Codegen)>>,
    /// Weak handle to the CPU device, so the worker can stop/start it itself
    /// when a memory-growth flush is needed (`Codegen::function_count()`
    /// crossing `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES` — see `worker_loop`). Set
    /// once, via `set_cpu`, right after `Arc<MipsCpu<T,C>>` is constructed in
    /// `Machine::new` (mirrors `Mc::set_cpu`'s own `Arc::downgrade`
    /// injection — the CPU doesn't exist yet when `Jitv2`/`CompileQueue` are
    /// first built, so this can't be a constructor parameter). `Weak`, not
    /// `Arc`: a strong reference here would be a real cycle (MipsCpu owns the
    /// executor, which owns `Jitv2`, which would own this) that nothing
    /// would ever break — `Weak::upgrade` failing just means the machine is
    /// mid-teardown and there's nothing to stop/start anymore, which is fine
    /// to skip.
    cpu: Mutex<Option<Weak<dyn Device>>>,
    /// Weak handle back to the owning `Jitv2` (behind whatever `Arc<Mutex<..>>`
    /// the caller shares it in) — the worker needs this to clear the page
    /// pool during its own flush (`worker_loop`'s doc comment explains why
    /// that's safe despite §6.1.3's usual CPU-thread-only contract: the CPU
    /// is provably stopped for the whole operation). Set via `set_owner`,
    /// same injection-after-construction reasoning as `cpu`.
    jitv2: Mutex<Option<Weak<Mutex<Jitv2>>>>,
    /// Mirror of the worker's own `codegen.function_count()`, updated after
    /// every compile — exists purely so `j2 stats` can read a function count
    /// while the worker owns the real `Codegen` by value (not behind any
    /// lock a status command could take without contending the hot compile
    /// path). Shared (`Arc`, like `running`) because `worker_loop` runs on
    /// its own thread and needs to write it without a `&CompileQueue`. Reset
    /// to 0 alongside every real flush (`worker_loop`'s own
    /// `flush_from_jit_thread` call) so it never drifts from what
    /// `function_count()` would report if you could ask the real `Codegen`
    /// directly.
    function_count: Arc<AtomicU32>,
    /// Toggle for `worker_loop`'s deferred-finalize batching (`j2 batch
    /// on|off`) — `Arc<AtomicBool>` for the same reason `running` is: the
    /// worker thread needs to read it every loop iteration without a lock,
    /// and the CPU/console thread needs to flip it from outside — see
    /// `worker_loop`'s own doc comment for what flips on when this is `true`.
    /// Defaults to `false` under `developer` (diagnostics builds want the
    /// simpler per-compile-immediate-finalize behavior, not batched
    /// publishes complicating a step-by-step investigation) and to `true`
    /// otherwise (production runs want the tighter arena packing batching
    /// gives).
    batch_enabled: Arc<AtomicBool>,
}

impl CompileQueue {
    /// Construct the queue without starting the worker thread. Call
    /// [`Self::start`] to spawn it.
    pub fn new() -> Self {
        let (producer, consumer) = rtrb::RingBuffer::new(COMPILE_QUEUE_CAPACITY);
        Self {
            producer,
            consumer: Some(consumer),
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
            cpu: Mutex::new(None),
            jitv2: Mutex::new(None),
            function_count: Arc::new(AtomicU32::new(0)),
            batch_enabled: Arc::new(AtomicBool::new(!cfg!(feature = "developer"))),
        }
    }

    /// Enable or disable deferred-finalize batching on the worker thread —
    /// see `worker_loop`'s doc comment. Safe to call at any time; takes
    /// effect on the worker's next loop iteration (or immediately for a
    /// not-yet-started queue). Returns the previous value.
    pub fn set_batch_enabled(&self, enabled: bool) -> bool {
        self.batch_enabled.swap(enabled, Ordering::Relaxed)
    }

    #[inline]
    pub fn batch_enabled(&self) -> bool {
        self.batch_enabled.load(Ordering::Relaxed)
    }

    /// Mirror of the worker's own `Codegen::function_count()` — see
    /// `function_count`'s field doc comment for why this exists instead of
    /// reading the real `Codegen` directly. Meaningful only while the
    /// worker is actually running (`j2 stats`/`j2 status` callers should
    /// prefer `Jitv2::codegen.lock()` when it's `Some`, and fall back to
    /// this only when it's `None`, i.e. the worker currently owns it).
    #[inline]
    pub fn function_count(&self) -> u32 {
        self.function_count.load(Ordering::Relaxed)
    }

    /// Current occupancy of the compile-request ring buffer — `producer`
    /// stays on `CompileQueue` at all times (only `consumer` moves into the
    /// worker thread on `start()`), so this is readable regardless of
    /// whether the worker is running. `j2 status`'s live "how full is it
    /// right now" reading, distinct from `JitStats::compile_queue_depth_sum`'s
    /// historical average.
    #[inline]
    pub fn queue_occupancy(&self) -> usize {
        COMPILE_QUEUE_CAPACITY - self.producer.slots()
    }

    /// Inject the CPU device handle — see the `cpu` field's doc comment for
    /// why this is a separate setter rather than a constructor parameter.
    /// Safe to call at any time (even while the worker is running); takes
    /// effect on that worker's next flush-threshold check, or immediately
    /// for a not-yet-started queue.
    pub fn set_cpu(&self, cpu: Weak<dyn Device>) {
        *self.cpu.lock() = Some(cpu);
    }

    /// Inject the weak handle back to the owning `Jitv2` — see the `jitv2`
    /// field's doc comment. Same call-after-construction reasoning as
    /// `set_cpu` (the `Arc<Mutex<Jitv2>>` doesn't exist until after this
    /// `CompileQueue`, which lives inside it, has already been constructed).
    pub fn set_owner(&self, jitv2: Weak<Mutex<Jitv2>>) {
        *self.jitv2.lock() = Some(jitv2);
    }

    /// Push a compile request. Non-blocking: per §6.4, a full queue drops the
    /// request rather than backing up the exec thread — the page that wanted
    /// it stays hot and will re-trigger the request on a later arrival.
    /// Returns `false` if the request was dropped. `stats` is `Jitv2::stats`
    /// — under `developer`, records this dispatch, whether it was dropped
    /// for a full queue, and the queue's occupancy at this exact moment
    /// (`JitStats`'s own doc comments on the three `compile_queue_*` fields)
    /// for `j2 status`'s FIFO-fullness section. Accepted unconditionally
    /// (not `#[cfg]`-gated itself) so this signature doesn't change across
    /// feature combinations; the instrumentation work inside is what's
    /// gated, to keep the extra `slots()`/atomic touches off this
    /// per-dispatch-gate-miss hot path outside a diagnostics build.
    pub fn send(&mut self, req: CompileRequest, #[allow(unused_variables)] stats: &JitStats) -> bool {
        #[cfg(feature = "developer")]
        {
            // slots() before push(): occupancy right now, not after this
            // one lands — matches "depth the compile thread is running at"
            // rather than counting this dispatch's own contribution to it.
            let occupancy = COMPILE_QUEUE_CAPACITY - self.producer.slots();
            stats.compile_queue_dispatches.fetch_add(1, Ordering::Relaxed);
            stats.compile_queue_depth_sum.fetch_add(occupancy as u64, Ordering::Relaxed);
        }
        let accepted = self.producer.push(req).is_ok();
        #[cfg(feature = "developer")]
        if !accepted {
            stats.compile_queue_full.fetch_add(1, Ordering::Relaxed);
        }
        accepted
    }

    /// Discard every `CompileRequest` currently sitting in `consumer`
    /// without processing it. Every `CompileRequest::page` is a raw pointer
    /// into `Jitv2::pages` (`Vec<PhysicalCodePage>`) — a request enqueued
    /// before a `mega_flush` still holds a pointer into the just-cleared
    /// `Vec` after it, and `comp::handle_request` dereferencing it is a real
    /// use-after-free (confirmed live: `PhysicalCodePage::publish` segfault,
    /// `jitv2-compile` thread, immediately after a `flush_from_jit_thread`
    /// that never drained the queue first). Must run with nothing else
    /// popping `consumer` concurrently — true both when called from the
    /// worker thread itself right after `cpu.stop()` (nothing else touches
    /// this `Consumer` while the CPU can't enqueue new requests and this
    /// thread isn't popping anything else), and when called by
    /// `flush_from_cpu_thread`'s caller right after `stop()` hands the
    /// `Consumer` back (the worker thread is joined, provably not popping).
    fn drain_pending(consumer: &mut rtrb::Consumer<CompileRequest>) {
        while consumer.pop().is_ok() {}
    }

    /// Public entry point for [`Self::drain_pending`] when the caller only
    /// has `&mut CompileQueue`, not the raw `Consumer` (i.e. after `stop()`
    /// has stashed it back into `self.consumer`) — see `drain_pending`'s own
    /// doc comment for why this must run. No-op if the worker is currently
    /// running (`self.consumer` is `None` while it owns the `Consumer`) —
    /// callers in that state should be draining via the worker thread's own
    /// call to `drain_pending` instead (`worker_loop`'s pre-flush drain).
    pub fn drain_pending_queue(&mut self) {
        if let Some(consumer) = self.consumer.as_mut() {
            Self::drain_pending(consumer);
        }
    }

    /// Spawn the compile-thread worker. No-op if already running (`codegen`
    /// is dropped in that case — same "start is idempotent" contract as
    /// before). `bus` is the executor's `sysad` — the worker reads the page
    /// snapshot off it at compile time (§6.5 step 2). `codegen` is the
    /// shared `Codegen` this queue and `jitv2_inline_compile` take turns
    /// owning (`Jitv2::codegen` — see that field's doc comment); the worker
    /// owns it for its entire run and hands it back via `stop()`. `stats` is
    /// `Jitv2::stats`, cloned in by the caller (mirrors `cpu`/`jitv2` below —
    /// see `JitStats`'s own doc comment for why this is an `Arc` handed in
    /// rather than reached via a lock on every `handle_request` call).
    /// Threaded through unconditionally (cheap: one more `Arc` clone) rather
    /// than duplicating this whole function behind `#[cfg(feature =
    /// "developer")]` — the actual counter increments are what's gated, in
    /// `comp::handle_request`.
    pub fn start(&mut self, bus: Arc<dyn BusDevice>, codegen: crate::jitv2::codegen::Codegen, stats: Arc<JitStats>) {
        if self.thread.is_some() {
            return;
        }
        let consumer = match self.consumer.take() {
            Some(c) => c,
            None => return, // start() called concurrently or before the prior stop() finished
        };
        self.running.store(true, Ordering::SeqCst);
        self.function_count.store(codegen.function_count(), Ordering::Relaxed);
        let running = self.running.clone();
        let cpu = self.cpu.lock().clone();
        let jitv2 = self.jitv2.lock().clone();
        let function_count = self.function_count.clone();
        let batch_enabled = self.batch_enabled.clone();
        self.thread = Some(
            std::thread::Builder::new()
                .name("jitv2-compile".to_string())
                .spawn(move || Self::worker_loop(consumer, running, bus, codegen, cpu, jitv2, function_count, stats, batch_enabled))
                .expect("jitv2-compile spawn"),
        );
    }

    /// Stop the worker thread and join it, reclaiming the consumer and the
    /// shared `Codegen` so a later `start()` can resume with both. No-op (and
    /// returns `None`) if not running. The `Codegen` returned here is
    /// whatever the worker was using at the moment it exited — untouched by
    /// `stop()` itself, so the caller (`j2 inline off`→`on`, or
    /// `MipsExecutor` reclaiming it for inline dispatch) must put it back
    /// into `Jitv2::codegen` before anything tries to compile through the
    /// inline path.
    pub fn stop(&mut self) -> Option<crate::jitv2::codegen::Codegen> {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            if let Ok((consumer, codegen)) = handle.join() {
                self.consumer = Some(consumer);
                return Some(codegen);
            }
        }
        None
    }

    /// Worker body: pop requests until stopped and hand each to
    /// `comp::handle_request` (reachability walk + codegen + publish, §6.5;
    /// dump-to-disk corpus collection lives behind the `jitv2_corpus_dump`
    /// feature in the same function, see `jitv2/comp.rs`). Owns the
    /// `Analyzer` scratch state for the thread's whole lifetime (meant to be
    /// reused across jobs, not rebuilt per request); `codegen` is the shared
    /// one moved in by `start`. Backs off briefly when the queue is empty
    /// rather than busy-spinning — compile requests are bursty (arrival
    /// threshold crossings), not latency-critical.
    ///
    /// When `batch_enabled` is set (`j2 batch on`), compiles route through
    /// `comp::handle_request_deferred` instead of the immediate
    /// `comp::handle_request`: each successful compile accumulates a
    /// `PendingPublish` rather than finalizing+publishing on the spot, and
    /// the whole pending batch is finalized together
    /// (`comp::flush_pending_batch`) whenever `codegen.provider_crossed_page()`
    /// reports the *next* allocation would spill onto a new host-page
    /// segment, or the queue drains empty (whichever comes first) — see
    /// `paged_memory`'s module doc comment for why batching the
    /// `finalize_definitions()` call is what actually lets functions pack
    /// tightly instead of each getting its own page. `pending` is discarded
    /// (never flushed) immediately before `do_flush` runs: `do_flush` resets
    /// `codegen` (freeing the whole arena — every not-yet-finalized `FuncId`
    /// in `pending` would dangle) and clears the page pool (every
    /// `PendingPublish::page` would dangle too), so there's nothing safe left
    /// to publish by that point — same reasoning as `drain_pending` already
    /// applies to in-flight `CompileRequest`s for the identical reset.
    ///
    /// After every successful compile, checks `codegen.function_count()`
    /// against the flush threshold (`Codegen::function_count`'s doc comment)
    /// — if crossed, stops the CPU (via `cpu`, upgraded from `Weak`; skipped
    /// entirely if unset or already gone — nothing to flush for if the
    /// machine has no CPU to pause), flushes in place (page pool +
    /// `codegen.reset()` — NOT `Jitv2::flush`, which would try to stop this
    /// same compile queue from within itself; this worker just does the
    /// pool clear + reset directly, since it's already the thing that would
    /// need pausing), and restarts the CPU. Returns `(consumer, codegen)` on
    /// exit so `stop()` can hand both back to a later `start()`.
    fn worker_loop(
        mut consumer: rtrb::Consumer<CompileRequest>,
        running: Arc<AtomicBool>,
        bus: Arc<dyn BusDevice>,
        mut codegen: crate::jitv2::codegen::Codegen,
        cpu: Option<Weak<dyn Device>>,
        jitv2: Option<Weak<Mutex<Jitv2>>>,
        function_count: Arc<AtomicU32>,
        stats: Arc<JitStats>,
        batch_enabled: Arc<AtomicBool>,
    ) -> (rtrb::Consumer<CompileRequest>, crate::jitv2::codegen::Codegen) {
        let mut analyzer = crate::jitv2::analyzer::Analyzer::new();
        let mut pending: Vec<crate::jitv2::comp::PendingPublish> = Vec::new();
        // Pauses the CPU and flushes in place — shared by both flush
        // triggers below (function-count threshold, and an out-of-memory
        // compile failure). See the two call sites for why each one needs
        // this; the sequencing itself (cpu.stop() outside the Jitv2 lock,
        // drain the queue before mega_flush clears the pool, cpu.start()
        // after releasing the lock) is identical either way.
        let do_flush = |consumer: &mut rtrb::Consumer<CompileRequest>,
                         codegen: &mut crate::jitv2::codegen::Codegen,
                         function_count: &Arc<AtomicU32>,
                         pending: &mut Vec<crate::jitv2::comp::PendingPublish>| {
            if let (Some(cpu), Some(jit)) = (
                cpu.as_ref().and_then(Weak::upgrade),
                jitv2.as_ref().and_then(Weak::upgrade),
            ) {
                // Discard, not flush: every PendingPublish::page and every
                // not-yet-finalized FuncId this batch is holding is about to
                // dangle (flush_from_jit_thread clears the page pool and
                // resets codegen's whole arena) — see worker_loop's own doc
                // comment for the full reasoning, same as drain_pending's
                // existing treatment of in-flight CompileRequests below.
                pending.clear();
                // cpu.stop() must run with Jitv2's lock NOT held — it locks
                // the executor and, through it, this same Mutex<Jitv2>
                // again (own page-pool stats print), which would
                // self-deadlock this thread on its own non-reentrant lock
                // if taken here first. Lock only for the actual flush/reset
                // (flush_from_jit_thread's own doc comment), release
                // before cpu.start().
                cpu.stop();
                // Every request still sitting in `consumer` right now
                // points into the pool `flush_from_jit_thread` is about to
                // clear — drain them before the flush, not after, or the
                // next pop() dereferences a dangling PhysicalCodePage
                // pointer (see drain_pending's own doc comment for the
                // crash this was confirmed to cause).
                Self::drain_pending(consumer);
                unsafe { jit.lock().flush_from_jit_thread(cpu.as_ref(), codegen); }
                cpu.start();
                // flush_from_jit_thread reset codegen back to 0.
                function_count.store(0, Ordering::Relaxed);
            }
        };
        while running.load(Ordering::Relaxed) {
            match consumer.pop() {
                Ok(req) => {
                    let batching = batch_enabled.load(Ordering::Relaxed);
                    let ran_out_of_memory = if batching {
                        #[cfg(feature = "developer")]
                        { crate::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending, &stats) }
                        #[cfg(not(feature = "developer"))]
                        { crate::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending) }
                    } else {
                        #[cfg(feature = "developer")]
                        { crate::jitv2::comp::handle_request(&req, &bus, &mut analyzer, &mut codegen, &stats) }
                        #[cfg(not(feature = "developer"))]
                        { crate::jitv2::comp::handle_request(&req, &bus, &mut analyzer, &mut codegen) }
                    };
                    // Keep CompileQueue::function_count's mirror in sync —
                    // see that field's doc comment for why it exists
                    // (`j2 stats` can't read the real codegen.function_count()
                    // directly while this thread owns `codegen` by value).
                    function_count.store(codegen.function_count(), Ordering::Relaxed);
                    if ran_out_of_memory {
                        // The compile that just ran couldn't get memory —
                        // flush immediately, regardless of the byte
                        // threshold below (the arena is provably already
                        // full, so there's nothing to gain from checking).
                        // The request that just failed is gone
                        // (handle_request already returned), so it'll only
                        // be retried on this offset's next real arrival —
                        // same as any other "retry later" outcome.
                        do_flush(&mut consumer, &mut codegen, &function_count, &mut pending);
                    } else if codegen.packing_stats().1 > CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES {
                        // Real bytes reserved in the arena (not a
                        // function-count proxy — see
                        // CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES's own doc
                        // comment) crossing the threshold means this
                        // thread's own Cranelift memory arena has grown
                        // unboundedly (nothing else ever frees it — see
                        // Codegen::reset's doc comment) and needs flushing
                        // pre-emptively, before it actually runs out.
                        do_flush(&mut consumer, &mut codegen, &function_count, &mut pending);
                    } else if batching && codegen.provider_crossed_page() {
                        // The compile that just ran started a fresh
                        // host-page segment — everything accumulated in
                        // `pending` before it packed into the previous
                        // segment as tightly as it's going to; finalize and
                        // publish that batch now rather than let it keep
                        // growing indefinitely (an unbounded `pending` would
                        // mean an unbounded window of "compiled but not yet
                        // dispatchable").
                        #[cfg(feature = "developer")]
                        stats.record_batch_flush(pending.len(), crate::jitv2::BatchFlushReason::PageCross);
                        crate::jitv2::comp::flush_pending_batch(&mut codegen, &mut pending);
                    }
                }
                Err(_) => {
                    // Queue-drain fallback: even under batching, a lone
                    // compile followed by a quiet period must not sit
                    // unpublished indefinitely waiting for a page to fill —
                    // flush whatever's pending before backing off. A no-op
                    // (via flush_pending_batch's own empty check) on every
                    // iteration where nothing is pending, which is the
                    // common case when batching is off.
                    if !pending.is_empty() {
                        #[cfg(feature = "developer")]
                        stats.record_batch_flush(pending.len(), crate::jitv2::BatchFlushReason::QueueDrain);
                        crate::jitv2::comp::flush_pending_batch(&mut codegen, &mut pending);
                        function_count.store(codegen.function_count(), Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
        (consumer, codegen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64};

    #[test]
    fn null_gen_ptr_is_not_compilable() {
        let page = PhysicalCodePage::new(0, std::ptr::null());
        assert!(!page.is_compilable());
    }

    #[test]
    fn physical_code_page_size_is_within_expected_bounds() {
        // Guardrail, not a strict contract: entries is now inline
        // (JITV2_INITIAL_PAGE_CAPACITY's own doc comment — no more Box
        // hiding the real per-page cost), so an accidental field bloat here
        // directly multiplies by the whole pool capacity at startup. Fails
        // loudly if PhysicalCodePage ever grows past a sanity ceiling rather
        // than silently ballooning Jitv2::new's one-shot allocation.
        let page_size = std::mem::size_of::<PhysicalCodePage>();
        let entry_size = std::mem::size_of::<JitEntry>();
        println!("size_of::<PhysicalCodePage>() = {page_size} bytes ({} bytes/entry x {ENTRIES_PER_PAGE} entries + bitmaps)", entry_size);
        assert!(page_size < 128 * 1024, "PhysicalCodePage grew unexpectedly large: {page_size} bytes — check for accidental field bloat before raising this ceiling");
    }

    #[test]
    fn reads_through_gen_pointer() {
        let counter = AtomicU64::new(42);
        let page = PhysicalCodePage::new(7, &counter as *const AtomicU64);
        assert!(page.is_compilable());
        assert_eq!(page.current_gen(), 42);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(page.current_gen(), 43);
    }

    #[test]
    fn entry_starts_unpublished_and_undenylisted() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_entry_valid(4));
        assert!(!page.is_denylisted(4));
        assert!(page.entries[4].func.is_null());
    }

    /// `try_schedule` must be a genuine test-and-set: the first caller for a
    /// given offset wins (returns `true`); every subsequent caller for the
    /// same offset, before `clear_scheduled` runs, must lose (returns
    /// `false`) — this is what stops `exec_decoded`'s dispatch gate from
    /// sending a duplicate `CompileRequest` for the same offset every time a
    /// hot PC re-satisfies the gate's trigger conditions while the first
    /// request is still in flight (see `PhysicalCodePage::scheduled_bits`'
    /// doc comment).
    #[test]
    fn try_schedule_is_test_and_set() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule(4), "first caller for a fresh offset must win");
        assert!(!page.try_schedule(4), "second caller before clear_scheduled must lose");
        assert!(!page.try_schedule(4), "still losing on a third call");

        // A different offset is independent.
        assert!(page.try_schedule(5), "a different offset must not be blocked by offset 4's bit");
    }

    #[test]
    fn clear_scheduled_allows_a_fresh_try_schedule() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule(4));
        assert!(!page.try_schedule(4));

        page.clear_scheduled(4);
        assert!(page.try_schedule(4), "after clear_scheduled, a fresh request for the same offset must be allowed again");
    }

    #[test]
    fn clear_scheduled_on_an_unset_offset_is_a_harmless_no_op() {
        // The jitv2_inline_compile path calls handle_request (and therefore
        // clear_scheduled, via its scope guard) without ever having called
        // try_schedule first — must not panic or affect other offsets.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.clear_scheduled(4);
        assert!(page.try_schedule(4), "offset must still be schedulable after a no-op clear");
    }

    #[test]
    fn entry_valid_only_when_bit_set_and_gen_matches() {
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        // gen matches but bit not set: still invalid.
        page.entries[offset].gen.store(5, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset));

        // Publish: set the bit -> now valid.
        page.valid_bits[offset >> 6].fetch_or(1 << (offset & 63), Ordering::Release);
        assert!(page.is_entry_valid(offset));

        // Page mutates (gen bumps past what the entry was compiled against):
        // bit is still set, but the entry must read as stale.
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset), "stale entry (gen mismatch) must not be reported valid");
    }

    #[test]
    fn kill_clears_valid_bit_but_not_denylist() {
        // emit_fpu_entry_guard's FR-mismatch arm (jit_kill_entry) uses this
        // to un-publish a compiled-for-the-wrong-FR-mode entry: the JIT gate
        // must stop dispatching it, but a later visit is expected and
        // welcome to recompile the same offset fresh — unlike denylist,
        // which is permanent (§6.4 sticky rejection).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 4usize;
        page.entries[offset].gen.store(0, Ordering::Relaxed);
        page.valid_bits[offset >> 6].fetch_or(1 << (offset & 63), Ordering::Release);
        assert!(page.is_entry_valid(offset));

        page.kill(offset);

        assert!(!page.is_published(offset), "kill must clear the valid bit");
        assert!(!page.is_entry_valid(offset));
        assert!(!page.is_denylisted(offset), "kill must not sticky-reject the offset — a fresh compile is expected to follow");

        // A later re-publish (simulating the next visit's fresh compile)
        // must work normally — kill leaves the offset fully recompilable.
        page.entries[offset].gen.store(0, Ordering::Relaxed);
        page.valid_bits[offset >> 6].fetch_or(1 << (offset & 63), Ordering::Release);
        assert!(page.is_entry_valid(offset), "offset must be re-publishable after kill");
    }

    #[test]
    fn publish_recompiling_a_stale_entry_updates_func_and_gen_together() {
        // Regression test for the recompile-ordering race: an entry whose
        // gen has drifted stale (page mutated, valid_bits bit still 1) gets
        // recompiled in place by handle_request (comp.rs) — the ONE case
        // where publish() is called on an offset whose bit is already set.
        // valid_bits' Release/Acquire pairing gives no fresh synchronization
        // on that path (the bit's value doesn't change), so `gen` itself
        // must be the ordering point: func must be visible before gen ever
        // reads as matching current_gen(). This test can't force a true
        // concurrent interleaving, but it does verify the sequential
        // contract publish() must uphold for that ordering argument to hold
        // at all: func actually gets updated, and is_entry_valid only
        // reports true once gen matches (i.e. after publish() completes).
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        let old_fn = 0x1000usize as *const ();
        assert!(page.publish(offset, old_fn, 5, 1, 0));
        assert!(page.is_entry_valid(offset));
        assert_eq!(page.entries[offset].func, old_fn);

        // Page mutates: entry goes stale (bit stays 1, gen no longer matches).
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset), "must read stale immediately after the page mutates");

        // Recompile in place (comp.rs's handle_request path for a
        // stale-but-still-published entry) — gen_snap=6 was captured before
        // this second compile started, matching the page's now-current gen.
        let new_fn = 0x2000usize as *const ();
        assert!(page.publish(offset, new_fn, 6, 1, 0));
        assert!(page.is_entry_valid(offset), "recompiled entry must read valid once publish completes");
        assert_eq!(page.entries[offset].func, new_fn, "func must be the NEW function, not the stale one, once gen reads as current");
    }

    #[test]
    fn denylist_bit_is_independent_of_valid_bit() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        page.denylist_bits[offset >> 6].fetch_or(1 << (offset & 63), Ordering::Relaxed);
        assert!(page.is_denylisted(offset));
        assert!(!page.is_entry_valid(offset), "denylisting must not itself mark an entry valid");
    }

    /// Minimal BusDevice whose gen_ptr always returns the same fixed counter,
    /// standing in for a real RAM/ROM device in these pool-only tests.
    struct FakeDevice(AtomicU64);
    impl BusDevice for FakeDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
    }

    #[test]
    fn page_for_allocates_once_and_caches_by_pfn() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        let slot_a = jit.page_for(3, 3 * PAGE_SIZE, &dev).unwrap();
        let slot_b = jit.page_for(3, 3 * PAGE_SIZE, &dev).unwrap();
        assert_eq!(slot_a, slot_b, "second arrival at the same pfn must reuse the slot");
        let slot_c = jit.page_for(4, 4 * PAGE_SIZE, &dev).unwrap();
        assert_ne!(slot_a, slot_c);
    }

    #[test]
    fn page_for_returns_none_when_pool_exhausted() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        assert!(jit.page_for(0, 0, &dev).is_some());
        assert!(jit.page_for(1, PAGE_SIZE, &dev).is_none(), "pool of 1 must reject a second distinct pfn");
    }

    #[test]
    fn mega_flush_resets_pool_and_lookup() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let first = jit.page_for(0, 0, &dev).unwrap();
        jit.mega_flush();
        let second = jit.page_for(0, 0, &dev).unwrap();
        assert_eq!(first, second, "slots renumber from 0 after a flush");
        assert!(jit.page_for(1, PAGE_SIZE, &dev).is_none(), "pool capacity still enforced after flush");
    }

    #[test]
    fn mega_flush_clears_per_entry_gen_so_a_reused_slot_starts_with_a_fresh_call_counter() {
        // Regression test for the pre-publish call-counter staleness bug a
        // reused (post-flush) slot could otherwise have: entries[i].gen
        // doubles as PhysicalCodePage::count_dispatch_and_check_threshold's
        // counter before an entry is ever published — mega_flush's in-place
        // reset (PhysicalCodePage::reset_to_unclaimed) must zero it just
        // like func/valid_bits, or a slot claimed for a brand-new physical
        // page after a flush would inherit whatever count was left over from
        // its previous occupant.
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let slot = jit.page_for(0, 0, &dev).unwrap() as usize;

        // Drive offset 4's pre-publish counter up close to a real threshold
        // without actually publishing it.
        for _ in 0..3 {
            assert!(!jit.pages[slot].count_dispatch_and_check_threshold(4, 10));
        }
        assert_eq!(jit.pages[slot].entries[4].gen.load(Ordering::Relaxed), 3);

        jit.mega_flush();
        let new_slot = jit.page_for(1, PAGE_SIZE, &dev).unwrap() as usize;
        assert_eq!(slot, new_slot, "single-capacity pool reuses the same physical slot");
        assert_eq!(jit.pages[new_slot].entries[4].gen.load(Ordering::Relaxed), 0,
            "a reused slot's pre-publish call counter must start fresh, not inherit the previous occupant's count");
    }

    #[test]
    #[cfg(feature = "developer")]
    fn code_size_by_instr_count_buckets_by_instr_count_and_tracks_min_max_sum() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let slot = jit.page_for(0, 0, &dev).unwrap() as usize;
        let page = &jit.pages[slot];

        // Two entries at instr_count=3 (sizes 100, 300), one at instr_count=5 (size 50).
        assert!(page.publish(0, 0x1000 as *const (), 0, 3, 100));
        assert!(page.publish(1, 0x2000 as *const (), 0, 3, 300));
        assert!(page.publish(2, 0x3000 as *const (), 0, 5, 50));

        let hist = jit.code_size_by_instr_count();
        let bucket3 = hist[3].expect("instr_count=3 bucket must be present");
        assert_eq!(bucket3.count, 2);
        assert_eq!(bucket3.sum_bytes, 400);
        assert_eq!(bucket3.min_bytes, 100);
        assert_eq!(bucket3.max_bytes, 300);

        let bucket5 = hist[5].expect("instr_count=5 bucket must be present");
        assert_eq!(bucket5.count, 1);
        assert_eq!(bucket5.sum_bytes, 50);
        assert_eq!(bucket5.min_bytes, 50);
        assert_eq!(bucket5.max_bytes, 50);

        assert!(hist[4].is_none(), "instr_count=4 has no published entries");
    }

    #[test]
    #[cfg(feature = "developer")]
    fn record_reject_increments_both_failed_compiles_and_the_reason_bucket() {
        let stats = JitStats::default();
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::CraneliftVerifierError);

        assert_eq!(stats.failed_compiles.load(Ordering::Relaxed), 3);
        assert_eq!(stats.reject_reasons[RejectReason::EntryExcluded.index()].load(Ordering::Relaxed), 2);
        assert_eq!(stats.reject_reasons[RejectReason::CraneliftVerifierError.index()].load(Ordering::Relaxed), 1);
        assert_eq!(stats.reject_reasons[RejectReason::AnalyzerCodegenDisagreement.index()].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn compile_queue_start_stop_drains_without_hanging() {
        // FakeDevice's read32 always errors, so handle_request bails before
        // any filesystem I/O — no cwd/tempdir isolation needed here.
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default()));
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        for i in 0..8u16 {
            assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: i, compiled_for_fr1: true }, &stats));
        }
        // stop() joins the worker; must return promptly even with requests in flight.
        q.stop();
    }

    /// Every word decodes as `ADDIU r1, r0, 1` — a real, compilable
    /// instruction, unlike `FakeDevice` (which always errors so
    /// `handle_request` bails before ever reaching codegen). Needed for a
    /// genuine end-to-end batching test: `handle_request_deferred` must
    /// actually produce a `FuncId` for there to be anything to batch.
    struct AddiuDevice(AtomicU64);
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok((crate::mips_isa::OP_ADDIU << 26) | (1 << 16) | 1)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
    }

    #[test]
    fn batching_eventually_publishes_via_queue_drain_fallback() {
        // End-to-end: a real worker thread with batching on, one compile
        // request, no second request to ever trigger a page-crossing flush.
        // Without the queue-drain fallback this entry would sit in `pending`
        // forever (nothing else would ever flush it) — this is the specific
        // regression the fallback trigger exists to prevent, exercised
        // through the real threaded path rather than just the synchronous
        // handle_request_deferred/flush_pending_batch unit tests.
        //
        // The polling loop below runs inside catch_unwind specifically so
        // q.stop() always executes afterward, even if the deadline assert!
        // fires: without that, a timeout panic would unwind straight out of
        // this function with the worker thread still running, holding a raw
        // pointer (CompileRequest::page) into `page` below — which is about
        // to be dropped as this function unwinds. That's a genuine
        // use-after-free from the still-live orphaned thread, not a benign
        // leaked-thread annoyance — confirmed live as the actual cause of an
        // intermittent SIGSEGV in unrelated, later-running tests during a
        // full-workspace parallel run (this test's deadline is only ever at
        // real risk of firing under heavy system load, which isolated/
        // single-test runs never reproduce — hence why this wasn't caught
        // immediately).
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        // Force batch on regardless of the build-mode default (developer
        // defaults to off, non-developer to on — this test needs it on to
        // exercise the queue-drain fallback specifically, not testing the
        // default itself).
        q.set_batch_enabled(true);
        q.start(dev, crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default()));

        // A real (non-null) gen counter: page.publish() calls current_gen()
        // unconditionally, which debug_asserts is_compilable() — unlike
        // FakeDevice-based tests elsewhere in this module, this test's
        // AddiuDevice produces real compilable code, so publish() actually
        // gets reached and needs a real counter behind it.
        let gen_counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &gen_counter as *const AtomicU64);
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true }, &stats));

        // 30s, not the original 5s: this test passes in ~0.1s in isolation,
        // but the compile-worker thread genuinely needs real CPU time to get
        // scheduled — 5s proved too tight under a full-workspace parallel
        // test run (confirmed live: this deadline firing, not any actual
        // correctness bug, was tripping under heavy contention from every
        // other test's threads competing for the same cores).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !page.is_entry_valid(4) {
                assert!(std::time::Instant::now() < deadline, "entry never published — queue-drain fallback did not fire");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn batching_page_cross_trigger_publishes_without_waiting_for_queue_drain() {
        // Send enough compile requests, each targeting entry offset 0 of its
        // own distinct physical page, to force at least one page-crossing
        // flush (in the Cranelift arena's host-page-segment sense — not to
        // be confused with the distinct MIPS *physical* pages these requests
        // target, a coincidental naming overlap) before the queue ever
        // drains. Deliberately one request per PhysicalCodePage, all at
        // offset 0: a single walk_bounded from offset 0 with
        // MAX_INSTRS_PER_COMPILE=usize::MAX would otherwise treat many
        // requests against *different offsets of the same page* as one
        // giant sequential region (every word here decodes as the same
        // branch-free ADDIU), which isn't 300 independent compiles at
        // all — this was tried first and confirmed the wrong shape for this
        // test; distinct pages avoids that entirely and matches how real
        // page-cross-worthy traffic (many distinct hot pages arriving close
        // together) actually looks.
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.set_batch_enabled(true);
        q.start(dev, crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default()));

        const N: usize = 300;
        let gen_counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        let mut pages: Vec<PhysicalCodePage> = gen_counters.iter()
            .enumerate()
            .map(|(i, counter)| PhysicalCodePage::new(i as Pfn, counter as *const AtomicU64))
            .collect();
        let stats = JitStats::default();
        for page in pages.iter_mut() {
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));
        }

        // Poll for the *first* page specifically, well before the full
        // queue could possibly have drained (300 requests, 200µs backoff
        // only between empty-queue polls — draining that many keeps the
        // worker continuously busy, not backed off): if only the page-cross
        // trigger (not queue-drain) is what eventually publishes it, this
        // still succeeds this early. `stop()` itself does NOT guarantee a
        // full drain (it just flips `running` and joins whatever the loop's
        // current iteration is, same as any other "stop now" contract) — so
        // this must observe the page-cross trigger firing before stopping,
        // not rely on stop() to finish the queue first.
        //
        // catch_unwind for the same reason as the queue-drain-fallback test
        // above: without it, a deadline timeout here would unwind out of
        // this function with the worker thread still running against
        // `pages`/`gen_counters`, about to be dropped — a genuine
        // use-after-free from the orphaned thread, not a benign leaked
        // thread. This was the confirmed root cause of an intermittent
        // SIGSEGV surfacing in unrelated, later-running tests under
        // full-workspace parallel load.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !pages[0].is_entry_valid(0) {
                assert!(std::time::Instant::now() < deadline, "entry never published within the timeout");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn compile_queue_send_before_start_still_delivered_after_start() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));
        q.start(dev, crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default()));
        q.stop();
    }

    #[test]
    fn compile_queue_send_drops_when_full() {
        let mut q = CompileQueue::new();
        // Don't start the worker: nothing drains, so capacity fills exactly.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        let mut accepted = 0;
        for i in 0..(COMPILE_QUEUE_CAPACITY + 10) {
            if q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: i as u16, compiled_for_fr1: true }, &stats) {
                accepted += 1;
            }
        }
        assert_eq!(accepted, COMPILE_QUEUE_CAPACITY, "queue must drop pushes past capacity, not block or panic");
    }

    #[test]
    fn compile_queue_start_is_idempotent() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev.clone(), crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default()));
        q.start(dev, crate::jitv2::codegen::Codegen::new(), std::sync::Arc::new(JitStats::default())); // must not panic or spawn a second thread
        q.stop();
    }

    #[test]
    fn compile_queue_stop_without_start_is_a_noop() {
        let mut q = CompileQueue::new();
        q.stop(); // must not panic
    }
}
