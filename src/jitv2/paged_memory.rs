//! Custom `cranelift_jit::JITMemoryProvider` for lazy, page-batched
//! finalization (see `rules/jitv2/jit-v2-design.md`'s memory-packing notes).
//!
//! `cranelift_jit::ArenaMemoryProvider` (the stock implementation) already
//! does everything we need *except* tell the caller when a new host-page
//! segment was just started — its internals (`segments`, `position`,
//! `finalized`) are `pub(crate)`-sealed, and `JITMemoryProvider`'s trait
//! surface (`allocate`/`free_memory`/`finalize`) exposes nothing about
//! segment boundaries either. There is no way to observe "did the last
//! allocation cross into a new host page" through the public API — the only
//! way to get that signal is to own the allocator.
//!
//! This is a near-verbatim port of `cranelift_jit::memory::arena`'s
//! `ArenaMemoryProvider`/`Segment` (same bump-allocator-into-a-`PROT_NONE`-
//! reservation shape, same `region` crate primitives, so it inherits
//! identical cross-platform behavior on Linux/macOS/Windows with no
//! OS-specific code of its own) with one addition: a shared [`PagedArenaState`]
//! that `allocate()` updates on every call (page-crossing signal, packing
//! stats). `Codegen` polls it after every compile to decide when to flush a
//! batch of pending (compiled-but-not-yet-finalized) functions — see
//! `Codegen::provider_crossed_page`.
//!
//! Why this is needed at all: `finalize_definitions()` is what flips a
//! segment from RW to RX and marks it permanently `finalized` (blocking any
//! further allocation into it) — call it after every single compile (as
//! `compile_region` used to) and every function gets its own segment, always
//! rounded up to a full host page regardless of actual code size. Deferring
//! `finalize_definitions()` across several compiles lets them pack into the
//! same segment, but `JITModule::get_finalized_function` asserts the
//! function isn't still pending finalization — so a caller batching finalize
//! calls also has to batch handing back `JitFn` pointers (and therefore
//! `page.publish()`) until the batch's one `finalize_definitions()` call
//! actually runs.

use std::io;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::ModuleResult;

/// Shared state a `PagedArenaMemoryProvider` publishes and `Codegen` polls —
/// split out from the provider itself because `cranelift_jit::JITModule`
/// takes ownership of the provider as an opaque `Box<dyn JITMemoryProvider +
/// Send>` and never gives any of it back (no downcast, no accessor). An
/// `Arc` shared between the provider (which writes it from inside
/// `allocate`/`finalize`, called only from `JITModule::define_function`/
/// `finalize_definitions`, both driven single-threaded by `Codegen`'s own
/// caller) and `Codegen` (which reads it after each such call) is the only
/// channel back out. Atomics rather than a `Mutex` since this is
/// same-thread producer/consumer in practice (nothing here is genuinely
/// concurrent) — atomics just make that safe to express without a lock.
#[derive(Default)]
pub struct PagedArenaState {
    /// Set true whenever an `allocate()` call started a brand-new segment
    /// (as opposed to packing into, or growing, the existing unfinalized
    /// one). `Codegen::provider_crossed_page` reads-and-clears this after
    /// every compile.
    crossed_page: AtomicBool,
    /// Sum of every segment's bump-allocator cursor (`Segment::position`) —
    /// the real code+relocation bytes written, across every segment ever
    /// allocated (finalized or not). See `PagedArenaMemoryProvider::packing_stats`.
    used_bytes: AtomicU64,
    /// Sum of every segment's reserved length (`Segment::len`), always a
    /// multiple of the host page size.
    reserved_bytes: AtomicU64,
    /// The arena's real (hugepage-aligned, on Linux) base address and total
    /// reserved length — written once, by `new_with_size`, before the
    /// provider is ever handed to `JITModule`; read-only from then on (no
    /// `Ordering` subtlety needed beyond Relaxed — this is set-once-then-frozen,
    /// not a value that changes concurrently with reads). `j2 hugepages`
    /// (`developer` only) uses this to scope its `/proc/self/smaps`
    /// `AnonHugePages` query to exactly this arena's address range.
    arena_ptr: AtomicU64,
    arena_len: AtomicU64,
}

impl PagedArenaState {
    pub fn crossed_page(&self) -> bool {
        self.crossed_page.swap(false, Ordering::Relaxed)
    }

    /// `(bytes_actually_used, bytes_reserved)` — see `PagedArenaMemoryProvider::packing_stats`.
    pub fn packing_stats(&self) -> (u64, u64) {
        (self.used_bytes.load(Ordering::Relaxed), self.reserved_bytes.load(Ordering::Relaxed))
    }

    /// `(base_address, len)` of the arena's real (possibly hugepage-aligned)
    /// reservation — `(0, 0)` before `new_with_size` has run. Dev diagnostic
    /// only (`j2 hugepages`); nothing on the hot compile path reads this.
    #[cfg(feature = "developer")]
    pub fn arena_range(&self) -> (u64, u64) {
        (self.arena_ptr.load(Ordering::Relaxed), self.arena_len.load(Ordering::Relaxed))
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (addr + align - 1) & !(align - 1)
}

/// Transparent-hugepage size this arena aligns/collapses against — 2MiB, the
/// standard THP size on Linux (the only OS any of this hugepage machinery
/// runs on at all — see this constant's `#[cfg]`: Windows has no madvise or
/// THP concept, large pages there require `MEM_LARGE_PAGES` +
/// `SeLockMemoryPrivilege` at allocation time, a different mechanism
/// entirely; macOS has no public THP/superpage API on Apple Silicon, and the
/// old x86 `VM_FLAGS_SUPERPAGE_SIZE_2MB` is dead). Not queried at runtime
/// (the real value lives in
/// `/sys/kernel/mm/transparent_hugepage/hpage_pmd_size`, a file read neither
/// `region` nor `libc` wrap) — 2MiB is correct for both mainstream Linux
/// hugepage-capable architectures (x86_64, aarch64) this project ships on;
/// every operation below is advisory (`madvise` failures are logged, never
/// fatal) and alignment only wastes a little reserved-but-unused address
/// space if that ever stops being true, never incorrect behavior.
#[cfg(target_os = "linux")]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Best-effort `madvise(MADV_HUGEPAGE)` over the whole reservation, called
/// once right after `region::alloc` — asks the kernel to prefer backing this
/// VMA with transparent hugepages as it gets populated (khugepaged can also
/// promote it later without this, but async promotion can take a scan cycle
/// or more to kick in; asking upfront means a freshly-compiled hot function
/// doesn't have to wait for that). Purely advisory: failure is logged once
/// and otherwise ignored — this is a throughput optimization, never a
/// correctness requirement, and the arena works identically (just without
/// the TLB-pressure benefit) if the kernel declines or the platform lacks
/// `MADV_HUGEPAGE` entirely (see the `#[cfg]` gate on this whole family of
/// functions).
#[cfg(target_os = "linux")]
fn madvise_hugepage(ptr: *mut u8, len: usize) {
    let rc = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE) };
    if rc != 0 {
        eprintln!("jitv2: madvise(MADV_HUGEPAGE) failed on paged arena reservation: {}", std::io::Error::last_os_error());
    }
}
#[cfg(not(target_os = "linux"))]
fn madvise_hugepage(_ptr: *mut u8, _len: usize) {}

/// Best-effort `madvise(MADV_COLLAPSE)` over `[ptr, ptr+len)`, which must
/// already be hugepage-aligned and hugepage-sized (caller's responsibility —
/// see `PagedArenaMemoryProvider::advance_hugepage_collapse_watermark`, the
/// only caller). Unlike `MADV_HUGEPAGE` (a standing preference the kernel
/// acts on lazily, via khugepaged's background scanner), `MADV_COLLAPSE`
/// (Linux 6.1+) synchronously compacts the region into a hugepage right now
/// — worth paying for once a region is fully written and sealed (RX,
/// finalized) rather than waiting for khugepaged's own schedule, which is
/// tuned for general workloads, not "this code just got hot, please collapse
/// it immediately." Best-effort: a kernel without `MADV_COLLAPSE` support
/// (pre-6.1) returns `ENOSYS`/`EINVAL`, silently ignored — no version probe
/// needed, the syscall's own failure path is the probe.
#[cfg(target_os = "linux")]
fn madvise_collapse(ptr: *mut u8, len: usize) {
    let _ = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_COLLAPSE) };
}
#[cfg(not(target_os = "linux"))]
fn madvise_collapse(_ptr: *mut u8, _len: usize) {}

/// Dev diagnostic (`j2 hugepages`): sum the `AnonHugePages:` field of every
/// `/proc/self/smaps` VMA entry that falls (even partially) within
/// `[range_start, range_start+range_len)`. Lets a live run confirm THP is
/// actually landing on the JIT arena, rather than assuming `MADV_HUGEPAGE`/
/// `MADV_COLLAPSE` succeeded from their own (best-effort, silently-ignored)
/// return codes alone — a host with `transparent_hugepage=never` in sysfs,
/// or a container/cgroup that disables THP, makes every `madvise` call here
/// a silent no-op, and this is the only way short of external tooling
/// (`grep AnonHugePages /proc/<pid>/smaps`) to notice that's happening
/// before it shows up as an unexplained iTLB-miss regression.
///
/// smaps format per VMA: a header line (`<start>-<end> <perms> ...`)
/// followed by indented `Key:    N kB` lines until the next header. Only
/// `AnonHugePages` is summed; every other field is skipped. Returns `None`
/// if `/proc/self/smaps` can't be opened/read at all (containerized/
/// sandboxed environments sometimes restrict it) — the caller reports that
/// as "unavailable," not as "0 hugepages," since those mean different things.
#[cfg(all(target_os = "linux", feature = "developer"))]
pub fn anon_hugepages_in_range(range_start: u64, range_len: u64) -> Option<u64> {
    let range_end = range_start + range_len;
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut total_kb: u64 = 0;
    let mut in_range = false;
    for line in smaps.lines() {
        if let Some((addrs, _rest)) = line.split_once(' ') {
            if let Some((start_hex, end_hex)) = addrs.split_once('-') {
                if let (Ok(start), Ok(end)) = (u64::from_str_radix(start_hex, 16), u64::from_str_radix(end_hex, 16)) {
                    // A real smaps header line always parses both halves as
                    // hex; anything that doesn't (an indented "Key: N kB"
                    // line) falls through to the `else` below instead.
                    in_range = start < range_end && end > range_start;
                    continue;
                }
            }
        }
        if in_range {
            if let Some(rest) = line.strip_prefix("AnonHugePages:") {
                if let Some(kb) = rest.trim().strip_suffix(" kB").and_then(|n| n.trim().parse::<u64>().ok()) {
                    total_kb += kb;
                }
            }
        }
    }
    Some(total_kb * 1024)
}
#[cfg(not(all(target_os = "linux", feature = "developer")))]
pub fn anon_hugepages_in_range(_range_start: u64, _range_len: u64) -> Option<u64> { None }

/// Port of `cranelift_jit::memory::set_readable_and_executable` — `pub(crate)`
/// in that crate, so not reachable from here; duplicated verbatim rather than
/// reimplemented from scratch, to stay exactly in sync with the icache/BTI
/// handling the stock `ArenaMemoryProvider` relies on. Clears the icache for
/// the newly-written code before flipping the mapping to RX (some CPUs have
/// errata around doing this after, per the original's own comment), then
/// applies ARM BTI protection if requested and supported. Does *not* flush
/// the instruction pipeline (`wasmtime_jit_icache_coherence::pipeline_flush_mt`)
/// — same as the original, that's the caller's job once every segment in a
/// batch has been finalized (`PagedArenaMemoryProvider::finalize`).
fn set_readable_and_executable(ptr: *mut u8, len: usize, branch_protection: BranchProtection) {
    unsafe {
        wasmtime_jit_icache_coherence::clear_cache(ptr as *const std::ffi::c_void, len)
            .expect("Failed cache clear")
    };

    unsafe {
        region::protect(ptr, len, region::Protection::READ_EXECUTE)
            .expect("unable to make jitv2 paged arena segment readable+executable");
    }

    if branch_protection == BranchProtection::BTI {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        if std::arch::is_aarch64_feature_detected!("bti") {
            let prot = libc::PROT_EXEC | libc::PROT_READ | /* PROT_BTI */ 0x10;
            unsafe {
                assert!(
                    libc::mprotect(ptr as *mut libc::c_void, len, prot) >= 0,
                    "unable to make jitv2 paged arena segment readable+executable with BTI: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

#[derive(Debug)]
struct Segment {
    ptr: *mut u8,
    len: usize,
    position: usize,
    target_prot: region::Protection,
    finalized: bool,
}

impl Segment {
    fn new(ptr: *mut u8, len: usize, target_prot: region::Protection) -> Self {
        debug_assert_eq!(ptr as usize % region::page::size(), 0);
        debug_assert_eq!(len % region::page::size(), 0);
        let mut segment = Segment { ptr, len, target_prot, position: 0, finalized: false };
        segment.set_rw();
        segment
    }

    fn set_rw(&mut self) {
        unsafe {
            region::protect(self.ptr, self.len, region::Protection::READ_WRITE)
                .expect("unable to change memory protection for jitv2 paged arena segment");
        }
    }

    fn finalize(&mut self, branch_protection: BranchProtection) {
        if self.finalized {
            return;
        }
        if self.target_prot == region::Protection::READ_EXECUTE {
            set_readable_and_executable(self.ptr, self.len, branch_protection);
        } else {
            unsafe {
                region::protect(self.ptr, self.len, self.target_prot)
                    .expect("unable to change memory protection for jitv2 paged arena segment");
            }
        }
        self.finalized = true;
    }

    fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        assert!(self.has_space_for(size, align));
        self.position = align_up(self.position, align);
        let ptr = unsafe { self.ptr.add(self.position) };
        self.position += size;
        ptr
    }

    fn has_space_for(&self, size: usize, align: usize) -> bool {
        !self.finalized && align_up(self.position, align) + size <= self.len
    }
}

/// `ArenaMemoryProvider`-alike with an added page-crossing signal — see
/// this module's doc comment for the full rationale.
pub struct PagedArenaMemoryProvider {
    alloc: ManuallyDrop<Option<region::Allocation>>,
    ptr: *mut u8,
    size: usize,
    position: usize,
    segments: Vec<Segment>,
    /// How much of `[ptr, ptr+size)` has already been `MADV_COLLAPSE`d, as a
    /// byte offset from `ptr` — always a multiple of `HUGE_PAGE_SIZE` (or 0
    /// on a platform without hugepage support, where it never advances).
    /// Monotonic: `finalize()` is the only writer, advancing it by whole
    /// hugepage-sized steps as segments seal past each boundary — see
    /// `advance_hugepage_collapse_watermark`. Segments are allocated
    /// contiguously (`self.position` only ever grows, `allocate_segment`
    /// always places the next segment right after the last), so a single
    /// watermark is sufficient — there are never gaps to track separately.
    collapsed_up_to: usize,
    /// Shared with whoever constructed this provider (`Codegen`) — see
    /// `PagedArenaState`'s own doc comment for why this indirection exists
    /// (`JITModule` takes ownership of the provider as an opaque boxed trait
    /// object and never gives any of it back).
    state: Arc<PagedArenaState>,
}

unsafe impl Send for PagedArenaMemoryProvider {}

impl PagedArenaMemoryProvider {
    pub fn new_with_size(reserve_size: usize, state: Arc<PagedArenaState>) -> Result<Self, region::Error> {
        let size = align_up(reserve_size, region::page::size());
        // Over-reserve by one extra hugepage and use the hugepage-aligned
        // address within that larger mapping as the arena's real base — the
        // padding before/after is virtual address space only (still
        // PROT_NONE / never touched), never physically backed, so this
        // costs nothing but address space. `region::alloc` gives no
        // alignment control of its own (a plain `mmap(NULL, size, ...)`
        // under the hood — kernel picks a page-aligned but not necessarily
        // hugepage-aligned base), so this is the standard way to get a
        // hugepage-aligned region through it. `self.alloc` keeps the
        // *original*, untrimmed `Allocation` for correct freeing (its
        // `Drop` frees by its own stored base/size, unaffected by what
        // `self.ptr` below points at) — only `self.ptr`/`self.size` (what
        // segments actually get carved out of) are the aligned sub-region.
        // On a platform with no hugepage support (HUGE_PAGE_SIZE undefined),
        // this degenerates to exactly the old unaligned behavior.
        #[cfg(target_os = "linux")]
        let (mut alloc, ptr, size) = {
            let over_reserved = size + HUGE_PAGE_SIZE;
            let mut alloc = region::alloc(over_reserved, region::Protection::NONE)?;
            let base = alloc.as_mut_ptr::<u8>() as usize;
            let aligned = align_up(base, HUGE_PAGE_SIZE);
            (alloc, aligned as *mut u8, size)
        };
        #[cfg(not(target_os = "linux"))]
        let (mut alloc, ptr, size) = {
            let alloc = region::alloc(size, region::Protection::NONE)?;
            let ptr = alloc.as_ptr::<u8>() as *mut u8;
            (alloc, ptr, size)
        };
        let _ = &mut alloc; // silence unused_mut when the aligned-address branch doesn't need it
        madvise_hugepage(ptr, size);
        #[cfg(feature = "developer")]
        {
            state.arena_ptr.store(ptr as u64, Ordering::Relaxed);
            state.arena_len.store(size as u64, Ordering::Relaxed);
        }
        Ok(Self {
            alloc: ManuallyDrop::new(Some(alloc)),
            segments: Vec::new(),
            ptr,
            size,
            position: 0,
            collapsed_up_to: 0,
            state,
        })
    }

    /// Called from `finalize()` after every segment in the batch has been
    /// flipped RW->RX: `MADV_COLLAPSE` every hugepage-sized region that has
    /// now become fully sealed (every byte in it belongs to some finalized
    /// segment) since the last call, advancing `collapsed_up_to`. Never
    /// touches the region past the last *finalized* segment's end — the
    /// arena's own bump cursor (`self.position`) can extend further if a
    /// not-yet-finalized segment already claimed address space beyond it,
    /// and collapsing memory that's still being actively written by
    /// Cranelift (RW, mid-compile) would be both wasted work (it'll just get
    /// split again on the next write past a THP boundary) and pointless
    /// (the whole point is compacting *finished*, stable code).
    #[cfg(target_os = "linux")]
    fn advance_hugepage_collapse_watermark(&mut self) {
        let finalized_end = self.segments.iter()
            .filter(|s| s.finalized)
            .map(|s| (s.ptr as usize - self.ptr as usize) + s.len)
            .max()
            .unwrap_or(0);
        let target = (finalized_end / HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE; // round DOWN — only whole, fully-sealed hugepages
        if target > self.collapsed_up_to {
            let len = target - self.collapsed_up_to;
            let region_ptr = unsafe { self.ptr.add(self.collapsed_up_to) };
            madvise_collapse(region_ptr, len);
            self.collapsed_up_to = target;
        }
    }
    #[cfg(not(target_os = "linux"))]
    fn advance_hugepage_collapse_watermark(&mut self) {}

    fn record_packing(&self) {
        // Recomputed from scratch (sum across all segments) rather than
        // incrementally adjusted, since a segment's `position`/`len` can
        // both grow after it was first counted (packing more functions in,
        // or the "resize the last segment" growth path) — simplest to just
        // re-sum on every allocation than track deltas correctly for both
        // cases. O(segment count) per compile, and segment count is bounded
        // by however many host pages the arena has handed out so far, not
        // by function count — cheap in practice.
        let used: u64 = self.segments.iter().map(|s| s.position as u64).sum();
        let reserved: u64 = self.segments.iter().map(|s| s.len as u64).sum();
        self.state.used_bytes.store(used, Ordering::Relaxed);
        self.state.reserved_bytes.store(reserved, Ordering::Relaxed);
    }

    fn allocate_inner(&mut self, size: usize, align: u64, protection: region::Protection) -> io::Result<*mut u8> {
        let align = usize::try_from(align).expect("alignment too big");
        assert!(align <= region::page::size(), "alignment over page size is not supported");

        if let Some(i) = self.segments.iter().position(|seg| {
            seg.target_prot == protection && !seg.finalized && seg.has_space_for(size, align)
        }) {
            let ptr = self.segments[i].allocate(size, align);
            self.state.crossed_page.store(false, Ordering::Relaxed);
            self.record_packing();
            return Ok(ptr);
        }

        if let Some(last) = self.segments.len().checked_sub(1) {
            if self.segments[last].target_prot == protection && !self.segments[last].finalized {
                let additional_size = align_up(size, region::page::size());
                if self.position + additional_size <= self.size {
                    self.segments[last].len += additional_size;
                    self.segments[last].set_rw();
                    self.position += additional_size;
                    let ptr = self.segments[last].allocate(size, align);
                    self.state.crossed_page.store(false, Ordering::Relaxed);
                    self.record_packing();
                    return Ok(ptr);
                }
            }
        }

        self.allocate_segment(size, protection)?;
        self.state.crossed_page.store(true, Ordering::Relaxed);
        let i = self.segments.len() - 1;
        let ptr = self.segments[i].allocate(size, align);
        self.record_packing();
        Ok(ptr)
    }

    fn allocate_segment(&mut self, size: usize, target_prot: region::Protection) -> Result<(), io::Error> {
        let size = align_up(size, region::page::size());
        let ptr = unsafe { self.ptr.add(self.position) };
        if self.position + size > self.size {
            return Err(io::Error::new(io::ErrorKind::Other, "pre-allocated jit memory region exhausted"));
        }
        self.position += size;
        self.segments.push(Segment::new(ptr, size, target_prot));
        Ok(())
    }

    pub(crate) fn finalize(&mut self, branch_protection: BranchProtection) {
        for segment in &mut self.segments {
            segment.finalize(branch_protection);
        }
        self.advance_hugepage_collapse_watermark();
        wasmtime_jit_icache_coherence::pipeline_flush_mt().expect("Failed pipeline flush");
    }

    pub(crate) unsafe fn free_memory(&mut self) {
        if self.ptr == ptr::null_mut() {
            return;
        }
        self.segments.clear();
        let _: Option<region::Allocation> = self.alloc.take();
        self.ptr = ptr::null_mut();
    }
}

impl Drop for PagedArenaMemoryProvider {
    fn drop(&mut self) {
        if self.ptr == ptr::null_mut() {
            return;
        }
        let is_live = self.segments.iter().any(|seg| seg.finalized);
        if !is_live {
            unsafe { self.free_memory() };
        }
    }
}

impl JITMemoryProvider for PagedArenaMemoryProvider {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        self.allocate_inner(
            size,
            align,
            match kind {
                JITMemoryKind::Executable => region::Protection::READ_EXECUTE,
                JITMemoryKind::Writable => region::Protection::READ_WRITE,
                JITMemoryKind::ReadOnly => region::Protection::READ,
            },
        )
    }

    unsafe fn free_memory(&mut self) {
        self.free_memory();
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        self.finalize(branch_protection);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_arena(size: usize) -> (PagedArenaMemoryProvider, Arc<PagedArenaState>) {
        let state = Arc::new(PagedArenaState::default());
        let arena = PagedArenaMemoryProvider::new_with_size(size, state.clone()).unwrap();
        (arena, state)
    }

    #[test]
    fn first_allocation_starts_a_segment_and_reports_crossed_page() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page(), "the very first allocation must start a new segment");
    }

    #[test]
    fn crossed_page_is_edge_triggered_and_clears_on_read() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        assert!(!state.crossed_page(), "a second read without an intervening allocation must report false");
    }

    #[test]
    fn packing_into_the_same_unfinalized_segment_does_not_cross_a_page() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(!state.crossed_page(), "a second small allocation must pack into the first segment, not start a new one");
    }

    #[test]
    fn allocation_bigger_than_a_page_starts_a_new_segment() {
        let (mut arena, state) = new_arena(16 << 20);
        let page = region::page::size();
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        // A big allocation grows the existing unfinalized segment in place
        // (allocate_inner's "resize the last segment" branch) rather than
        // starting a distinct one — that's still packing, not a page
        // crossing, as long as the segment isn't finalized yet.
        arena.allocate(page * 3, 16, JITMemoryKind::Executable).unwrap();
        assert!(!state.crossed_page(), "growing the same unfinalized segment must not report a page crossing");
    }

    #[test]
    fn finalizing_forces_the_next_allocation_to_cross_a_page() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        arena.finalize(BranchProtection::None);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page(), "a finalized segment must never accept another allocation");
    }

    #[test]
    fn packing_stats_track_used_vs_reserved_bytes() {
        let (mut arena, state) = new_arena(1 << 20);
        let (used0, reserved0) = state.packing_stats();
        assert_eq!(used0, 0);
        assert_eq!(reserved0, 0);

        // Both sizes are multiples of the align, so no alignment padding
        // between them — the sum comes out exact and easy to hand-verify.
        arena.allocate(96, 16, JITMemoryKind::Executable).unwrap();
        let (used1, reserved1) = state.packing_stats();
        assert_eq!(used1, 96);
        assert_eq!(reserved1, region::page::size() as u64);

        arena.allocate(208, 16, JITMemoryKind::Executable).unwrap();
        let (used2, reserved2) = state.packing_stats();
        assert_eq!(used2, 304, "used bytes must sum across packed allocations");
        assert_eq!(reserved2, region::page::size() as u64, "reserved must stay one page while packing continues to fit");
    }

    #[test]
    fn over_capacity_returns_err_not_panic() {
        let (mut arena, _state) = new_arena(1 << 20);
        arena.allocate(900_000, 1, JITMemoryKind::Executable).unwrap();
        assert!(arena.allocate(200_000, 1, JITMemoryKind::Executable).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn arena_base_is_hugepage_aligned() {
        // The whole point of over-reserving by one extra hugepage in
        // new_with_size — see its own doc comment. Every segment is carved
        // out of self.ptr, so if this holds, every segment's own address is
        // automatically hugepage-aligned too (segments are 4KiB-page
        // multiples, which divide evenly into a 2MiB-aligned base).
        let (arena, _state) = new_arena(1 << 20);
        assert_eq!(arena.ptr as usize % HUGE_PAGE_SIZE, 0, "arena base must be hugepage-aligned");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn collapse_watermark_advances_only_past_fully_finalized_hugepages() {
        // Arena big enough to span more than one hugepage, so we can prove
        // the watermark tracks real boundaries rather than always being 0 or
        // always being everything.
        let (mut arena, _state) = new_arena(HUGE_PAGE_SIZE * 3);

        // Fill and finalize enough small segments to cross the first
        // hugepage boundary, but stop partway into the second — the
        // watermark must advance exactly one hugepage's worth, not more (the
        // still-open, not-yet-finalized segment past the boundary must never
        // be counted as "sealed").
        let chunk = region::page::size();
        let mut bytes_allocated = 0usize;
        while bytes_allocated < HUGE_PAGE_SIZE + chunk {
            arena.allocate(chunk, 16, JITMemoryKind::Executable).unwrap();
            arena.finalize(BranchProtection::None); // seals the growing segment each time, forcing the next allocate() to start a fresh one
            bytes_allocated += align_up(chunk, region::page::size());
        }

        assert_eq!(arena.collapsed_up_to, HUGE_PAGE_SIZE,
            "watermark must advance exactly one whole hugepage once that much has been finalized, not the partial second hugepage too");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn collapse_watermark_does_not_advance_below_one_hugepage() {
        let (mut arena, _state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        arena.finalize(BranchProtection::None);
        assert_eq!(arena.collapsed_up_to, 0, "a single small finalized segment must not advance the watermark -- it's nowhere near a full hugepage");
    }
}
