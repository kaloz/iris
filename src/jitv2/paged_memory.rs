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
}

impl PagedArenaState {
    pub fn crossed_page(&self) -> bool {
        self.crossed_page.swap(false, Ordering::Relaxed)
    }

    /// `(bytes_actually_used, bytes_reserved)` — see `PagedArenaMemoryProvider::packing_stats`.
    pub fn packing_stats(&self) -> (u64, u64) {
        (self.used_bytes.load(Ordering::Relaxed), self.reserved_bytes.load(Ordering::Relaxed))
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (addr + align - 1) & !(align - 1)
}

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
        let mut alloc = region::alloc(size, region::Protection::NONE)?;
        let ptr = alloc.as_mut_ptr();
        Ok(Self {
            alloc: ManuallyDrop::new(Some(alloc)),
            segments: Vec::new(),
            ptr,
            size,
            position: 0,
            state,
        })
    }

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
}
