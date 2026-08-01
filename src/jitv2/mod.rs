//! JIT v2: physical-page PIC region compiler. See `rules/jitv2/jit-v2-design.md`.

pub mod jitv2;
pub mod comp;
pub mod opcode_support;
pub mod analyzer;
pub mod codegen;
pub mod paged_memory;
pub mod equiv_test;

pub use jitv2::{
    CompileQueue, CompileRequest, JitEntry, JitFn, JitStats, Jitv2, PageSlot, Pfn, PhysicalCodePage,
    BITMAP_WORDS, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES, COMPILE_QUEUE_CAPACITY, ENTRIES_PER_PAGE,
    JITV2_INITIAL_PAGE_CAPACITY, PAGE_SIZE,
    min_calls_before_compile, set_min_calls_before_compile,
};
pub use paged_memory::{PagedArenaMemoryProvider, PagedArenaState};
#[cfg(feature = "developer")]
pub use jitv2::{BatchFlushReason, CodeSizeBucket, RejectReason, REJECT_REASON_COUNT};
