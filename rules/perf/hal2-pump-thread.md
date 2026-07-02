# HAL2 dedicated pump thread — do not use thread::sleep

**Keywords:** hal2,audio,scratch,stutter,pump,hptimer,sleep,windows

Phase 3 briefly moved Codec A output from `TimerManager` to a `HAL2-Pump` thread
that called `thread::sleep(period - elapsed)` with `period ≈ 1/44100 s` (~23 µs).

## Why it failed

`hptimer` (`src/hptimer.rs`) **spins** for waits under ~200 µs so 44.1 kHz audio
ticks stay on schedule. `thread::sleep` on Windows — even with `timeBeginPeriod(1)`
from `Hal2::start()` — cannot reliably sleep for 23 µs; effective wakeups land near
**1 ms**, so DMA was drained ~40× too slowly. The cpal ring buffer underran constantly
(scratchy/crackling audio).

Slow PDMA drain may also stress the IRIX audio path; treat audio regressions as
P0 before chasing unrelated kernel panics.

## Correct approach

- **Now:** Codec A uses dedicated `HAL2-Pump` thread with hptimer-style spin (no `thread::sleep`).
- **Legacy:** Codec A was on `TimerManager` / `hptimer` before Phase 3 master plan.
- **Future isolation:** If a dedicated thread is retried, it must use the same
  spin/park policy as `timer_thread_loop` (spin for `delay < 200 µs`), not sleep.

## Verify after changes

Telnet monitor `hal2 status` — `cpal underruns` should stay low during desktop idle
and under `glxgears`. Premiere: `wsl\run-iris-premiere.bat`.

## Kernel panic after earlier crashes

If you see `Read TLB Miss` at `0xff800000` repeatedly, the **root disk may be
corrupted** from a prior panic — not a JIT bug. Enable `overlay = true` on
`[scsi.1]` and restore from a clean `scsi1.raw` copy. See
`rules/testing/disk-image-hygiene.md`.
