# Premiere Windows TLBMISS at 0xff800000 — usually JIT, not disk

**Keywords:** premiere, Windows, TLBMISS, ff800000, jit-profile, max_tier, kernel fault
**Category:** jit

## Symptom

`run-iris-premiere.bat` panics with `PANIC: TLBMISS: KERNEL FAULT`, bad addr
`0xff800000`, PC in `0x8800xxxx` kernel range. Log shows JIT promoting blocks
(e.g. `8800797c`) to **Full** tier and PC spinning at `88007978–88007984`.

Same `scsi1.raw` boots fine on Linux GUI with overlay — disk is not the cause.

## Cause

Windows premiere enables MIPS JIT with `max_tier=2` (Full) plus a saved
`jit-profile.bin` that replays hot kernel blocks at Full tier. A miscompiled
Full block can issue a bad KSEG3 read → read TLB miss that looks like kernel
corruption.

COW overlay with `dirty sectors: 0` also rules out filesystem damage from this
session (see `rules/testing/disk-image-hygiene.md` for when panics *are* disk).

## Workarounds

1. `wsl\run-iris-premiere-safe.bat` — `IRIS_JIT_MAX_TIER=1`, isolated profile
2. Delete or rename `jit-profile.bin` (project root or `%USERPROFILE%\.iris\`)
3. `set IRIS_JIT=0` before premiere.bat — interpreter-only A/B test
4. Compare: Linux GUI may run without JIT env / lower tier

## Fix direction

Profile replay must respect `IRIS_JIT_MAX_TIER` (capped in dispatch.rs).
Profile entries replay at Loads tier max; Full is in-session promotion only.
Load-only blocks at **Full** tier must **never** graduate out of speculative mode at the
stable threshold — graduating Loads/Alu is fine for speed. Premiere defaults to
`max_tier=1` (Loads) on Windows until Full-tier codegen is verified.
