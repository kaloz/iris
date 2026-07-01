# x86_64 Loads-tier helper limit + speculative graduation

**Keywords:** x86_64, Loads, silent quit, userspace, helper diamond, speculative
**Category:** jit

## Symptom

IRIX boots at 50–80 MIPS; apps launch quickly then **quit with no error**. Same
with premiere (`max_tier=1`) and full-tier experimental launcher.

## Cause

On x86_64, Cranelift regalloc2 miscompiles blocks with **more than one** load
helper diamond (`rules/jit/cranelift-regalloc2-helper-diamond-limit-is-platform-dependent.md`).
`trace_block` only limited helpers for Full tier and used `max_helpers=64`.

Graduating **Loads** blocks out of speculative mode after 50 stable hits removed
rollback/demotion for those miscompiles → silent userspace corruption.

## Fix (dispatch.rs)

1. `max_helpers = 1` on x86_64, `3` on aarch64 — applies to **Loads and Full**.
2. `speculative_may_graduate`: Alu yes; Loads only on aarch64; Full never.

## Verify

```bat
wsl\run-iris-premiere-nojit.bat    # stable → was JIT
wsl\capture-app-crash.ps1 -Verify  # JIT VERIFY FAIL pinpoints PC if still broken
```

Also check RAM: IRIX 6.5 should use 384 MB (`iris-windows-384.toml`), not 512 MB.
