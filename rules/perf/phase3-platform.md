# Phase 3 platform notes

## Unified config

`MachineConfig` now includes `[jit]`, `[perf]`, and `[machine]` TOML sections. CLI applies
`cfg.jit.apply_env()` at startup; iris-gui syncs Debug tab edits into `cfg.jit` before Start.

## Windows CI

Default `ci_socket = "127.0.0.1:19851"` (TCP). Unix builds keep `/tmp/iris.sock`.
`iris-ci` connects to either form. Launch headless CI with `wsl/run-iris-ci.bat`.

## HAL2

Codec A output uses the shared **hptimer** (`TimerManager`) recurring timer at the
codec pitch rate (~44.1 kHz). A Phase 3 experiment with `thread::sleep` on a
dedicated pump thread caused scratchy audio on Windows — see
`rules/perf/hal2-pump-thread.md`. Codec B/AES still use hptimer as well.

## Graphics

- `Rex3Screen.fb_borrowed`: skip 16 MB copy when refresh uses heartbeat-only FB path.
- `CompositorSource.dirty_y0/y1`: partial GL texture upload for status-bar-only frames.
- GUI: partial egui upload (Phase 2) + live VRAM borrow (Phase 3).

## JIT stores

Set `[jit] compile_stores = true` (or uncheck "Disable JIT stores" in GUI) to clear
`IRIS_JIT_NO_STORES`. Write-log rollback remains the safety net per `rules/jit/store-compilation.md`.
