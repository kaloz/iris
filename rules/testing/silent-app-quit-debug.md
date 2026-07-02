# Silent IRIX app quit — debug capture checklist

**Keywords:** userspace, app quit, JIT verify, monitor, SYSLOG, 512 MB, IRIX 6.5
**Category:** testing

When IRIX apps close with no error dialog, collect evidence before guessing at fixes.

## Quick triage

1. **RAM layout** — IRIX 6.5: use `[128,128,64,64]` (384 MB), not 512 MB (`iris-windows-384.toml`). Verify `hinv -t memory` matches config after cold start.
2. **JIT A/B** — `wsl\run-iris-premiere-nojit.bat` vs `wsl\run-iris-premiere.bat` on the same apps.
3. **Disk** — `cow status` in monitor; reset overlay if dirty sectors grew after panics.

## Capture bundle (send to developer)

| Artifact | How |
|----------|-----|
| TOML `banks` | `irix-install/iris-windows.toml` |
| Guest RAM | `hinv -t memory` in IRIX shell |
| Host log | `wsl\capture-app-crash.ps1` → `premiere-debug.log` |
| Monitor | telnet 8888: `stop` → `status` / `regs` / `bt` / `dt 80` |
| Guest | `ps -ef`, `tail -50 /var/adm/SYSLOG` after quit |
| JIT A/B | stable with no-JIT? yes/no |

## Monitor commands at quit moment

```text
stop
status
regs
bt
dt 80
exception all on
cow status
mc status
```

Developer build adds `debug on`, `log mips mask insn`, `dt file crash-trace.txt 1048576`.

## JIT stderr signatures

- `JIT VERIFY FAIL` / `REAL CODEGEN MISMATCH` — codegen bug at listed PC
- Rising `rollbacks` / `demotions` in `JIT: ... ⟲` lines — speculative path fighting bad blocks

See also `rules/jit/speculative-safety-net.md`, `rules/testing/disk-image-hygiene.md`.
