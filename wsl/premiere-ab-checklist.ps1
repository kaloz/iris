# A/B checklist for YouTube premiere recording (Windows native IRIS).
# Usage: .\wsl\premiere-ab-checklist.ps1

Write-Host ""
Write-Host "=== IRIS Premiere A/B Checklist ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "TAKE A (slow baseline — show interpreter / GUI CPU path)" -ForegroundColor Yellow
Write-Host "  Build:  cargo build -p iris-gui --release   (no premiere feature)"
Write-Host "  Run:    wsl\run-iris-gui-windows.bat"
Write-Host "  JIT:    Debug tab -> IRIS_JIT off"
Write-Host "  Expect: Low MIPS in status bar; high host CPU at idle desktop"
Write-Host ""

Write-Host "TAKE B (fast — premiere CLI + warmed profiles)" -ForegroundColor Green
Write-Host "  Warm:   wsl\warm-jit-profiles.ps1  (once per session)"
Write-Host "  Run:    wsl\run-iris-premiere.bat"
Write-Host "  Config: irix-install\iris-windows.toml (scale=1, export from GUI)"
Write-Host "  Expect: 80-150+ MIPS with JIT; lower idle CPU with idle-pause"
Write-Host ""

Write-Host "TAKE B-alt (in-process GUI with premiere core + GL capture)" -ForegroundColor Green
Write-Host "  Run:    wsl\run-iris-gui-premiere.bat"
Write-Host "  Sets:   IRIS_JIT=1, IRIS_GUI_GL=1, lightning+idle-pause build"
Write-Host "  Note:   Still slower than CLI OpenGL for heavy 3D"
Write-Host ""

Write-Host "Verify (telnet 127.0.0.1 8888):" -ForegroundColor Cyan
Write-Host "  perf snapshot   (Phase 3 aggregate: JIT, REX3, HAL2, affinity)"
Write-Host "  rex jit status"
Write-Host "  hal2 status   (cpal underruns should stay low; codec A uses dedicated pump thread)"
Write-Host ""
Write-Host "Benchmark helper: wsl\profile.ps1"
Write-Host "CI smoke:         wsl\smoke-premiere.ps1"
Write-Host ""

Write-Host "GUI workflow:" -ForegroundColor Cyan
Write-Host "  File -> Prepare for premiere...  (exports TOML + JIT defaults)"
Write-Host ""
