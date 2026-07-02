# Warm MIPS + REX3 JIT profiles before recording.
# Usage: .\wsl\warm-jit-profiles.ps1
# Requires: iris built (run-iris-premiere.bat builds if needed).

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host ""
Write-Host "=== IRIS JIT warm-up ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script launches premiere iris.exe with your config."
Write-Host "After IRIX boots:"
Write-Host "  1. Log in (telnet 127.0.0.1 2323 if needed)"
Write-Host "  2. Open a GL app (glxgears, 4Dwm, a demo)"
Write-Host "  3. Interact for 5-10 minutes"
Write-Host "  4. Quit iris cleanly (monitor: quit, or close window)"
Write-Host ""
Write-Host "Profiles saved under:"
Write-Host "  $env:USERPROFILE\.iris\jit-profile.bin"
Write-Host "  $env:USERPROFILE\.iris\rex-jit-profile.bin"
Write-Host ""
Write-Host "Monitor (optional): telnet 127.0.0.1 8888 -> rex jit status"
Write-Host ""

& (Join-Path $PSScriptRoot "run-iris-premiere.bat")
