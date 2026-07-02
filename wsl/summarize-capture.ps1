# Summarize premiere-debug.log after a silent-quit session.
param(
    [string]$LogFile = "premiere-debug.log"
)

$root = Split-Path -Parent $PSScriptRoot
$path = Join-Path $root $LogFile

if (-not (Test-Path $path)) {
    Write-Error "Missing $path — run wsl\capture-app-crash.ps1 first."
}

Write-Host "=== Log summary: $LogFile ===" -ForegroundColor Cyan
Write-Host ""

$verify = Select-String -Path $path -Pattern "JIT VERIFY FAIL|REAL CODEGEN MISMATCH" -SimpleMatch:$false
if ($verify) {
    Write-Host "JIT VERIFY failures ($($verify.Count)):" -ForegroundColor Red
    $verify | Select-Object -Last 10 | ForEach-Object { Write-Host "  $($_.Line)" }
} else {
    Write-Host "No JIT VERIFY FAIL lines found." -ForegroundColor Green
}

Write-Host ""
$jitLines = Select-String -Path $path -Pattern "^JIT: \d+ total" | Select-Object -Last 5
if ($jitLines) {
    Write-Host "Last JIT stats lines:" -ForegroundColor Yellow
    $jitLines | ForEach-Object { Write-Host "  $($_.Line)" }
}

Write-Host ""
Write-Host "Share this file + monitor stop/status/bt/dt + hinv -t memory if still broken." -ForegroundColor Gray
