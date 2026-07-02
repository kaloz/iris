# Capture host stderr + print checklist when IRIX apps quit silently.
# Usage:
#   wsl\capture-app-crash.ps1                    # JIT on, default config
#   wsl\capture-app-crash.ps1 -NoJit             # interpreter A/B
#   wsl\capture-app-crash.ps1 -Verify            # IRIS_JIT_VERIFY=1
#   wsl\capture-app-crash.ps1 -Config irix-install\iris-windows-384.toml
param(
    [string]$Config = "irix-install\iris-windows.toml",
    [switch]$NoJit,
    [switch]$Verify,
    [string]$LogFile = "premiere-debug.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== IRIS silent-quit capture ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Before reproducing, note your TOML banks line:" -ForegroundColor Yellow
if (Test-Path $Config) {
    Select-String -Path $Config -Pattern "^banks\s*=" | ForEach-Object { Write-Host "  $($_.Line)" }
} else {
    Write-Host "  (config not found: $Config)" -ForegroundColor Red
}
Write-Host ""
Write-Host "Second terminal:  telnet 127.0.0.1 8888" -ForegroundColor Yellow
Write-Host "When an app quits, run in monitor:" -ForegroundColor Yellow
Write-Host "  stop`n  status`n  regs`n  bt`n  dt 80`n  cow status`n  mc status"
Write-Host ""
Write-Host "In IRIX shell after quit:" -ForegroundColor Yellow
Write-Host "  hinv -t memory"
Write-Host "  ps -ef | grep -i <appname>"
Write-Host "  tail -50 /var/adm/SYSLOG"
Write-Host ""
Write-Host "Logging to: $LogFile" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop iris when done." -ForegroundColor Gray
Write-Host ""

& wsl\ensure-build.bat cli
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:IRIS_JIT_PROBE = "500"
$env:IRIS_JIT_PROBE_MIN = "100"
$env:IRIS_JIT_MAX_TIER = "1"

if ($NoJit) {
    $env:IRIS_JIT = "0"
    Remove-Item Env:IRIS_JIT_VERIFY -ErrorAction SilentlyContinue
    Write-Host "Mode: NO JIT (interpreter only)" -ForegroundColor Magenta
} else {
    $env:IRIS_JIT = "1"
    if ($Verify) {
        $env:IRIS_JIT_VERIFY = "1"
        Write-Host "Mode: JIT + VERIFY (slow, catches codegen mismatches)" -ForegroundColor Magenta
    } else {
        Remove-Item Env:IRIS_JIT_VERIFY -ErrorAction SilentlyContinue
        Write-Host "Mode: JIT (premiere default)" -ForegroundColor Magenta
    }
}

$logPath = Join-Path $root $LogFile
$header = @"
=== IRIS capture session $(Get-Date -Format o) ===
Config: $Config
NoJit: $NoJit  Verify: $Verify
Banks: $(if (Test-Path $Config) { (Select-String -Path $Config -Pattern '^banks\s*=').Line } else { 'n/a' })
===
"@

$header | Out-File -FilePath $logPath -Encoding utf8

& "target\release\iris.exe" --config $Config 2>&1 | Tee-Object -FilePath $logPath -Append
