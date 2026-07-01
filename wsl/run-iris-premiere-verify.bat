@echo off
REM JIT verify mode: re-runs each compiled block through interpreter (slow, diagnostic).
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
set IRIS_JIT=1
set IRIS_JIT_PROBE=500
set IRIS_JIT_PROBE_MIN=100
set IRIS_JIT_MAX_TIER=1
set IRIS_JIT_VERIFY=1
echo.
echo Premiere VERIFY: JIT on + IRIS_JIT_VERIFY=1. Watch for "JIT VERIFY FAIL" in log.
echo Use wsl\capture-app-crash.ps1 to tee stderr to premiere-debug.log
echo.
"target\release\iris.exe" --config irix-install\iris-windows.toml
