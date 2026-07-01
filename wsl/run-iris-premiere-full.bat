@echo off
REM Experimental: Full-tier JIT (max_tier=2). Slower on hot kernel loops and may
REM TLBMISS-panic on Windows — use run-iris-premiere.bat (tier 1) for recording.
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
set IRIS_JIT=1
set IRIS_JIT_PROBE=500
set IRIS_JIT_PROBE_MIN=100
set IRIS_JIT_MAX_TIER=2
echo.
echo Premiere FULL tier (experimental): may be slow or panic. Prefer run-iris-premiere.bat.
echo.
"target\release\iris.exe" --config irix-install\iris-windows.toml
