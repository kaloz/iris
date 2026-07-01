@echo off
REM Stable premiere: Loads-tier JIT only (no Full), fresh profile path.
REM Use when premiere.bat panics with TLBMISS KERNEL FAULT at 0xff800000.
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
set IRIS_JIT=1
set IRIS_JIT_PROBE=500
set IRIS_JIT_PROBE_MIN=100
set IRIS_JIT_MAX_TIER=1
set IRIS_JIT_PROFILE=jit-profile-safe.bin
echo.
echo Safe premiere: JIT max_tier=Loads (1), isolated profile (no Full-tier replay).
echo If this boots but premiere.bat does not, rename/delete jit-profile.bin in this folder.
echo Monitor: telnet 127.0.0.1 8888
echo.
"target\release\iris.exe" --config irix-install\iris-windows-safe.toml
