@echo off
REM One-liner for YouTube / recording: full-feature iris CLI, JIT env, premiere config.
REM Warm JIT profiles first — see wsl\README.md "JIT warm-up".
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
set IRIS_JIT=1
set IRIS_JIT_PROBE=500
set IRIS_JIT_PROBE_MIN=100
set IRIS_JIT_MAX_TIER=1
echo.
echo Premiere mode: iris CLI + OpenGL + JIT. Click window to grab mouse; Right Ctrl releases.
echo Monitor: telnet 127.0.0.1 8888  ^|  rex jit status  ^|  hal2 status
echo.
"target\release\iris.exe" --config irix-install\iris-windows.toml
