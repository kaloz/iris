@echo off
REM A/B test: interpreter-only (no MIPS JIT). If apps stop quitting, suspect JIT.
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
set IRIS_JIT=0
echo.
echo Premiere NO-JIT: same config as premiere.bat but IRIS_JIT=0 (interpreter only).
echo Launch the same apps for ~10 min and compare stability vs run-iris-premiere.bat
echo.
"target\release\iris.exe" --config irix-install\iris-windows.toml
