@echo off
REM iris-gui with premiere embedded core (lightning + idle-pause) + JIT + optional GL capture.
REM Warm JIT profiles first — see wsl\README.md "JIT warm-up".
cd /d "%~dp0.."
set IRIS_GUI_FEATURES=premiere
set IRIS_GUI_STAMP=target\release\.iris-gui-premiere.stamp
set IRIS_GUI_WANT=premiere

call "%~dp0ensure-build.bat" gui
if errorlevel 1 exit /b 1

set IRIS_JIT=1
set IRIS_JIT_PROBE=500
set IRIS_JIT_PROBE_MIN=100
set IRIS_JIT_MAX_TIER=1
set IRIS_GUI_GL=1

echo.
echo Premiere GUI: in-process iris + JIT + OpenGL capture (IRIS_GUI_GL=1).
echo For max 3D recording use run-iris-premiere.bat (native CLI OpenGL).
echo.
start "" "target\release\iris-gui.exe"
