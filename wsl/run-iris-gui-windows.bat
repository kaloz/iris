@echo off
REM Native Windows iris-gui — proper mouse capture and fullscreen.
cd /d "%~dp0.."
if not exist "target\release\iris-gui.exe" (
  echo Building iris-gui first — this takes several minutes...
  cargo +nightly-x86_64-pc-windows-msvc build -p iris-gui --release
  if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
  )
)
start "" "target\release\iris-gui.exe"
