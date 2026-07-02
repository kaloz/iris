@echo off
REM Double-click or run from cmd: starts iris-gui in WSL (Indy launcher UI).
wsl -d Ubuntu bash -lc "cd ~/iris-wsl-build && ./wsl/run-iris-gui.sh"
