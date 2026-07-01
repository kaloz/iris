@echo off
REM Double-click or run from cmd: starts IRIS CLI in WSL with your IRIX disk.
wsl -d Ubuntu bash -lc "cd ~/iris-wsl-build && ./wsl/run-iris.sh"
