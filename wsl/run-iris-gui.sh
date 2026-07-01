#!/usr/bin/env bash
# Launch iris-gui (egui front-end). First run: File → Import iris.toml →
#   ~/iris-wsl-build/irix-install/iris-wsl.toml
set -euo pipefail
WSL_ROOT="${IRIS_WSL_ROOT:-$HOME/iris-wsl-build}"
cd "$WSL_ROOT"

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

exec ./target/release/iris-gui "$@"
