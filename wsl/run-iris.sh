#!/usr/bin/env bash
# Launch IRIS CLI with the installed IRIX disk (WSLg window).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WSL_ROOT="${IRIS_WSL_ROOT:-$HOME/iris-wsl-build}"
cd "$WSL_ROOT"

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

CONFIG="${IRIS_CONFIG:-irix-install/iris-wsl.toml}"
exec ./target/release/iris --config "$CONFIG" "$@"
