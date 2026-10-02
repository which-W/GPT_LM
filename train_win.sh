#!/usr/bin/env bash
# 供 Windows 上的 Git Bash 使用；PowerShell 用户使用 train_win.ps1。
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
"$PYTHON" -m train --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 "$@"
