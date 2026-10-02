#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m train --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 "$@"
