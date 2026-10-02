#!/usr/bin/env bash
# 默认使用全部 MoE 层；混合模型可通过命令行参数配置。
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m moe.train_moe --use_moe --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 --checkpoint_dir checkpoints_moe "$@"
