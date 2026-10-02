#!/usr/bin/env bash
# 用进程启动器创建训练进程，训练程序不会再重复创建子进程。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-1}"
"$PYTHON" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" -m distributed.train_distribute_moe_ddp --distributed --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 "$@"
