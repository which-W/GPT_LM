#!/usr/bin/env bash
# 默认复用训练模型评估，适合单显卡；独立 vLLM 后端可通过额外参数启用。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m SFT.sft_train --model_id Qwen/Qwen2.5-Math-1.5B --train_data_path data/gsm8k-train.jsonl --val_data_path data/gsm8k-val.jsonl --prompt_path prompts/r1_zero.prompt --batch_size 16 --micro_batch_size 2 --max_steps 200 --eval_backend transformers "$@"
