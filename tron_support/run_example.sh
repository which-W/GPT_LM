#!/usr/bin/env bash
# 在项目根目录生成 Tron 配置，可通过环境变量选择并行规模。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
EXP_NAME="${EXP_NAME:-example_dp1_tp1}"
OUT_DIR="${OUT_DIR:-tmp}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM-360M-Instruct}"
TOTAL_GPUS=$((TP_SIZE * DP_SIZE))

# 公开模型允许省略令牌；受限模型通过 HF_TOKEN 提供访问凭据。
"$PYTHON" -m tron_support.create_config --out_dir "$OUT_DIR" --exp_name "$EXP_NAME" --tp "$TP_SIZE" --dp "$DP_SIZE" --model_name "$MODEL_NAME" --mbs 4 --grad_acc_steps 4 --seq_len 1024 --hf_token "${HF_TOKEN:-}"
echo "配置已保存到 $OUT_DIR/$EXP_NAME/config.json"
echo "启动命令：$PYTHON -m utils.torchrun --nproc_per_node $TOTAL_GPUS -m tron_support.train --config $OUT_DIR/$EXP_NAME/config.json"

# 设置 RUN_TRAINING=1 时立即启动，否则仅生成配置。
if [ "${RUN_TRAINING:-0}" = "1" ]; then
    CUDA_DEVICE_MAX_CONNECTIONS=1 "$PYTHON" -m utils.torchrun --nproc_per_node "$TOTAL_GPUS" -m tron_support.train --config "$OUT_DIR/$EXP_NAME/config.json"
fi
