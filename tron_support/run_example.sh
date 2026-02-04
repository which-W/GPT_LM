#!/bin/bash
# 完整的训练示例脚本

set -e  # 遇到错误立即退出

echo "======================================"
echo "Tron Support DP+TP 训练示例"
echo "======================================"

# 配置参数
EXP_NAME="example_dp2_tp2"
OUT_DIR="tmp"
TP_SIZE=2
DP_SIZE=2
MODEL_NAME="HuggingFaceTB/SmolLM-360M-Instruct"
MBS=4
GRAD_ACC=4
SEQ_LEN=1024
HF_TOKEN=${HF_TOKEN:-""}

# 检查 HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "警告: HF_TOKEN 环境变量未设置"
    echo "请设置: export HF_TOKEN='your_token'"
    echo "或者在命令中传递: --hf_token 'your_token'"
fi

# 计算总 GPU 数
TOTAL_GPUS=$((TP_SIZE * DP_SIZE))
echo ""
echo "配置信息:"
echo "  实验名称: $EXP_NAME"
echo "  模型: $MODEL_NAME"
echo "  TP Size: $TP_SIZE"
echo "  DP Size: $DP_SIZE"
echo "  总 GPU 数: $TOTAL_GPUS"
echo "  Micro Batch Size: $MBS"
echo "  Gradient Accumulation: $GRAD_ACC"
echo "  Sequence Length: $SEQ_LEN"
echo "  Global Batch Size: $((MBS * GRAD_ACC * DP_SIZE))"
echo ""

# 步骤 1: 创建配置文件
echo "步骤 1/3: 创建配置文件..."
python create_config.py \
    --out_dir "$OUT_DIR" \
    --exp_name "$EXP_NAME" \
    --tp $TP_SIZE \
    --dp $DP_SIZE \
    --model_name "$MODEL_NAME" \
    --mbs $MBS \
    --grad_acc_steps $GRAD_ACC \
    --seq_len $SEQ_LEN \
    --use_wandb \
    --hf_token "$HF_TOKEN"

echo ""
echo "配置文件已创建: $OUT_DIR/$EXP_NAME/config.json"
echo ""

# 步骤 2: 显示配置内容
echo "步骤 2/3: 配置文件内容预览..."
echo "----------------------------------------"
cat "$OUT_DIR/$EXP_NAME/config.json" | head -20
echo "..."
echo "----------------------------------------"
echo ""

# 步骤 3: 提示如何启动训练
echo "步骤 3/3: 启动训练"
echo ""
echo "本地训练命令:"
echo "----------------------------------------"
echo "CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \\"
echo "    --nproc_per_node $TOTAL_GPUS \\"
echo "    train.py --config $OUT_DIR/$EXP_NAME/config.json"
echo "----------------------------------------"
echo ""

# 询问是否立即启动训练
read -p "是否立即启动训练? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "启动训练..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
        --nproc_per_node $TOTAL_GPUS \
        train.py --config "$OUT_DIR/$EXP_NAME/config.json"
else
    echo "跳过训练。使用上述命令手动启动。"
fi

echo ""
echo "======================================"
echo "完成!"
echo "======================================"
