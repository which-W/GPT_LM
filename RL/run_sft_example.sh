#!/bin/bash
# SFT训练完整流程示例脚本

set -e  # 遇到错误时退出

# 配置参数
DATA_DIR="./data"
CHECKPOINT_DIR="./checkpoints"
SFT_CHECKPOINT_DIR="./sft_checkpoints"
PRETRAIN_MODEL="$CHECKPOINT_DIR/checkpoint_final.pt"
TOKENIZER_PATH="./tokenizer.py"

# 模型参数
D_MODEL=512
N_HEAD=8
N_LAYER=6
D_FF=2048
VOCAB_SIZE=30000
MAX_SEQ_LEN=512

# 训练参数
BATCH_SIZE=4
NUM_EPOCHS=3
MAX_LR=5e-5
MIN_LR=5e-6

echo ""
echo "步骤1: 创建数据目录"
mkdir -p $DATA_DIR
mkdir -p $SFT_CHECKPOINT_DIR

echo ""
echo "步骤2: 生成示例SFT数据"
python prepare_sft_data.py \
    --mode sample \
    --output $DATA_DIR/sft_sample.jsonl \
    --num_samples 500

echo ""
echo "步骤3: 分割训练集和验证集"
python prepare_sft_data.py \
    --mode split \
    --input $DATA_DIR/sft_sample.jsonl \
    --train_output $DATA_DIR/sft_train.jsonl \
    --val_output $DATA_DIR/sft_val.jsonl \
    --val_ratio 0.1

echo ""
echo "步骤4: 开始SFT训练"
python sft_train.py \
    --train_data_path $DATA_DIR/sft_train.jsonl \
    --valid_data_path $DATA_DIR/sft_val.jsonl \
    --pretrain_checkpoint $PRETRAIN_MODEL \
    --tokenizer_path $TOKENIZER_PATH \
    --checkpoint_dir $SFT_CHECKPOINT_DIR \
    --d_model $D_MODEL \
    --n_head $N_HEAD \
    --n_layer $N_LAYER \
    --d_ff $D_FF \
    --vocab_size $VOCAB_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --warmup_ratio 0.1 \
    --prompt_template default \
    --log_interval 50 \
    --eval_interval 200 \
    --save_interval 500

echo ""
echo "步骤5: 测试推理"
echo "运行单次推理测试..."
python sft_inference.py \
    --mode single \
    --checkpoint_path $SFT_CHECKPOINT_DIR/sft_checkpoint_final.pt \
    --tokenizer_path $TOKENIZER_PATH \
    --instruction "解释什么是深度学习" \
    --d_model $D_MODEL \
    --n_head $N_HEAD \
    --n_layer $N_LAYER \
    --d_ff $D_FF \
    --vocab_size $VOCAB_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 256

echo ""
echo ""
echo "模型保存在: $SFT_CHECKPOINT_DIR/sft_checkpoint_final.pt"
echo ""
echo "后续步骤:"
echo "1. 运行交互式对话:"
echo "   python sft_inference.py --mode interactive --checkpoint_path $SFT_CHECKPOINT_DIR/sft_checkpoint_final.pt --tokenizer_path $TOKENIZER_PATH ..."
echo ""
echo "2. 批量推理:"
echo "   python sft_inference.py --mode batch --checkpoint_path $SFT_CHECKPOINT_DIR/sft_checkpoint_final.pt --test_file test.jsonl --output_file results.jsonl ..."
