#!/bin/bash

# MoE Transformer 训练示例脚本

# 示例1: 训练标准MoE模型 (所有层都使用MoE)，推荐这个使用moe全层
./.venv/bin/python3 -m moe_model.train_moe \
    --use_moe \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --d_ff 2048 \
    --vocab_size 30000 \
    --max_seq_len 512 \
    --n_experts 8 \
    --top_k 2 \
    --use_moe_aux_loss \
    --moe_aux_loss_weight 0.01 \
    --batch_size 8 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 2000 \
    --total_steps 10000 \
    --train_data_path ./data/TinyStories-train.bin \
    --valid_data_path ./data/TinyStories-valid.bin \
    --checkpoint_dir ./checkpoints_moe \
    --save_interval 1000 \
    --log_interval 100 \
    --eval_interval 500 \
    --use_wandb \
    --wandb_project "moe-transformer" \
    --wandb_run_name "moe-8experts-top2"



# 示例2: 训练混合MoE模型 (部分层使用MoE)
# ./.venv/bin/python3 train_moe.py \
#     --use_hybrid_moe \
#     --moe_layer_indices "0,2,4" \
#     --d_model 512 \
#     --n_head 8 \
#     --n_layer 6 \
#     --d_ff 2048 \
#     --vocab_size 30000 \
#     --max_seq_len 512 \
#     --n_experts 8 \
#     --top_k 2 \
#     --use_moe_aux_loss \
#     --moe_aux_loss_weight 0.01 \
#     --batch_size 8 \
#     --max_lr 3e-4 \
#     --min_lr 3e-5 \
#     --warmup_steps 2000 \
#     --total_steps 10000 \
#     --train_data_path ./data/train.bin \
#     --valid_data_path ./data/val.bin \
#     --checkpoint_dir ./checkpoints_hybrid_moe \
#     --save_interval 1000 \
#     --log_interval 100 \
#     --eval_interval 500



# # 示例3: 多GPU训练
# ./.venv/bin/python3 train_moe.py \
#     --use_moe \
#     --device_ids "0,1,2,3" \
#     --main_device 0 \
#     --d_model 768 \
#     --n_head 12 \
#     --n_layer 12 \
#     --d_ff 3072 \
#     --vocab_size 50000 \
#     --max_seq_len 1024 \
#     --n_experts 16 \
#     --top_k 2 \
#     --use_moe_aux_loss \
#     --moe_aux_loss_weight 0.01 \
#     --batch_size 16 \
#     --max_lr 1e-4 \
#     --min_lr 1e-5 \
#     --warmup_steps 4000 \
#     --total_steps 50000 \
#     --train_data_path ./data/train.bin \
#     --valid_data_path ./data/val.bin \
#     --checkpoint_dir ./checkpoints_moe_large \
#     --save_interval 2000 \
#     --log_interval 100 \
#     --eval_interval 1000 \
#     --dtype bfloat16 \
#     --use_wandb \
#     --wandb_project "moe-transformer" \
#     --wandb_run_name "moe-16experts-multi-gpu"


# # 示例4: 从检查点恢复训练
# ./.venv/bin/python3 train_moe.py \
#     --use_moe \
#     --resume_from ./checkpoints_moe/checkpoint_step_5000.pt \
#     --d_model 512 \
#     --n_head 8 \
#     --n_layer 6 \
#     --d_ff 2048 \
#     --vocab_size 30000 \
#     --max_seq_len 512 \
#     --n_experts 8 \
#     --top_k 2 \
#     --use_moe_aux_loss \
#     --moe_aux_loss_weight 0.01 \
#     --batch_size 8 \
#     --max_lr 3e-4 \
#     --min_lr 3e-5 \
#     --warmup_steps 2000 \
#     --total_steps 10000 \
#     --train_data_path ./data/train.bin \
#     --valid_data_path ./data/val.bin \
#     --checkpoint_dir ./checkpoints_moe \
#     --save_interval 1000
