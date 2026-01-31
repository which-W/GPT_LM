#!/bin/bash
# MoE分布式训练启动脚本

# ==========================================
# 示例1: 4卡全MoE训练
# ==========================================
torchrun --nproc_per_node=4 train_moe_distributed.py \
    --distributed \
    --train_data_path data/train.bin \ 
    --valid_data_path data/val.bin \
    --d_model 512 \
    --n_head 8 \
    --n_layer 12 \
    --d_ff 2048 \
    --vocab_size 30000 \
    --max_seq_len 512 \
    --n_experts 8 \
    --top_k 2 \
    --use_moe_aux_loss \
    --moe_aux_loss_weight 0.01 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 2000 \
    --total_steps 50000 \
    --dtype bfloat16 \
    --max_grad_norm 1.0 \
    --save_interval 5000 \
    --eval_interval 1000 \
    --log_interval 100 \
    --use_wandb \
    --wandb_project "moe-transformer" \
    --wandb_run_name "4gpu-8experts-top2" \
    --checkpoint_dir "./checkpoints"


# ==========================================
# 示例2: 8卡混合MoE训练
# 只在某些层使用MoE
# ==========================================
# torchrun --nproc_per_node=8 train_moe_distributed.py \
#     --distributed \
#     --use_hybrid_moe \
#     --moe_layer_indices 2 5 8 11 \
#     --train_data_path data/train.bin \
#     --valid_data_path data/val.bin \
#     --d_model 1024 \
#     --n_head 16 \
#     --n_layer 24 \
#     --d_ff 4096 \
#     --vocab_size 50000 \
#     --max_seq_len 2048 \
#     --n_experts 16 \
#     --top_k 2 \
#     --use_moe_aux_loss \
#     --batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --max_lr 3e-4 \
#     --min_lr 3e-5 \
#     --warmup_steps 5000 \
#     --total_steps 100000 \
#     --dtype bfloat16 \
#     --use_wandb \
#     --wandb_project "moe-transformer-large" \
#     --wandb_run_name "8gpu-hybrid-moe"