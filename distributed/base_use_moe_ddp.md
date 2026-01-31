# MoE分布式训练完整实现

这是一个将Mixture of Experts (MoE)与PyTorch分布式数据并行(DDP)相结合的完整训练系统。

## 📁 文件说明

### 核心训练脚本
- **`train_distributed_moe_ddp.py`** - MoE分布式训练主脚本
  - 结合专家并行和数据并行
  - 手动梯度同步机制
  - 支持混合MoE模型
  - WandB集成

### 启动工具
- **`run_moe_ddp.py`** - 便捷启动工具
  - 预设配置(small/medium/large)
  - 自动检测GPU
  - 简化命令行参数
  
- **`train_distributed_moe_ddp.sh`** - Shell启动脚本
  - 包含多个训练示例
  - 快速复制粘贴使用

## 🚀 快速开始

### 1. 最简单的方式（使用预设配置）

```bash
# 使用便捷启动工具
python run_moe_train.py \
    --preset small \
    --num_gpus 4 \
    --train_data data/train.bin \
    --val_data data/val.bin \
    --wandb
```

### 2. 使用torchrun（更灵活）

```bash
torchrun --nproc_per_node=4 train_moe_distributed.py \
    --distributed \
    --train_data_path data/train.bin \
    --valid_data_path data/val.bin \
    --n_experts 8 \
    --top_k 2 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --total_steps 50000 \
    --dtype bfloat16 \
    --use_wandb
```

### 3. 使用shell脚本

```bash
# 编辑 train_moe_distributed.sh 中的配置
bash train_moe_distributed.sh
```
## ⚠️ 重要说明

### 为什么不使用标准DDP？

MoE模型的专家已经分布在多个GPU上，直接使用DDP会导致：
1. 设备冲突（DDP期望所有参数在同一GPU）
2. 梯度同步错误（每个rank只有部分专家）

**解决方案**：手动梯度同步
```python
# 在每个训练步后
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
```
## 🛠️ 故障排查

### OOM (显存不足)
```bash
# 减小批次大小
--batch_size 2

# 使用梯度累积
--gradient_accumulation_steps 8

# 减少专家数
--n_experts 4

# 使用混合MoE
--use_hybrid_moe
```

### 训练很慢
```bash
# 减少Top-K
--top_k 2  # 从4改成2

# 增加批次大小
--batch_size 8

# 检查专家负载均衡
# 查看日志中的 Aux Loss
```

### 损失不下降
```bash
# 降低辅助损失权重
--moe_aux_loss_weight 0.001

# 降低学习率
--max_lr 3e-4

# 增加专家数
--n_experts 16
```

## 🔬 实验追踪

使用WandB监控训练：
```bash
--use_wandb \
--wandb_project "my-moe-project" \
--wandb_run_name "4gpu-8experts"
```

关键指标：
- `train/loss`: 主损失（语言模型）
- `train/aux_loss`: 负载均衡损失
- `train/total_loss`: 总损失

## 💡 最佳实践

1. **从小规模开始**
   ```bash
   # 先用小配置测试
   python run_moe_train.py --preset small --num_gpus 2
   ```

2. **监控辅助损失**
   - Aux Loss < 0.01: 负载均衡良好
   - Aux Loss > 0.1: 需要调整权重

3. **定期保存检查点**
   ```bash
   --save_interval 5000  # 每5000步保存
   ```

4. **使用bfloat16**
   - A100/H100上可加速2倍
   - 更稳定than float16

## 🎯 适用场景

✅ **适合使用MoE的情况**:
- 需要大模型容量但计算受限
- 有多GPU资源可用
- 数据量充足（需要大批次）
- 关注参数效率

❌ **不适合的情况**:
- 单GPU训练
- 计算资源充足但显存不足
- 小规模数据集
- 需要最快推理速度

**开始训练**：
```bash
python run_moe_train.py \
    --preset small \
    --num_gpus 4 \
    --train_data your_train.bin \
    --val_data your_val.bin \
    --wandb
```

祝训练顺利！🚀