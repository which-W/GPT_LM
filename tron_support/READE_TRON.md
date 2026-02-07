# Tron Support - DP+TP 混合并行训练框架

支持了 Data Parallel (DP) 和 Tensor Parallel (TP) 的混合并行训练。

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -e .

# 或手动安装
pip install torch transformers datasets safetensors wandb jinja2
```

### 2. 创建配置文件

```bash
# 示例：DP=2, TP=2，总共使用 4 个 GPU
python create_config.py \
    --out_dir tmp \
    --exp_name test_dp2_tp2 \
    --tp 2 \
    --dp 2 \
    --model_name HuggingFaceTB/SmolLM-360M-Instruct \
    --grad_acc_steps 4 \
    --mbs 4 \
    --seq_len 1024 \
    --use_wandb \
    --hf_token <YOUR_HF_TOKEN>
```

**参数说明：**
- `--tp`: 张量并行度（将模型权重切分到多个 GPU）
- `--dp`: 数据并行度（不同 GPU 处理不同数据）
- `--mbs`: Micro Batch Size（每个 GPU 每次处理的样本数）
- `--grad_acc_steps`: 梯度累积步数
- `--seq_len`: 序列长度

**Global Batch Size 计算：**
```
Global Batch Size = mbs × grad_acc_steps × dp
                  = 4 × 4 × 2 = 32 个样本
```

### 3. 本地训练

```bash
# 使用 4 个 GPU (DP=2, TP=2)
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node 4 \
    train.py --config tmp/test_dp2_tp2/config.json
```

### 4. 提交到 Slurm 集群

```bash
python submit_slurm_jobs.py \
    --inp_dir tmp/test_dp2_tp2 \
    --qos high \
    --hf_token <YOUR_HF_TOKEN>
```

## 并行策略详解

### GPU 分配示例（DP=2, TP=2）

```
┌─────────────────────────────────────┐
│          4个GPU的分配方式              │
├─────────────────────────────────────┤
│                                     │
│  Data Replica 0:                    │
│  ┌─────────┬─────────┐              │
│  │ TP GPU0 │ TP GPU1 │              │
│  └─────────┴─────────┘              │
│      ↑         ↑                    │
│      └─────────┘                    │
│   模型权重被切分                      │
│                                     │
│  Data Replica 1:                    │
│  ┌─────────┬─────────┐              │
│  │ TP GPU2 │ TP GPU3 │              │
│  └─────────┴─────────┘              │
│      ↑         ↑                    │
│      └─────────┘                    │
│   模型权重被切分                      │
│                                     │
└─────────────────────────────────────┘
```

### 张量并行 (TP)

**权重矩阵切分：**
```python
# 原始权重 W: [4096, 4096]
# TP=2 时切分为：
GPU0: W[:, 0:2048]    # 前一半列
GPU1: W[:, 2048:4096] # 后一半列
```

**前向传播：**
```python
# 输入 X: [batch, seq, 4096]
# GPU0 计算: Y0 = X @ W0  # [batch, seq, 2048]
# GPU1 计算: Y1 = X @ W1  # [batch, seq, 2048]
# 最终拼接: Y = [Y0, Y1]  # [batch, seq, 4096]
```

### 数据并行 (DP)

**数据切分：**
```python
# Global Batch = 32 samples
# DP=2 时切分为：
GPU组0: samples[0:16]   # 前 16 个样本
GPU组1: samples[16:32]  # 后 16 个样本
```

**梯度同步：**
```python
# 每个 DP 组独立计算梯度
# 梯度累积完成后，使用 All-Reduce 同步梯度
# 最终每个 GPU 都有相同的梯度用于更新参数
```

## 配置文件结构

生成的 `config.json` 包含以下部分：

```json
{
    "environment": {
        "HF_TOKEN": "your_token",
        "OMP_NUM_THREADS": "16",
        "TOKENIZERS_PARALLELISM": "false"
    },
    "model": {
        "name": "HuggingFaceTB/SmolLM-360M-Instruct",
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3
    },
    "distributed": {
        "tp_size": 2,
        "dp_size": 2
    },
    "training": {
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "seq_length": 1024,
        "learning_rate": 0.0003,
        "max_tokens": null,
        "total_train_steps": 10000
    },
    "dataset": {
        "name": "roneneldan/TinyStories",
        "num_workers": 4
    },
    "checkpoint": {
        "save_frequency": 1000,
        "save_dir": "./checkpoints"
    },
    "logging": {
        "use_wandb": true,
        "run_name": "test_dp2_tp2"
    }
}
```

## 性能优化建议

### 1. 选择合适的并行度

| GPU数量 | 推荐配置 | 说明 |
|---------|---------|------|
| 2 | DP=2, TP=1 | 小模型优先数据并行 |
| 4 | DP=2, TP=2 | 平衡配置 |
| 8 | DP=4, TP=2 | 或 DP=2, TP=4 |
| 16 | DP=8, TP=2 | 大规模数据并行 |

### 2. 调整 Batch Size

```python
# 显存不足时：
- 减小 micro_batch_size
- 增加 gradient_accumulation_steps

# 例如：
mbs=2, grad_acc=8  # 显存占用少，但速度慢
mbs=8, grad_acc=2  # 显存占用多，但速度快

# 保持 Global Batch Size 不变：
mbs × grad_acc × dp = constant
```

## 常见问题

### Q1: OOM (Out of Memory)

**解决方法：**
```bash
# 1. 减小 micro batch size
--mbs 2

# 2. 增加梯度累积
--grad_acc_steps 8

# 3. 减少序列长度
--seq_len 512

# 4. 增加 TP
--tp 4  # 分散模型权重
```

### Q2: 训练速度慢

**检查项：**
```bash
# 1. 确保设置了 CUDA_DEVICE_MAX_CONNECTIONS
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 2. 检查梯度累积步数
# 过大会导致更新频率低

# 3. 检查数据加载
# 增加 num_workers
```

### Q3: 损失不下降

**检查项：**
```bash
# 1. 学习率是否合适
--learning_rate 3e-4  # 默认值

# 2. 检查梯度同步
# 确保 DP > 1 时梯度正确同步

# 3. 检查数据集
# 确保数据质量和多样性
```

## 示例命令合集

```bash
# 1. 单机 4 卡训练 (DP=4)
python create_config.py --out_dir tmp --exp_name dp4 --dp 4 --tp 1 \
    --model_name HuggingFaceTB/SmolLM-360M-Instruct --mbs 8 --seq_len 1024

torchrun --nproc_per_node 4 train.py --config tmp/dp4/config.json

# 2. 单机 4 卡训练 (DP=2, TP=2)
python create_config.py --out_dir tmp --exp_name dp2_tp2 --dp 2 --tp 2 \
    --model_name HuggingFaceTB/SmolLM-360M-Instruct --mbs 4 --seq_len 1024

torchrun --nproc_per_node 4 train.py --config tmp/dp2_tp2/config.json

# 3. 单机 8 卡训练 (DP=4, TP=2)
python create_config.py --out_dir tmp --exp_name dp4_tp2 --dp 4 --tp 2 \
    --model_name meta-llama/Llama-2-7b-hf --mbs 2 --seq_len 2048 \
    --grad_acc_steps 8 --hf_token <TOKEN>

torchrun --nproc_per_node 8 train.py --config tmp/dp4_tp2/config.json

# 4. 提交到 Slurm
python submit_slurm_jobs.py --inp_dir tmp/dp4_tp2 --qos high \
    --hf_token <TOKEN>
```


