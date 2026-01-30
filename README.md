# GPT Language Model Implementation

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**一个模块化的深度学习项目，专注于Transformer语言模型的实现与优化**


</div>

## 📖 项目简介

本项目是一个完整的深度学习教学与研究项目，从零实现了Transformer语言模型的各个组件。项目采用模块化设计，不仅涵盖了标准的Transformer架构，还实现了MoE（Mixture of Experts）、分布式训练、vLLM推理加速等前沿技术。

### 🎯 项目目标

- **教学导向**: 通过模块化实现帮助理解Transformer的每个组件
- **研究友好**: 支持多种消融实验和架构变体
- **高性能**: 支持分布式训练和高效推理
- **可扩展**: 易于添加新特性和优化策略

## ✨ 功能特性

### 🧠 核心模型组件
- **完整Transformer实现**: 包含所有核心组件的自定义实现
- **注意力机制**: 多头注意力 + RoPE位置编码
- **前馈网络**: SwiGLU/SiLU激活函数
- **归一化层**: RMSNorm（可切换）
- **自定义优化器**: AdamW优化器实现
- **学习率调度**: Cosine Annealing with Warmup

### 🚀 高级特性
- **MoE支持**: 混合专家模型（类似DeepSeek V2）
- **分布式训练**: 多GPU DDP训练支持
- **KV Cache**: 推理加速优化
- **vLLM兼容**: 支持PagedAttention推理
- **梯度累积**: 大批次训练支持
- **混合精度**: FP16/BF16训练

### 🔬 实验功能
- **消融实验**: 可配置的组件开关
- **架构变体**: Pre-norm/Post-norm切换
- **可视化监控**: WandB集成
- **检查点管理**: 完整的训练状态保存/恢复

## 🛠️ 安装指南

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (用于GPU训练)
- Git

### 克隆项目

```bash
git clone https://github.com/yourusername/GPT_LM.git
cd GPT_LM
```

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装PyTorch（根据您的CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy einops wandb tensorboard
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 🚀 快速开始

### 1. 数据准备

```bash
# 使用TinyStories数据集（示例）
python dataset_process.py --input_path data/TinyStories-train.txt --output_path data/TinyStories-train.bin
python dataset_process.py --input_path data/TinyStories-valid.txt --output_path data/TinyStories-valid.bin
```

### 2. 单GPU训练

```bash
# 基础训练
python train.py \
    --train_data_path data/TinyStories-train.bin \
    --valid_data_path data/TinyStories-valid.bin \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --batch_size 16 \
    --max_lr 3e-4 \
    --total_steps 10000 \
    --use_wandb
```

### 3. 分布式训练

```bash
# 多GPU训练
python distributed/train_distributed.py \
    --train_data_path data/TinyStories-train.bin \
    --valid_data_path data/TinyStories-valid.bin \
    --distributed \
    --world_size 4 \
    --batch_size 8 \
    --total_steps 50000 \
    --use_wandb
```

### 4. 模型推理

```bash
# 基础推理
python inference/inference.py \
    --model_path checkpoints/checkpoint_final.pt \
    --prompt "Once upon a time" \
    --max_length 100
```

## 📁 项目结构

```
GPT_LM/
├── 📂 核心组件
│   ├── transformer.py           # 主模型类
│   ├── transformer_block.py     # Transformer块
│   ├── attention.py            # 注意力机制
│   ├── emb.py                  # 嵌入层
│   ├── rmsnorm.py              # RMSNorm归一化
│   ├── rope.py                 # RoPE位置编码
│   ├── swiGLU.py               # SwiGLU激活函数
│   └── softmax.py              # 稳定Softmax
│
├── 📂 训练组件
│   ├── train.py                # 单GPU训练脚本
│   ├── adamw.py                # 自定义AdamW优化器
│   ├── shedule.py              # 学习率调度器
│   ├── cross_entropy.py        # 交叉熵损失
│   ├── get_batch.py            # 批次数据获取
│   ├── clip_gradient_noem.py   # 梯度裁剪
│   └── checpoint_use.py        # 检查点管理
│
├── 📂 高级特性
│   ├── 📂 moe_model/           # MoE实现
│   │   ├── moe_transformer.py
│   │   ├── moe_transformer_block.py
│   │   ├── moe_layer.py
│   │   ├── moe_experts.py
│   │   └── moe_router.py
│   │
│   ├── 📂 vllm_support/        # vLLM推理支持
│   │   ├── vllm_transformer.py
│   │   ├── vllm_transformer_block.py
│   │   ├── vllm_attention.py
│   │   └── 📂 engine/
│   │       ├── llm_engine.py
│   │       ├── scheduler.py
│   │       ├── block_manager.py
│   │       └── sequence.py
│   │
│   └── 📂 distributed/         # 分布式训练
│       ├── train_distributed.py
│       ├── run_train.py
│       └── base_use_zh.md
│
├── 📂 推理组件
│   ├── inference/
│   │   ├── inference.py
│   │   └── sd_inference.py
│   └── tokenizer.py
│
├── 📂 数据处理
│   ├── dataset_process.py
│   ├── tokenizer.json
│   └── 📂 data/
│       ├── TinyStories-train.bin
│       └── TinyStories-valid.bin
│
├── 📂 训练输出
│   ├── 📂 checkpoints/          # 模型检查点
│   └── 📂 wandb/               # 训练日志
│
├── 📂 配置文件
│   ├── .gitignore
│   ├── train_win.sh            # Windows训练脚本
│   ├── train_linux.sh          # Linux训练脚本
│   └── LICENSE                 # Apache 2.0许可证
│
└── README.md                   # 项目说明文档
```

## 🏗️ 模型架构

### 标准Transformer

```python
# 模型配置示例
model = TransformerLM(
    d_model=512,           # 模型维度
    n_head=8,              # 注意力头数
    n_layer=6,             # Transformer层数
    d_ff=2048,             # 前馈网络维度
    vocab_size=30000,      # 词表大小
    max_seq_len=512,       # 最大序列长度
    theta=10000.0,         # RoPE参数
    use_rms_norm=True,     # 使用RMSNorm
    norm_model="pre",      # Pre-norm架构
    ffn_type="swiglu"      # SwiGLU激活函数
)
```

### MoE Transformer

```python
# MoE模型配置
moe_model = MoETransformerLM(
    d_model=512,
    n_head=8,
    n_layer=8,
    n_experts=8,           # 每层专家数量
    top_k=2,               # 激活专家数
    use_moe_aux_loss=True, # 负载均衡损失
    moe_aux_loss_weight=0.01
)
```

### 混合MoE架构

```python
# 混合架构（类似DeepSeek V2）
hybrid_model = HybridMoETransformerLM(
    d_model=512,
    n_layer=8,
    moe_layer_indices=[2, 5, 8],  # 指定MoE层
    n_experts=8,
    top_k=2
)
```

## 🎯 训练指南

### 基础训练参数

```bash
python train.py \
    --train_data_path data/train.bin \
    --valid_data_path data/valid.bin \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --d_ff 2048 \
    --vocab_size 30000 \
    --max_seq_len 512 \
    --batch_size 16 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 2000 \
    --total_steps 10000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --checkpoint_dir checkpoints \
    --save_interval 5000 \
    --log_interval 100 \
    --eval_interval 500 \
    --use_wandb \
    --wandb_project "transformer-lm"
```

### 分布式训练

```bash
# 启动4GPU分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    distributed/train_distributed.py \
    --train_data_path data/train.bin \
    --valid_data_path data/valid.bin \
    --distributed \
    --world_size 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --total_steps 50000
```

## 🧪 实验与消融

### 消融实验参数

```bash
# 移除RMSNorm
python train.py --no_rms_norm --train_data_path data/train.bin ...

# 移除RoPE位置编码
python train.py --no_rope --train_data_path data/train.bin ...

# 切换归一化位置
python train.py --norm_rope post --train_data_path data/train.bin ...

# 切换激活函数
python train.py --ffn_type silu --train_data_path data/train.bin ...
```

### 实验监控

项目集成WandB进行实验追踪：

```python
# 自动记录训练指标
wandb.init(project="transformer-lm", config=vars(args))
wandb.log({
    'train/loss': loss,
    'train/learning_rate': lr,
    'val/loss': val_loss
})
```

## ⚡ 性能优化

### 推理优化

1. **KV Cache**: 减少重复计算
2. **PagedAttention**: vLLM风格内存管理
3. **批量推理**: 提高吞吐量

### 训练优化

1. **混合精度**: FP16/BF16训练
2. **梯度累积**: 模拟大批次
3. **分布式训练**: 多GPU并行
4. **数据并行**: 高效数据加载

### 内存优化

```python
# 检查点恢复训练
python train.py --resume_from checkpoints/checkpoint_step_5000.pt ...

# 梯度检查点（节省内存）
model = TransformerLM(..., use_gradient_checkpointing=True)
```
## 🤝 贡献指南

### 贡献方式

1. **报告问题**: 在Issues中提交bug或功能请求
2. **代码贡献**: Fork项目，创建分支，提交PR
3. **文档改进**: 完善文档和示例
4. **实验分享**: 分享您的实验结果和发现

### 开发流程

```bash
# 1. Fork并克隆项目
git clone https://github.com/yourusername/GPT_LM.git
cd GPT_LM

# 2. 创建开发分支
git checkout -b feature/your-feature

# 3. 进行开发
# ... 编写代码 ...

# 4. 提交更改
git commit -m "Add your feature"

# 5. 推送并创建PR
git push origin feature/your-feature
```

## 📄 警示

本项目仅供学习参考，禁止使用在商业项目之中

## 🙏 致谢

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始Transformer论文
- [DeepSeek V2](https://github.com/deepseek-ai/DeepSeek-V2) - MoE架构参考
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention实现参考
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - 训练数据集

## 📞 联系方式

- 项目主页: [https://github.com/which_W/GPT_LM](https://github.com/which-W/GPT_LM)
- 问题反馈: [[GitHub Issues](https://github.com/which_W/GPT_LM/issues)](https://github.com/which-W/GPT_LM/issues)
---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我一个Star！**

Made with ❤️ by the which_W

</div>
