# GPT模型SFT (Supervised Fine-Tuning) 训练指南

本指南说明如何使用预训练的GPT模型进行监督微调(SFT),使模型能够更好地遵循指令和对话。

## 🚀 快速开始

### 1. 准备示例数据

```bash
# 创建100条示例SFT数据
python prepare_sft_data.py \
    --mode sample \
    --output data/sft_sample.jsonl \
    --num_samples 100

# 分割训练集和验证集
python prepare_sft_data.py \
    --mode split \
    --input data/sft_sample.jsonl \
    --train_output data/sft_train.jsonl \
    --val_output data/sft_val.jsonl \
    --val_ratio 0.1
```

### 2. 开始SFT训练

```bash
python sft_train.py \
    --train_data_path data/sft_train.jsonl \
    --valid_data_path data/sft_val.jsonl \
    --pretrain_checkpoint checkpoints/checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --checkpoint_dir sft_checkpoints \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --d_ff 2048 \
    --vocab_size 30000 \
    --max_seq_len 512 \
    --batch_size 4 \
    --num_epochs 3 \
    --max_lr 5e-5 \
    --prompt_template default \
    --use_wandb
```

### 3. 推理测试

```bash
# 单次推理
python sft_inference.py \
    --mode single \
    --checkpoint_path sft_checkpoints/sft_checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --instruction "解释什么是深度学习" \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6

# 交互式对话
python sft_inference.py \
    --mode interactive \
    --checkpoint_path sft_checkpoints/sft_checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6
```

## 📊 数据准备

### 数据格式

SFT训练支持两种JSONL数据格式:

**格式1: 标准instruction格式**
```json
{
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

**格式2: 简化prompt-response格式**
```json
{
    "prompt": "User: 你好\n\nAssistant: ",
    "response": "你好!有什么可以帮助你的?"
}
```

### 数据转换工具

`prepare_sft_data.py` 提供了多种数据转换功能:

#### 1. CSV转JSONL

```bash
python prepare_sft_data.py \
    --mode csv \
    --input data/raw_data.csv \
    --output data/sft_data.jsonl
```

#### 2. 对话格式转JSONL

```bash
python prepare_sft_data.py \
    --mode conversation \
    --input data/conversations.json \
    --output data/sft_data.jsonl
```

#### 3. Q&A文本转JSONL

```bash
python prepare_sft_data.py \
    --mode qa_text \
    --input data/qa.txt \
    --output data/sft_data.jsonl
```

#### 4. 创建示例数据

```bash
python prepare_sft_data.py \
    --mode sample \
    --output data/sft_sample.jsonl \
    --num_samples 1000
```

## 🎯 SFT训练

### 基础训练命令

```bash
python sft_train.py \
    --train_data_path data/sft_train.jsonl \
    --valid_data_path data/sft_val.jsonl \
    --pretrain_checkpoint checkpoints/checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --checkpoint_dir sft_checkpoints \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --batch_size 4 \
    --num_epochs 3 \
    --max_lr 5e-5
```

### 重要参数说明

#### 模型参数
- `--pretrain_checkpoint`: **必需** 预训练模型路径
- `--d_model`, `--n_head`, `--n_layer`: 必须与预训练模型一致

#### 训练参数
- `--batch_size`: 批次大小,建议2-8
- `--num_epochs`: 训练轮数,通常1-5轮
- `--max_lr`: 学习率,SFT通常使用较小值(1e-5 ~ 1e-4)

#### 提示词模板
- `--prompt_template`: 选择模板类型
  - `default`: 标准格式
  - `alpaca`: Alpaca格式
  - `chat`: 对话格式

## 🔮 模型推理

### 单次推理

```bash
python sft_inference.py \
    --mode single \
    --checkpoint_path sft_checkpoints/sft_checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --instruction "写一首关于春天的诗" \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --temperature 0.8 \
    --top_p 0.95
```

### 批量推理

```bash
python sft_inference.py \
    --mode batch \
    --checkpoint_path sft_checkpoints/sft_checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --test_file data/test.jsonl \
    --output_file results/predictions.jsonl \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6
```

### 交互式对话

```bash
python sft_inference.py \
    --mode interactive \
    --checkpoint_path sft_checkpoints/sft_checkpoint_final.pt \
    --tokenizer_path /path/to/tokenizer.py \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --temperature 0.7 \
    --max_new_tokens 512
```

### 生成参数调优

- `--temperature`: 温度参数(0.1-2.0)
  - 较低值(0.5-0.7): 更确定性的输出
  - 较高值(0.8-1.2): 更有创造性的输出
  
- `--top_p`: Nucleus采样(0.1-1.0)
  - 推荐值: 0.9-0.95
  
- `--top_k`: Top-K采样
  - 推荐值: 40-50


## 📁 文件结构

```
.
├── sft_train.py              # SFT训练主脚本
├── sft_inference.py          # 推理脚本
├── prepare_sft_data.py       # 数据准备工具
├── README_SFT.md             # 本文档
├── data/
│   ├── sft_train.jsonl       # 训练数据
│   └── sft_val.jsonl         # 验证数据
├── sft_checkpoints/          # SFT检查点目录
└── results/                  # 推理结果目录
