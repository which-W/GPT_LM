# GPT_LM：语言模型学习与实验项目

这个项目从基础张量运算开始实现自回归语言模型，同时包含 MoE、mHC、Engram、缓存推理、分布式训练和后训练实验。代码适合学习和验证机制；各实验的性能和训练效果需要通过实际数据衡量。

## 项目中有哪些内容

| 文件或目录 | 作用 |
| --- | --- |
| `transformer.py`、`transformer_block.py` | 基础语言模型：词嵌入、因果注意力、前馈网络、归一化和词表输出 |
| `attention.py`、`rope.py` | MHA、GQA、MQA、MLA 注意力变体，旋转位置编码与 KV 缓存 |
| `emb.py`、`Linnear.py`、`rmsnorm.py`、`layernorm.py`、`swiGLU.py` | 手写基础层，保留原文件名以兼容已有导入 |
| `cross_entropy.py`、`adamw.py`、`schedule.py`、`clip_gradient_noem.py` | 损失、优化器、学习率调度和梯度裁剪 |
| `tokenizer.py`、`dataset_process.py`、`get_batch.py` | BPE 分词器训练、文本编码和预测窗口采样 |
| `train.py`、`checkpoint_use.py` | 基础模型训练、验证及检查点管理 |
| `moe/` | 每个 token 选择少数专家计算，支持全 MoE 和部分层使用 MoE 的混合模型 |
| `mhc/` | 多残差流及混合连接的实验模型 |
| `engram/` | n-gram 哈希记忆检索与 MoE 的组合模型 |
| `distributed/` | 基础模型 DDP 和手动同步专家梯度的多进程训练 |
| `inference/` | 普通缓存生成、投机采样和自研分页缓存生成 |
| `vllm_support/` | 分页缓存、块复用、请求调度和批量生成的教学实现 |
| `SFT/`、`EI/` | 监督微调，以及筛选正确答案后再训练的专家迭代 |
| `RL/DPO.py`、`RL/GRPO.py` | 基础模型的 Hugging Face 包装、偏好优化和组相对策略优化 |
| `tron_support/` | LLaMA 权重加载、张量并行和数据并行实验 |
| `utils/`、`tests/` | 共用功能、回归测试与训练入口短测试 |

普通训练的数据流是：文本 → 分词器 → token 数组 → T+1 长度的窗口 → 前 T 个 token 输入模型 → 后 T 个 token 作为标签 → 交叉熵 → 反向传播 → 更新参数。

推理先计算整个提示词，再每次只输入一个新 token，读取已有 KV 缓存。投机采样使用草稿模型提出候选，目标模型根据概率分布接受或替换。分页引擎把请求的缓存分配到物理块中，通过调度复用和回收。

## 环境与依赖

支持 Python 3.11–3.12。依赖以 `pyproject.toml` 和 `uv.lock` 为准，PyTorch 三个相关包统一使用 CUDA 12.4 索引。

```bash
uv sync --locked
```

也可以在自己的虚拟环境中安装 `requirements.txt`。它列出相同的直接依赖；精确复现完整依赖树应使用锁文件。项目已有的 `.venv` 没有被本次修复重新安装。

真实 vLLM 是 Linux 可选依赖：`uv sync --locked --extra vllm`。SFT/EI 默认评估后端为 Transformers，与训练模型共享设备，支持 Windows 和单 GPU。`vllm_support/` 使用项目自己的实现，不需要安装外部 vLLM。

## 数据格式与分词器

现有 TinyStories `.bin` 文件由 **int64** 写入，因此所有训练入口默认 `--data_dtype int64`。旧代码按 uint16 读取会把一个 token 拆成多个数字，改变训练数据。外部数据如果使用 uint16 或 uint32，必须显式指定对应参数。

新预处理生成同名 `.meta.json`，记录 dtype、token 数量和分词器指纹。读取时核对元数据，避免格式或分词器混用。

```bash
python dataset_process.py --input_path data/my_train.txt --output_path data/my_train.bin --tokenizer_path tokenizer_tinystories.json --dtype int64
```

`tokenizer.py` 只有直接执行时才会训练分词器，导入不会覆盖文件。重新训练分词器后应重新生成数据，并训练匹配词表的模型。`mapping_dicts/` 的历史映射不能直接用于当前分词器，必须确保来源词表一致。

## 基础模型训练

在项目根目录运行。建议将修复后的训练结果保存到新目录，保留历史模型供比较：

```bash
python train.py --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --data_dtype int64 --tokenizer_path tokenizer_tinystories.json --checkpoint_dir checkpoints_fixed --dtype float32
```

支持 `--no_rope`、`--no_rms_norm`、`--norm_rope pre/post`、`--ffn_type swiglu/silu` 消融选项。`--no_rms_norm` 同时关闭层内与最终归一化。CUDA 的 float16、bfloat16 设置使用自动混合精度，并保留单精度参数；FP16 启用梯度缩放。

Windows PowerShell 使用 `./train_win.ps1`；Git Bash 使用 `bash train_win.sh`；Linux 使用 `bash train_linux.sh`。脚本接受额外训练参数。完整选项见 `python train.py --help`。

新检查点保存模型结构、优化器、迭代次数和分词器内容。通过 `--resume_from` 恢复训练时，必须使用相同结构和分词器。

## 模型变体

```bash
python -m mhc.train_mhc --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --checkpoint_dir checkpoints_mhc
python -m moe.train_moe --use_moe --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --checkpoint_dir checkpoints_moe
python -m moe.train_moe --use_hybrid_moe --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --checkpoint_dir checkpoints_hybrid
python -m engram.train_engram_moe --checkpoint_dir checkpoints_engram
```

MoE 可在 CPU、单 GPU 或同一进程的多个 GPU 上放置专家，使用张量复制访问专家。当前没有实现跨进程 all-to-all 专家分片。Engram 默认训练真实二进制数据；传入 `--demo_random` 才使用随机演示数据。哈希表使用有界大小，避免按词表的 n 次方分配。

## 分布式训练

Linux 多 GPU 示例：

```bash
python -m utils.torchrun --nproc_per_node 2 -m distributed.train_distribute_ddp --distributed --backend nccl --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --checkpoint_dir checkpoints_ddp
```

`utils.torchrun` 在 Linux 调用标准启动器，在 Windows 关闭部分 PyTorch 构建缺少的 libuv 后端。Windows CPU 多进程验证使用 `--backend gloo`；NCCL 多 GPU 训练需要支持 NCCL 的 Linux 环境。

MoE 分布式入口是 `distributed.train_distribute_moe_ddp`。进程按一致顺序同步所有参数，包括本进程未激活、但其他进程有梯度的专家。所有进程参与损失集合通信，只有主进程输出日志并保存文件。

## 推理

```bash
python -m inference.inference --model_path checkpoints_fixed/checkpoint_final.pt --tokenizer_path tokenizer_tinystories.json --prompt "Once upon a time"
python -m inference.vllm_inference --model_path checkpoints_fixed/checkpoint_final.pt --tokenizer_path tokenizer_tinystories.json --prompt "Once upon a time"
```

普通生成按检查点配置加载基础、mHC、MoE 或 Engram 模型。分页生成当前支持基础 Transformer。投机采样需要相同分词器、相同输出词表维度及支持缓存回退的模型。

历史检查点未记录完整配置。词表大小、层数、隐藏维度和前馈维度可从权重推断；头数、上下文长度及 RoPE 参数使用旧默认值或显式覆盖。现有历史文件默认按 8 个头、512 上下文和 theta=10000 解释。优先使用内嵌分词器；采样会屏蔽实际分词器之外的输出位置。

## SFT、EI、DPO 与 GRPO

SFT/EI 使用 Hugging Face 模型和分词器。GSM8K JSONL 每行包含 `question`、`answer`，答案可用 `####` 分隔推理与结果。EI 还支持包含 `problem`、`answer` 的 Parquet 数据。

```bash
python -m SFT.sft_train --model_id Qwen/Qwen2.5-Math-1.5B --train_data_path data/gsm8k-train.jsonl --val_data_path data/gsm8k-val.jsonl --prompt_path prompts/r1_zero.prompt --eval_backend transformers --device cuda:0
python -m EI.ei_train --model_id Qwen/Qwen2.5-Math-1.5B --train_data_path data/gsm8k-train.jsonl --val_data_path data/gsm8k-val.jsonl --prompt_path prompts/r1_zero.prompt --eval_backend transformers --device cuda:0
```

SFT 分别处理提示词、回答和填充位置，熵正则与主损失一起反向传播。EI 一轮没有正确候选时跳过该轮更新。

DPO 数据必须包含真实的 `prompt`、`chosen`、`rejected` 三个字符串字段，格式为 JSON 数组或 JSONL：

```json
{"prompt":"问题","chosen":"偏好的回答","rejected":"不偏好的回答"}
```

```bash
python -m RL.DPO --train_data_path data/preferences.jsonl --checkpoint_path checkpoints_fixed/checkpoint_final.pt --output_dir dpo_output
python -m RL.GRPO --help
```

DPO/GRPO 当前支持基础 Transformer。GRPO 使用逐 token 策略比率、截断目标及 KL 项；默认奖励模型是教学用启发式评分，不能据此证明数学答案正确，也没有经过训练的奖励网络。

Tron 的说明见 `tron_support/READE_TRON.md`。预训练权重必须匹配实际 LLaMA 结构，按名称严格映射。公开模型不强制要求 HF_TOKEN，受限模型仍需相应访问权限。

## 验证与历史结果

```bash
python -m unittest discover -s tests -v
python tests/smoke_training.py
```

第一条验证数据兼容、参数转换、训练预算和输出保护。第二条使用临时数据和本地小模型实际运行训练入口，并检查保存的步数、分词器及权重，不修改现有 TinyStories 数据和模型；Tron 只检查命令预览。可指定训练类型，例如 `python tests/smoke_training.py ddp moe_ddp --processes 2`。

详细修复及验证范围见 `FIXES.md`。如果历史模型曾使用错误 dtype 读取数据训练，其已学到的参数无法通过修改代码恢复正确训练结果，需要使用正确数据重新训练。现有历史文件保留供比较。

## 日常训练

当前 TinyStories 二进制数据使用原始 30000 词表。已从 Git 初始版本恢复对应分词器为 `tokenizer_tinystories.json`，并抽查训练/验证数据的首段编码匹配。当前 `tokenizer.json` 的 4642 词表与历史数据不匹配，不用于这些文件。训练读取会检查 token 编号范围。详细步骤见 [TRAINING.md](TRAINING.md)。

`train_daily.ps1` 现已统一接入基础、mHC、MoE、混合 MoE、Engram、DDP、MoE DDP、SFT、EI、DPO、GRPO 与 Tron 共 12 种训练方式。不指定类型时显示中文选择菜单：

```powershell
.\train_daily.ps1
.\train_daily.ps1 -ListTypes
.\train_daily.ps1 -TrainType moe -SmokeTest
.\train_daily.ps1 -TrainType engram -SmokeTest -DryRun
```

SFT/EI 需要 `ModelId`；DPO/GRPO 需要对应的 JSON/JSONL 数据；Tron 需要 `ConfigPath`，配置模板见 `configs/tron_template.json`。Windows 分布式默认使用 Gloo/CPU。
