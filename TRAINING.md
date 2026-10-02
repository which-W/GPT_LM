# 日常训练说明

本机为 RTX 3060 Laptop、6 GB 显存，支持 BF16。先使用基础 Transformer 与 TinyStories 英文故事数据，完成短测试，再进行正式训练和架构比较。

## 当前数据与分词器

现有 `data/TinyStories-train.bin` 和 `data/TinyStories-valid.bin` 使用 int64，token 编号最高为 29999。配套分词器为 `tokenizer_tinystories.json`，词表大小 30000，终止符编号为 0。

该文件恢复自 Git 提交 `b10af25`。抽查训练集前 22814 个 token、验证集前 23110 个 token，与对应原文本重新编码逐 token 匹配。`tokenizer.json` 的 4642 词表与这些历史文件不匹配，不能混用；原文件保留供其他实验使用。

本地预训练的数据读取会检查数据编号、分词器编号和模型词表大小。新生成数据还会核对元数据中的分词器指纹。编号范围相同并不能单独证明分词器完全一致，因此更换分词器时仍需重新编码数据。

## 统一训练入口与选择菜单

直接运行脚本会显示中文菜单，输入编号或类型名称；回车选择基础 Transformer：

```powershell
.\train_daily.ps1
```

只查看选项，或者预览某种训练实际使用的命令：

```powershell
.\train_daily.ps1 -ListTypes
.\train_daily.ps1 -TrainType moe -SmokeTest -DryRun
```

`-DryRun` 会检查参数与必需文件，然后打印命令；不会训练、下载模型或创建输出目录。未指定 `TrainType` 时，即使使用 `SmokeTest` 或 `DryRun`，也会显示菜单。脚本委托 `utils/training_launcher.py` 处理不同入口的参数，并为每次实际运行在输出目录记录一份 `launch_*.json`。

| TrainType | 训练内容 | 默认数据/必需输入 | 正式训练 / 短测试预算 |
| --- | --- | --- | --- |
| dense | 基础 Transformer | TinyStories token 文件 | 10000 / 100 步 |
| mhc | 多路残差连接 mHC | TinyStories token 文件 | 10000 / 100 步 |
| moe | 每层使用专家网络 | TinyStories token 文件 | 10000 / 100 步 |
| hybrid_moe | 普通前馈层与专家层交替 | TinyStories token 文件 | 10000 / 100 步 |
| engram | Engram 记忆模块 + MoE | TinyStories token 文件 | 10000 / 100 步 |
| ddp | 基础模型分布式训练 | TinyStories token 文件 | 10000 / 100 步 |
| moe_ddp | MoE 分布式训练 | TinyStories token 文件 | 10000 / 100 步 |
| sft | 用标准答案监督微调 | GSM8K JSONL；必需 ModelId | 200 / 2 步 |
| ei | 采样、筛选正确答案、再训练 | GSM8K JSONL；必需 ModelId | 5 / 1 轮 |
| dpo | 优化好答案相对坏答案的概率 | 必需偏好 JSON/JSONL | 1000 / 1 步 |
| grpo | 候选回答分组奖励与策略更新 | 必需提示词 JSON/JSONL | 1000 / 1 组更新 |
| tron | LLaMA 张量并行/数据并行 | 必需 ConfigPath 完整配置 | 按配置 / 2 步 |

所有预算均可调整。`-TotalSteps` 是目标总步数，恢复训练时也包含此前完成的步数；EI 使用 `-Rounds`。Engram 指定步数后按该预算训练，必要时重复遍历数据，`Epochs` 不再作为停止条件。GRPO 的一组更新包含采样和若干 PPO 小批次更新，因此不能直接按步数与普通预训练比较计算量。

## 先进行短测试

在 PowerShell 中进入项目根目录：

```powershell
cd 'E:\桌面\所有学习资料\deep_learning\GPT_LM'
.\train_daily.ps1 -TrainType dense -SmokeTest -OutputDir checkpoints_try_01
```

这会从头训练 100 步，每 10 步输出训练损失，每 50 步验证并保存模型。确认没有词表错误、显存不足或 NaN，且目录中出现检查点文件。100 步主要验证流程，不能据此要求生成流畅故事。

若 PowerShell 提示脚本执行受限，可以仅对本次调用绕过限制：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_daily.ps1 -TrainType dense -SmokeTest -OutputDir checkpoints_try_01
```

## 正式训练

```powershell
.\train_daily.ps1 -TrainType dense -OutputDir checkpoints_daily_01
```

默认配置：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| d_model | 256 | 每个 token 的隐藏向量维度 |
| n_layer / n_head | 4 / 4 | 模型层数与注意力头数 |
| d_ff | 1024 | 前馈网络内部维度 |
| vocab_size | 30000 | 与配套分词器一致的模型词表 |
| max_seq_len | 256 | 每段训练序列的 token 数 |
| batch_size | 4 | 每次更新使用的序列数 |
| dtype / device | auto / auto | 本机自动选择 CUDA BF16；CPU 使用 float32 |
| total_steps / warmup_steps | 10000 / 500 | 更新次数与预热次数 |
| max_lr / min_lr | 3e-4 / 3e-5 | 学习率上限与下限 |
| log / eval / save interval | 50 / 500 / 1000 | 打印、验证及保存间隔 |

模型约 1960 万参数。已用现有真实数据完成一次 CUDA BF16 前向、反向与优化器更新，参数和梯度有限，PyTorch 峰值已分配显存约 0.55 GiB。这不包含 CUDA 上下文和其他程序占用，也不代表长期训练性能。

10000 步表示 10000 次参数更新，不表示完整遍历数据一次。当前训练采用随机窗口采样；每步约处理 4×256=1024 个输入 token。先通过短测试观察速度，再决定训练预算。

输出目录未指定时使用时间戳生成新目录。指定的目录如果已有检查点，脚本会要求使用新目录或恢复训练，避免无意覆盖。

## 观察效果

训练损失 `Loss` 反映模型在当前训练样本上的预测误差。`Validation Loss` 反映未参与更新的验证数据，更适合比较模型。短期波动正常；应观察多次验证的趋势，并保持相同评估设置。

训练结束后进行故事续写：

```powershell
.\.venv\Scripts\python.exe -m inference.inference --model_path .\checkpoints_daily_01\checkpoint_final.pt --tokenizer_path .\tokenizer_tinystories.json --prompt "Once upon a time" --max_new_tokens 100
```

可选取中间检查点比较，比如 `checkpoint_step_5000.pt`。使用同一提示词和采样参数有助于比较，但随机采样结果仍有波动。TinyStories 主要训练英文故事续写，不能直接等同于中文问答或指令遵循能力。

## 中断与恢复

终端按 Ctrl+C 会停止训练。当前入口不会在中断时额外保存，因此从最近一次已经写完的检查点恢复：

```powershell
.\train_daily.ps1 -TrainType dense -OutputDir checkpoints_daily_01 -ResumeFrom .\checkpoints_daily_01\checkpoint_step_5000.pt
```

默认目标仍为 10000 步，因此此例接着执行剩余 5000 步。恢复时保持模型结构、序列长度、分词器及原计划中的总步数/预热设置一致。`BatchSize` 可以根据显存调整。修改总步数会重新计算余弦学习率轨迹，并非完全复现原计划。

每份检查点包含模型和优化器，这组配置每份约 230 MB；默认 10000 步会保存多个中间文件及最终文件，需预留几 GB 磁盘空间。

## 调整规模

显存不足时先降低批大小：

```powershell
.\train_daily.ps1 -BatchSize 2 -OutputDir checkpoints_batch2_01
```

从头开始的新实验也可缩短序列：

```powershell
.\train_daily.ps1 -BatchSize 2 -SeqLen 128 -OutputDir checkpoints_seq128_01
```

序列长度属于模型配置，不应在恢复已有检查点时随意改变。每次实验主要改变一个设置，并使用新目录，便于比较验证损失和生成效果。

也可以直接运行原有训练入口。例如：

```powershell
.\.venv\Scripts\python.exe -m train --train_data_path data/TinyStories-train.bin --valid_data_path data/TinyStories-valid.bin --tokenizer_path tokenizer_tinystories.json --data_dtype int64 --vocab_size 30000 --d_model 256 --n_head 4 --n_layer 4 --d_ff 1024 --max_seq_len 256 --batch_size 4 --device cuda --dtype bfloat16 --total_steps 10000 --warmup_steps 500 --checkpoint_dir checkpoints_custom_01
```

完整参数见 `python -m train --help`。当前电脑只有一张 GPU，日常训练直接使用单卡入口即可。基础流程和结果稳定后，再分别研究 MoE、mHC 或 Engram。SFT/EI 默认的 1.5B 模型使用全参数训练，不适合作为本机 6 GB 显存的直接起步配置；当前代码没有实现 LoRA/QLoRA。


## 比较预训练架构

用同一份数据、步数和模型维度进行实验，各自使用不同输出目录：

```powershell
.\train_daily.ps1 -TrainType mhc -SmokeTest -OutputDir checkpoints_mhc_try
.\train_daily.ps1 -TrainType moe -SmokeTest -NExperts 4 -TopK 2 -OutputDir checkpoints_moe_try
.\train_daily.ps1 -TrainType hybrid_moe -SmokeTest -OutputDir checkpoints_hybrid_try
.\train_daily.ps1 -TrainType engram -SmokeTest -BatchSize 2 -OutputDir checkpoints_engram_try
```

混合 MoE 默认在第 1、3 层使用专家网络，第 0、2 层使用普通前馈网络，索引从零开始。只有一层时使用该层作为专家层。Engram 的短测试只验证少量批次，并保存 `checkpoint_final.pt` 与验证表现最好的 `best_model.pt`；长期运行时还会在每五轮保存一次检查点。

脚本默认只使用第 0 张 GPU。MoE/Engram 的专家分配、记忆层和表大小可用 `ExtraArgs` 调整；需要多 GPU 时按原模块的设备参数配置。不要根据基础模型的 0.55 GiB 测试结果推断其他架构的显存占用。

## 分布式训练

本机为 Windows，分布式默认使用 Gloo/CPU。两进程可以验证通信流程，不会变成两张显卡：

```powershell
.\train_daily.ps1 -TrainType ddp -SmokeTest -Device cpu -Backend gloo -NProcesses 2
.\train_daily.ps1 -TrainType moe_ddp -SmokeTest -Device cpu -Backend gloo -NProcesses 2
```

入口通过 `utils.torchrun` 启动进程，并自动选择可用通信端口。Windows 的兼容处理关闭缺失的 libuv 后端。在支持 NCCL 的 Linux/CUDA 环境，使用 Python 入口并指定实际 GPU 数量：

```bash
python -m utils.training_launcher --train-type ddp --device cuda --backend nccl --n-processes 2 --smoke-test
```

`BatchSize` 表示每个进程的批大小。梯度累积使用 `-GradientAccumulationSteps`；总有效批大小还需乘以数据并行进程数。`moe_ddp` 的专家并行参数仍按其原模块设置。

## SFT 与 EI

这两种方式使用 Hugging Face 模型及其自带分词器，不使用 TinyStories 的二进制数据。必须指定模型名称或已有本地模型目录；提供在线模型名称时，正式启动可能下载该模型。当前实现为全参数训练，默认评估后端 `transformers` 复用训练模型。

```powershell
.\train_daily.ps1 -TrainType sft -ModelId '本地模型目录或HF模型名称' -SmokeTest -OutputDir checkpoints_sft_try
.\train_daily.ps1 -TrainType ei -ModelId '本地模型目录或HF模型名称' -SmokeTest -OutputDir checkpoints_ei_try
.\train_daily.ps1 -TrainType ei -ModelId '本地模型目录或HF模型名称' -Rounds 3 -Epochs 1 -OutputDir checkpoints_ei_01
```

默认读取 `data/gsm8k-train.jsonl`、`data/gsm8k-val.jsonl` 和 `prompts/r1_zero.prompt`；可用 `TrainDataPath`、`ValidDataPath`、`PromptPath` 替换。每行数据需要非空 `question`、`answer` 文本。脚本当前集成 JSONL；使用 EI 的 Parquet 分支时直接调用原模块。

SFT 短测试只使用两个训练样本、两个验证样本，更新两步。EI 短测试采样最多四道题，每题两个候选；没有正确答案就跳过该轮更新，也不会生成该轮模型检查点。EI 的自定义提示文件名需包含 `r1` 或 `question_only`，用于选择对应评分方式。

默认批大小为 1，微批大小为 1。`SeqLen` 控制训练截断长度；短测试每次最多生成 16 个 token。模型权重精度目前固定为 BF16，因此保留 `DType=auto`。CPU 可以验证流程，但速度与模型支持情况取决于所选模型。

## DPO 与 GRPO

准备自己的 JSON 数组或 JSONL 数据。DPO 每项需要三个文本字段：

```json
{"prompt":"问题", "chosen":"偏好的回答", "rejected":"不偏好的回答"}
```

GRPO 每项至少需要：

```json
{"prompt":"希望模型回答的问题"}
```

它们目前只包装基础 Transformer。优先用已经预训练的基础模型初始化：

```powershell
.\train_daily.ps1 -TrainType dpo -TrainDataPath data/preferences.jsonl -CheckpointPath checkpoints_daily_01/checkpoint_final.pt -SmokeTest -OutputDir checkpoints_dpo_try
.\train_daily.ps1 -TrainType grpo -TrainDataPath data/prompts.jsonl -CheckpointPath checkpoints_daily_01/checkpoint_final.pt -SmokeTest -OutputDir checkpoints_grpo_try
```

检查点存在时，从其中读取模型结构与内嵌分词器。没有指定检查点时，会用本地分词器创建小型随机基础模型，适合流程实验；这种模型尚未具备预训练能力。DPO 默认批大小和梯度累积均为 1；GRPO 当前使用 float32。

GRPO 默认奖励是教学用启发式评分，不等于数学正确性，也不是经过训练的奖励网络。短测试进行一组更新，每个提示生成两个候选，再进行一轮 PPO 更新。

DPO 输出为 Hugging Face 格式，定期保存 `checkpoint-步数` 目录。恢复完整 Trainer 状态需要 PyTorch 2.6 或更高版本；本机当前 2.5.1 的环境不支持这项恢复，入口会提前报错。可用的环境中，恢复还必须通过 `CheckpointPath` 指定原始基础模型，以保持冻结参考策略一致。

## Tron 配置入口

先复制 `configs/tron_template.json`，填入 `model.name` 和 `dataset.name`。模型应与项目的 LLaMA 实现及权重映射兼容；数据集必须有 `text` 列，支持 Hugging Face 数据集名称及其 `subset_name`。不要将 TinyStories `.bin` 文件作为这里的数据集名称。

```powershell
Copy-Item configs/tron_template.json configs/tron_my_config.json
# 编辑 tron_my_config.json 中的模型、数据集及训练参数。
.\train_daily.ps1 -TrainType tron -ConfigPath configs/tron_my_config.json -DryRun
.\train_daily.ps1 -TrainType tron -ConfigPath configs/tron_my_config.json -SmokeTest -Device cpu
```

脚本会在输出目录生成 `launcher_config.json` 副本，修改保存目录和本次显式指定的参数，原配置文件保持不变。短测试限制为两步且每步保存。进程数由 `tp_size × dp_size` 决定；如果同时设置 `NProcesses`，必须与此一致。

`Device` 和 `Backend` 控制本次运行使用 CPU/Gloo 或 CUDA/NCCL；Windows 默认选择 CPU/Gloo。Tron 的精度由模块按设备自动选择，保留 `DType=auto`。训练仍可能下载模型与数据集。受限模型的访问令牌建议通过 `HF_TOKEN` 环境变量提供；配置中的 `HF_TOKEN` 会转为子进程环境变量，不打印或写入配置副本。

## 常用参数与高级参数

| 脚本参数 | 用途 |
| --- | --- |
| TrainType / ListTypes | 选择训练方式 / 查看列表 |
| SmokeTest / DryRun | 小预算检查流程 / 只预览命令 |
| OutputDir / ResumeFrom | 保存目录 / 恢复已有训练状态 |
| TotalSteps / Rounds | 总步数 / EI 外层迭代轮数 |
| BatchSize / SeqLen | 批大小 / 本地上下文或后训练截断长度 |
| DModel / NHead / NLayer / DFF / VocabSize | 本地预训练的模型结构 |
| NExperts / TopK | MoE 的专家数量与激活数量 |
| Device / DType | 设备与计算精度；部分后训练精度固定 |
| LearningRate | 对应训练方式的学习率 |
| GradientAccumulationSteps | DDP、MoE DDP、SFT、DPO、Tron 的梯度累积 |
| TrainDataPath / ValidDataPath / TokenizerPath / DataDType | 按训练类型指定数据与分词器 |
| ModelId / CheckpointPath / ConfigPath | SFT/EI 模型 / DPO/GRPO 基础模型 / Tron 配置 |
| NProcesses / Backend | 分布式进程数与通信后端 |
| UseWandb | 显式启用已接入 WandB 的训练日志 |
| ExtraArgs | 以字符串数组传递原训练模块的其他参数 |

默认关闭 WandB，避免短测试要求登录。高级参数需要使用各模块实际支持的名称。例如：

```powershell
.\train_daily.ps1 -TrainType hybrid_moe -SmokeTest -ExtraArgs @('--moe_layer_indices', '0,2')
.\train_daily.ps1 -TrainType engram -SmokeTest -ExtraArgs @('--engram_table_size', '4096', '--engram_layers', '1', '3')
```

数据、模型、输出和恢复路径必须使用脚本的专用参数。其他高级参数放在默认参数之后，可以覆盖对应模块的默认值，也可能改变短测试预算；先使用 `DryRun` 确认命令。

完整恢复支持基础、mHC、MoE、混合 MoE、DDP、MoE DDP，以及条件满足的 DPO、Tron。Engram、SFT、EI、GRPO 当前没有统一的优化器恢复入口，使用 `ResumeFrom` 会明确报错。SFT/EI 可以用 `ModelId` 加载先前保存的模型权重开始新阶段，这不会恢复原优化器状态。

## 本次入口验证

已通过 19 项自动测试，并实际完成七种预训练入口的两步 CPU 训练，检查最终步数、内嵌分词器和权重有限性；DDP 与 MoE DDP 已验证双进程 Gloo，Engram 另通过 CUDA BF16 两步训练。DPO、GRPO 和 SFT 使用本地小模型完成了离线短测试。EI 已验证采样、评估及无正确答案时跳过更新的路径。Tron 已验证参数转换、配置副本和预览流程，尚未执行模型/数据下载后的完整训练；Linux NCCL 多 GPU 训练也未在本机验证。


可重复执行这些检查，不修改正式数据和已有模型：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_training.py
.\.venv\Scripts\python.exe tests/smoke_training.py ddp moe_ddp --processes 2
```

冒烟检查使用临时的随机小模型和测试样本，仅证明流程与保存行为可运行，不衡量模型能力；Tron 场景只做预览。
