"""日常训练的中文菜单、参数转换与启动检查。"""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TYPES = {
    "dense": ("基础 Transformer 预训练", "train"),
    "mhc": ("mHC 预训练", "mhc.train_mhc"),
    "moe": ("MoE 预训练", "moe.train_moe"),
    "hybrid_moe": ("混合 Dense/MoE 预训练", "moe.train_moe"),
    "engram": ("Engram + MoE 预训练", "engram.train_engram_moe"),
    "ddp": ("基础 Transformer 分布式训练", "distributed.train_distribute_ddp"),
    "moe_ddp": ("MoE 分布式训练", "distributed.train_distribute_moe_ddp"),
    "sft": ("SFT 监督微调", "SFT.sft_train"),
    "ei": ("EI 专家迭代", "EI.ei_train"),
    "dpo": ("DPO 偏好优化", "RL.DPO"),
    "grpo": ("GRPO 强化学习教学实验", "RL.GRPO"),
    "tron": ("Tron 张量并行/数据并行训练", "tron_support.train"),
}
PRETRAIN_TYPES = {"dense", "mhc", "moe", "hybrid_moe", "engram", "ddp", "moe_ddp"}
RESUME_TYPES = {"dense", "mhc", "moe", "hybrid_moe", "ddp", "moe_ddp", "dpo", "tron"}


@dataclass
class LaunchPlan:
    """预览和实际启动共用同一份命令与配置。"""
    train_type: str
    output_dir: Path
    command: list
    environment: dict
    generated_config: dict | None = None


def create_parser():
    parser = argparse.ArgumentParser(description="选择项目中的训练方式；未指定类型时显示中文菜单")
    parser.add_argument("--train-type", choices=TRAIN_TYPES)
    for flag in ("smoke-test", "dry-run", "list-types", "use-wandb"):
        parser.add_argument(f"--{flag}", action="store_true")
    for flag in ("output-dir", "resume-from", "train-data-path", "valid-data-path", "tokenizer-path",
                 "model-id", "checkpoint-path", "prompt-path", "config-path"):
        parser.add_argument(f"--{flag}")
    for flag in ("total-steps", "batch-size", "seq-len", "d-model", "n-head", "n-layer", "d-ff",
                 "vocab-size", "n-experts", "top-k", "epochs", "rounds", "n-processes",
                 "gradient-accumulation-steps"):
        parser.add_argument(f"--{flag}", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--backend", choices=["auto", "gloo", "nccl"], default="auto")
    parser.add_argument("--data-dtype", choices=["uint16", "uint32", "int64"], default="int64")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="在 -- 后传递训练模块的其他参数")
    return parser


def list_types():
    for index, (key, (label, _)) in enumerate(TRAIN_TYPES.items(), 1):
        print(f"{index:2d}. {key:12s} {label}")


def select_type(args):
    """菜单只收集所选训练必需的信息，其他参数沿用默认配置。"""
    if args.train_type:
        return
    list_types()
    keys = list(TRAIN_TYPES)
    while True:
        choice = input("选择训练编号或类型名称，回车选择 dense：").strip().lower() or "dense"
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            choice = keys[int(choice) - 1]
        if choice in TRAIN_TYPES:
            args.train_type = choice
            break
        print("请输入菜单中的编号或类型名称。")
    if choice in {"sft", "ei"} and not args.model_id:
        args.model_id = input("模型名称或本地 Hugging Face 模型目录：").strip()
    if choice in {"dpo", "grpo"} and not args.train_data_path:
        args.train_data_path = input("训练数据 JSON/JSONL 文件路径：").strip()
    if choice == "tron" and not args.config_path:
        args.config_path = input("Tron 配置 JSON 文件路径：").strip()


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_file(value, label):
    if not value:
        raise ValueError(f"请提供{label}")
    path = resolve_path(value)
    if not path.is_file():
        raise ValueError(f"{label}不存在：{path}")
    return str(path)


def validate_text_data(path, fields, jsonl_only=False):
    """在加载模型前检查文本训练集；逐行检查避免额外占用大量内存。"""
    data_path = Path(path)
    if jsonl_only and data_path.suffix.lower() != ".jsonl":
        raise ValueError("SFT/EI 数据应为含 question、answer 的 JSONL 文件")
    with data_path.open(encoding="utf-8-sig") as stream:
        records = (json.loads(line) for line in stream if line.strip()) if data_path.suffix.lower() == ".jsonl" else json.load(stream)
        count = 0
        for count, record in enumerate(records, 1):
            if not isinstance(record, dict) or any(not isinstance(record.get(key), str) or not record[key].strip() for key in fields):
                raise ValueError(f"{data_path.name} 第 {count} 项必须包含非空文本字段：{', '.join(fields)}")
        if not count:
            raise ValueError(f"训练数据不能为空：{data_path}")


def device_settings(args):
    """Windows 分布式入口使用 Gloo/CPU，支持 NCCL 的环境可使用 GPU。"""
    distributed = args.train_type in {"ddp", "moe_ddp", "tron"}
    device, backend = args.device, args.backend
    if distributed:
        if backend == "auto":
            backend = "gloo" if device == "cpu" or os.name == "nt" else "nccl"
        if device == "auto":
            device = "cpu" if backend == "gloo" else "cuda"
        if (backend == "gloo") != (device == "cpu"):
            raise ValueError("本项目分布式入口使用 Gloo/CPU 或 NCCL/CUDA，请匹配 Device 与 Backend")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = "float32"
        if device == "cuda" and args.train_type != "grpo":
            import torch
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device == "cpu" and dtype != "float32" and args.train_type not in {"sft", "ei"}:
        raise ValueError("CPU 训练请使用 -DType float32 或 auto")
    return device, dtype, backend


def build_plan(args, python=sys.executable):
    """构造可审阅命令，不创建目录，不启动训练，不下载模型。"""
    kind = args.train_type
    if kind not in TRAIN_TYPES:
        raise ValueError("请先选择训练类型")
    if args.smoke_test and args.resume_from:
        raise ValueError("短测试不与恢复训练同时使用")
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            raise ValueError(f"{name} 必须为正数")
    if args.resume_from and kind not in RESUME_TYPES:
        raise ValueError(f"{kind} 暂不支持恢复优化器状态；SFT/EI 可用 -ModelId 加载已保存模型，GRPO 可用 -CheckpointPath 加载基础模型权重")
    if args.resume_from:
        resume = resolve_path(args.resume_from)
        if not resume.exists():
            raise ValueError(f"恢复检查点不存在：{resume}")
        if kind in PRETRAIN_TYPES and not resume.is_file():
            raise ValueError("预训练恢复需要 .pt 检查点文件")
        if kind in {"dpo", "tron"} and not resume.is_dir():
            raise ValueError("DPO/Tron 恢复需要检查点目录")
        if kind == "dpo" and not (resume / "trainer_state.json").is_file():
            raise ValueError("DPO 恢复需要包含 trainer_state.json 的 Trainer 检查点，不能使用仅导出权重的目录")
        if kind == "dpo" and not args.checkpoint_path:
            raise ValueError("DPO 恢复还需 -CheckpointPath 指定原训练使用的基础模型，以保持参考策略一致")
        if kind == "dpo":
            from importlib.metadata import version
            torch_version = tuple(int(part) for part in version("torch").split("+")[0].split(".")[:2])
            if torch_version < (2, 6):
                raise ValueError("当前 Transformers 恢复 DPO 优化器状态需要 PyTorch 2.6 或更高版本；本机可先进行新的 DPO 训练")
    output = resolve_path(args.output_dir or f"checkpoints_{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists() and not output.is_dir():
        raise ValueError("输出路径必须是目录")
    if output.exists() and not args.resume_from:
        if any(path.suffix in {".pt", ".safetensors", ".bin"} or path.name.startswith(("checkpoint", "sft_steps", "ei_iter"))
               for path in output.rglob("*")):
            raise ValueError("输出目录已包含模型，请选择新目录或通过 -ResumeFrom 恢复训练")
    extra = list(args.extra_args)
    if extra[:1] == ["--"]:
        extra.pop(0)
    protected = {"--output_dir", "--checkpoint_dir", "--resume_from", "--checkpoint_path", "--config",
                 "--model_id", "--train_data_path", "--valid_data_path", "--val_data_path", "--tokenizer_path"}
    if any(item.startswith("--") and any(flag.startswith(item.split("=", 1)[0]) for flag in protected) for item in extra):
        raise ValueError("输出、恢复、模型和数据路径请通过脚本的专用参数设置，不能放入 ExtraArgs")
    if kind == "ei" and args.total_steps:
        raise ValueError("EI 使用 -Rounds 控制外层迭代，不能使用 -TotalSteps")
    if kind != "ei" and args.rounds:
        raise ValueError("-Rounds 仅用于 EI")
    if args.model_id and kind not in {"sft", "ei"}:
        raise ValueError("-ModelId 仅用于 SFT/EI；Tron 的模型在配置文件中设置")
    if args.checkpoint_path and kind not in {"dpo", "grpo"}:
        raise ValueError("-CheckpointPath 仅用于 DPO/GRPO 的基础模型初始化")
    if args.config_path and kind != "tron":
        raise ValueError("-ConfigPath 仅用于 Tron")
    if args.n_processes and kind not in {"ddp", "moe_ddp", "tron"}:
        raise ValueError("-NProcesses 仅用于分布式训练")
    if args.epochs and kind not in {"engram", "ei", "grpo"}:
        raise ValueError("-Epochs 仅用于 Engram、EI 或 GRPO")
    if args.gradient_accumulation_steps and kind not in {"ddp", "moe_ddp", "sft", "dpo", "tron"}:
        raise ValueError("当前训练方式不支持 -GradientAccumulationSteps")
    if args.backend != "auto" and kind not in {"ddp", "moe_ddp", "tron"}:
        raise ValueError("-Backend 仅用于分布式训练")
    if args.use_wandb and kind in {"engram", "dpo", "grpo"}:
        raise ValueError("当前训练方式没有接入 WandB 日志")
    if kind == "tron" and args.dtype != "auto":
        raise ValueError("Tron 的精度由模块根据设备自动选择，请保留 -DType auto")
    if args.prompt_path and kind not in {"sft", "ei"}:
        raise ValueError("-PromptPath 仅用于 SFT/EI")
    if args.tokenizer_path and kind in {"sft", "ei", "tron"}:
        raise ValueError("该训练方式使用 Hugging Face 模型自带分词器，无需 -TokenizerPath")
    if kind == "tron" and (args.train_data_path or args.valid_data_path):
        raise ValueError("Tron 数据集需要在配置文件的 dataset 中设置")
    if kind not in PRETRAIN_TYPES and any(getattr(args, name) is not None for name in
                                         ("d_model", "n_head", "n_layer", "d_ff", "vocab_size", "n_experts", "top_k")):
        raise ValueError("模型结构参数仅用于本地预训练；其他方式从所提供模型或配置文件读取结构")
    device, dtype, backend = device_settings(args)
    batch = args.batch_size or (4 if kind in PRETRAIN_TYPES else 1)
    seq = args.seq_len or 256
    if seq < 2:
        raise ValueError("序列长度至少为 2")
    total = args.total_steps or (100 if args.smoke_test else 10000)
    log = 10 if args.smoke_test else 50
    interval = min(total, 50 if args.smoke_test else 500)
    save = min(total, 50 if args.smoke_test else 1000)
    warmup = min(10 if args.smoke_test else 500, max(0, total - 1))
    evaluate = 5 if args.smoke_test else 20
    environment = {"PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    if not args.use_wandb:
        environment["WANDB_MODE"] = "disabled"
    command = [str(python), "-m", TRAIN_TYPES[kind][1]]

    def add(**options):
        for name, value in options.items():
            if value is not None:
                command.extend([f"--{name}", str(value)])

    config = None
    if kind in PRETRAIN_TYPES:
        train = require_file(args.train_data_path or "data/TinyStories-train.bin", "训练 token 文件")
        valid = require_file(args.valid_data_path or "data/TinyStories-valid.bin", "验证 token 文件")
        tokenizer = require_file(args.tokenizer_path or "tokenizer_tinystories.json", "分词器文件")
        dim, heads = args.d_model or 256, args.n_head or 4
        if dim % heads or (dim // heads) % 2:
            raise ValueError("d_model 必须能整除 n_head，且每个头的维度应为偶数")
        experts, top = args.n_experts or 4, args.top_k or 2
        if top > experts:
            raise ValueError("TopK 不能大于专家数")
        add(train_data_path=train, valid_data_path=valid, tokenizer_path=tokenizer, data_dtype=args.data_dtype,
            vocab_size=args.vocab_size or 30000, d_model=dim, n_head=heads, n_layer=args.n_layer or 4,
            d_ff=args.d_ff or 1024, batch_size=batch, checkpoint_dir=output, dtype=dtype)
        if kind == "engram":
            add(seq_len=seq, n_epochs=args.epochs or 1, max_steps=total, eval_steps=evaluate,
                log_interval=log, learning_rate=args.learning_rate or 3e-4, n_experts=experts, top_k=top)
            command.append("--device_ids")
            if device == "cuda":
                command.append("0")
        else:
            add(max_seq_len=seq, total_steps=total, warmup_steps=warmup, max_lr=args.learning_rate or 3e-4,
                min_lr=(args.learning_rate or 3e-4) / 10, log_interval=log, eval_interval=interval,
                eval_steps=evaluate, save_interval=save)
            if kind in {"ddp", "moe_ddp"}:
                processes = args.n_processes or 1
                command = [str(python), "-m", "utils.torchrun", "--nnodes", "1", "--nproc_per_node", str(processes),
                           "--rdzv_backend", "c10d", "--rdzv_endpoint", "127.0.0.1:0"] + command[1:]
                command.append("--distributed")
                add(backend=backend, gradient_accumulation_steps=args.gradient_accumulation_steps or 1)
                if args.use_wandb:
                    command.append("--use_wandb")
            else:
                add(device=device)
                if args.use_wandb:
                    command.append("--use_wandb")
            if kind in {"moe", "hybrid_moe", "moe_ddp"}:
                add(n_experts=experts, top_k=top)
                if kind != "moe_ddp":
                    command.append("--use_hybrid_moe" if kind == "hybrid_moe" else "--use_moe")
                    if kind == "hybrid_moe":
                        add(moe_layer_indices=",".join(str(index) for index in range(1, args.n_layer or 4, 2)) or "0")
                    if device == "cuda":
                        add(device_ids="0")
            if args.resume_from:
                add(resume_from=resolve_path(args.resume_from))
    elif kind in {"sft", "ei"}:
        if not args.model_id:
            raise ValueError("SFT/EI 需要 -ModelId 模型名称或本地 Hugging Face 模型目录；本机 6 GB 显存建议先用小模型验证")
        if args.dtype != "auto":
            raise ValueError("SFT/EI 模块当前固定使用 BF16 权重，请保留 -DType auto")
        train = require_file(args.train_data_path or "data/gsm8k-train.jsonl", "GSM8K 训练文件")
        valid = require_file(args.valid_data_path or "data/gsm8k-val.jsonl", "GSM8K 验证文件")
        prompt = require_file(args.prompt_path or "prompts/r1_zero.prompt", "提示模板")
        for path in (train, valid):
            validate_text_data(path, ("question", "answer"), jsonl_only=True)
        add(model_id=args.model_id, train_data_path=train, val_data_path=valid, prompt_path=prompt,
            output_dir=output, batch_size=batch, micro_batch_size=1, device=device,
            lr=args.learning_rate or (5e-6 if kind == "sft" else 5e-5),
            eval_backend="transformers", max_eval_samples=2 if args.smoke_test else 100,
            max_tokens=16 if args.smoke_test else 128, eval_every_steps=1 if args.smoke_test else 20)
        if kind == "sft":
            steps = args.total_steps or (2 if args.smoke_test else 200)
            if steps < 2:
                raise ValueError("SFT 至少需要 2 步")
            add(max_steps=steps, max_train_len=seq, gradient_accumulation_steps=args.gradient_accumulation_steps)
            if args.smoke_test:
                add(dataset_size=2)
        else:
            add(n_ei_steps=args.rounds or (1 if args.smoke_test else 5), epochs_per_ei=args.epochs or 1,
                ei_batch_size=4 if args.smoke_test else 512, rollouts=2 if args.smoke_test else 8,
                max_train_len=seq)
    elif kind in {"dpo", "grpo"}:
        train = require_file(args.train_data_path, "-TrainDataPath JSON/JSONL 文件")
        valid = require_file(args.valid_data_path, "验证文件") if args.valid_data_path else None
        fields = ("prompt", "chosen", "rejected") if kind == "dpo" else ("prompt",)
        for path in (train, valid):
            if path:
                validate_text_data(path, fields)
        tokenizer = require_file(args.tokenizer_path or "tokenizer_tinystories.json", "分词器文件")
        checkpoint = require_file(args.checkpoint_path, "基础 Transformer 检查点") if args.checkpoint_path else None
        steps = args.total_steps or (1 if args.smoke_test else 1000)
        add(train_data_path=train, val_data_path=valid, tokenizer_path=tokenizer, checkpoint_path=checkpoint,
            output_dir=output, max_steps=steps, batch_size=batch, device=device)
        if kind == "dpo":
            add(gradient_accumulation_steps=args.gradient_accumulation_steps or 1, dtype=dtype,
                learning_rate=args.learning_rate or 5e-6, max_length=seq,
                resume_from=resolve_path(args.resume_from) if args.resume_from else None)
        else:
            if args.dtype not in {"auto", "float32"}:
                raise ValueError("GRPO 当前使用 float32，请保留 -DType auto 或 float32")
            add(num_epochs=args.epochs or 2, learning_rate=args.learning_rate or 1e-5,
                max_seq_len=seq, max_prompt_length=max(1, seq // 2), max_gen_length=min(seq // 2, 16 if args.smoke_test else 128),
                num_samples_per_prompt=2 if args.smoke_test else 4, num_ppo_updates=1 if args.smoke_test else 3,
                ppo_batch_size=2 if args.smoke_test else 4, warmup_steps=0 if args.smoke_test else min(100, steps - 1),
                logging_steps=1 if args.smoke_test else 10, save_steps=min(steps, 1 if args.smoke_test else 500))
    else:
        source = require_file(args.config_path, "-ConfigPath Tron 配置文件")
        with open(source, encoding="utf-8-sig") as stream:
            config = copy.deepcopy(json.load(stream))
        for section in ("distributed", "training", "checkpoint", "logging", "model", "dataset", "environment"):
            if not isinstance(config.get(section), dict):
                raise ValueError(f"Tron 配置缺少 {section} 对象，请使用完整项目配置")
        required = {"distributed": ("tp_size", "dp_size"),
                    "training": ("seed", "micro_batch_size", "seq_length", "gradient_accumulation_steps", "learning_rate", "total_train_steps", "max_tokens"),
                    "checkpoint": ("save_frequency",), "environment": ("OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"),
                    "dataset": ("name", "num_workers", "num_proc"), "model": ("name",)}
        for section, fields in required.items():
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Tron 配置缺少 {section}.{field}；参见 configs/tron_template.json")
        if not config["model"]["name"] or not config["dataset"]["name"]:
            raise ValueError("请填写 Tron 配置中的 model.name 和 dataset.name")
        # 访问令牌仅传给子进程环境，避免出现在预览和配置副本中。
        token = config["environment"].pop("HF_TOKEN", None)
        if token:
            environment["HF_TOKEN"] = token
        parallel = config["distributed"]
        processes = parallel.get("tp_size", 1) * parallel.get("dp_size", 1)
        if processes < 1 or (args.n_processes and args.n_processes != processes):
            raise ValueError("Tron 的进程数必须等于配置中的 tp_size × dp_size")
        parallel["use_cpu"] = device == "cpu"
        config["checkpoint"].update(save_dir=str(output), load_path=str(resolve_path(args.resume_from)) if args.resume_from else None)
        config["logging"]["use_wandb"] = args.use_wandb
        if args.total_steps or args.smoke_test:
            config["training"].update(total_train_steps=args.total_steps or 2, max_tokens=None)
        if args.smoke_test:
            config["checkpoint"]["save_frequency"] = 1
        for key, value in (("micro_batch_size", args.batch_size), ("seq_length", args.seq_len),
                           ("learning_rate", args.learning_rate), ("gradient_accumulation_steps", args.gradient_accumulation_steps)):
            if value is not None:
                config["training"][key] = value
        command = [str(python), "-m", "utils.torchrun", "--nnodes", "1", "--nproc_per_node", str(processes),
                   "--rdzv_backend", "c10d", "--rdzv_endpoint", "127.0.0.1:0", "-m", "tron_support.train",
                   "--config", str(output / "launcher_config.json")]
    command.extend(extra)
    return LaunchPlan(kind, output, command, environment, config)


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.list_types:
        list_types()
        return 0
    try:
        select_type(args)
        plan = build_plan(args)
        print(f"训练方式：{TRAIN_TYPES[plan.train_type][0]} ({plan.train_type})")
        print(f"模型输出目录：{plan.output_dir}")
        if plan.train_type in {"ddp", "moe_ddp", "tron"} and "--backend" in plan.command:
            print("分布式后端：" + plan.command[plan.command.index("--backend") + 1])
        print("执行命令：" + (subprocess.list2cmdline(plan.command) if os.name == "nt" else shlex.join(plan.command)))
        if plan.generated_config is not None:
            print("Tron 配置副本：" + json.dumps(plan.generated_config, ensure_ascii=False, indent=2))
        if args.dry_run:
            print("命令预览结束，未启动训练或写入文件。")
            return 0
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        if plan.generated_config is not None:
            (plan.output_dir / "launcher_config.json").write_text(json.dumps(plan.generated_config, ensure_ascii=False, indent=2), encoding="utf-8")
        record = {"train_type": plan.train_type, "command": plan.command, "options": vars(args)}
        (plan.output_dir / f"launch_{datetime.now():%Y%m%d_%H%M%S_%f}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(plan.environment)
        sys.stdout.flush()
        return subprocess.call(plan.command, cwd=ROOT, env=environment)
    except (ValueError, TypeError, OSError, EOFError, KeyboardInterrupt) as error:
        print(f"训练入口：{error or '已取消'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
