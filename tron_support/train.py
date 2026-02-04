"""Training script for LLaMA model with DP+TP parallelism.
示例用法:
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 train.py --config tmp/test_dp2_tp2/config.json
"""
import os
import json
import time
import datetime
import argparse
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig

from tron_support.tensor_parallel_v.tensor_parallel import apply_tensor_parallel
import tron_support.process_group_manager as pgm
from tron_support.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format, get_mfu, get_num_params
from tron_support.checkpoint import CheckpointManager
from tron_support.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights
from tron_support.data import MicroBatchDataLoader
from tron_support.process_group_manager import setup_process_group_manager
from tron_support.data_parallel_v.data_parallel import DataParallelBucket
from tron_support.model import Llama
from tron_support.utils import download_model
import wandb


def train_step(model, data_loader, device):
    """执行一个训练步骤（包含梯度累积）"""
    acc_loss = 0.0
    
    # 判断是否需要梯度同步（当有数据并行时）
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1
    
    for i in range(data_loader.grad_acc_steps):
        # 获取下一个 micro-batch
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        position_ids = batch["position_ids"].to(device) if "position_ids" in batch else None

        # 禁用梯度同步，除了最后一个 micro-batch
        if requires_grad_sync:
            model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

        # 前向传播
        outputs = model(input_ids=input_ids, position_ids=position_ids)

        # 计算损失
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len * batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps
        
        # 反向传播
        loss.backward()

        acc_loss += loss.item()

    return acc_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = config["environment"]["OMP_NUM_THREADS"]
    os.environ["TOKENIZERS_PARALLELISM"] = config["environment"]["TOKENIZERS_PARALLELISM"]
    os.environ["DEVICE"] = "cpu" if config["distributed"]["use_cpu"] else "cuda"
    
    # HF Token 处理
    if config["environment"].get("HF_TOKEN") is None:
        if "HF_TOKEN" not in os.environ:
            raise ValueError("HF_TOKEN is neither set in the config file nor in the environment")
    else:
        if "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = config["environment"]["HF_TOKEN"]
        else:
            print("Warning: HF_TOKEN is set in both environment and config. Using environment variable.")
    
    # 数据类型设置
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not config["distributed"]["use_cpu"] else torch.float32

    # 分布式设置
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "gloo" if config["distributed"]["use_cpu"] else "nccl"
    
    # 验证 world_size
    assert world_size == config["distributed"]["tp_size"] * config["distributed"]["dp_size"], \
        f"world_size ({world_size}) must equal tp_size ({config['distributed']['tp_size']}) * dp_size ({config['distributed']['dp_size']})"

    # 设置设备
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # 初始化分布式进程组
    dist.init_process_group(
        rank=global_rank, 
        world_size=world_size, 
        backend=backend, 
        init_method=f"env://", 
        timeout=datetime.timedelta(minutes=3)
    )
    
    # 设置进程组管理器（只需要 TP 和 DP，CP=1, PP=1）
    setup_process_group_manager(
        tp_size=config["distributed"]["tp_size"],
        cp_size=1,  # 不使用 Context Parallel
        pp_size=1,  # 不使用 Pipeline Parallel
        dp_size=config["distributed"]["dp_size"]
    )
    
    # 确定哪个 rank 负责 wandb 日志
    is_wandb_rank = (pgm.process_group_manager.tp_rank == 0 and 
                     pgm.process_group_manager.dp_rank == 0)

    # 设置随机种子
    set_all_seed(config["training"]["seed"])

    # 创建数据加载器
    start_time = time.time()
    data_loader = MicroBatchDataLoader(
        micro_batch_size=config["training"]["micro_batch_size"],
        seq_length=config["training"]["seq_length"],
        dataset_name=config["dataset"]["name"],
        tokenizer_name=config["model"]["name"],
        grad_acc_steps=config["training"]["gradient_accumulation_steps"],
        device=device,
        num_workers=config["dataset"]["num_workers"],
        num_proc=config["dataset"]["num_proc"],
        num_samples=config["training"].get("num_samples", None),
        subset_name=config["dataset"].get("subset_name", None),
        split=config["dataset"].get("split", "train")
    )

    # 在第一个 rank 下载模型
    if pgm.process_group_manager.global_rank == 0:
        download_model(config["model"]["name"], os.environ["HF_TOKEN"])

    dist.barrier()

    print(f"Dataloader initialization time: {time.time()-start_time:.2f}s", is_print_rank=is_wandb_rank)
    
    tokens_per_step = data_loader.global_batch_size * config["training"]["seq_length"]
    
    if pgm.process_group_manager.global_rank == 0:
        print(f"Tokens per step: {to_readable_format(tokens_per_step)}", is_print_rank=is_wandb_rank)

    # 初始化 wandb
    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.init(
            project="tron_support",
            name=f"{config['logging']['run_name']}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
                "max_tokens": config["training"]["max_tokens"],
                "learning_rate": config["training"]["learning_rate"],
                "seed": config["training"]["seed"],
                "micro_batch_size": data_loader.micro_batch_size,
                "global_batch_size": data_loader.global_batch_size,
                "gradient_accumulation": data_loader.grad_acc_steps,
            },
        )

    # 创建模型配置
    if pgm.process_group_manager.global_rank == 0:
        print(f"rank {pgm.process_group_manager.global_rank}: Creating model config")
        model_config = AutoConfig.from_pretrained(config["model"]["name"])
        
        # 从配置文件覆盖模型参数
        for key in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "vocab_size", "hidden_size", "intermediate_size", 
                    "max_position_embeddings", "rms_norm_eps", "rope_theta"]:
            if key in config["model"]:
                setattr(model_config, key, config["model"][key])
        
        model_config.max_position_embeddings = config["training"]["seq_length"]
        objects = [model_config]
    else:
        objects = [None]

    # 广播模型配置到所有 rank
    dist.broadcast_object_list(objects, src=0, device=device)
    model_config = objects[0]
    print(f"rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks", 
          is_print_rank=(pgm.process_group_manager.global_rank == 0))

    dist.barrier()

    print(f"rank {pgm.process_group_manager.global_rank}: Initializing model on meta device", 
          is_print_rank=is_wandb_rank)

    start_time = time.time()

    # 在 meta 设备上初始化模型（不分配内存）
    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        # 应用张量并行
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

    # 从 safetensors 加载权重
    model = init_model_with_materialized_weights(
        model, 
        model_config, 
        save_dir=f"./hf_model_safetensors/"
    )

    # 转换到目标数据类型和设备
    model.to(dtype).to(device)
    
    # 应用数据并行
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)
    
    print(f"Model initialization time: {time.time()-start_time:.2f}s", is_print_rank=is_wandb_rank)
    
    model.train()
    num_params = get_num_params(model)
    print(f"Number of parameters: {to_readable_format(num_params)}", is_print_rank=is_wandb_rank)
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager()

    trained_tokens, step = 0, 0
    
    # 加载检查点（如果有）
    if config["checkpoint"]["load_path"]:
        step, trained_tokens = checkpoint_manager.load_checkpoint(
            model, optimizer, config["checkpoint"]["load_path"]
        )
    
    dist.barrier()
    
    # 训练循环
    while config["training"]["max_tokens"] is None or trained_tokens < config["training"]["max_tokens"]:
        step_start_time = time.time()
        optimizer.zero_grad()
        
        # 执行训练步骤
        loss = train_step(model, data_loader, device)
        
        # 在 DP ranks 之间平均损失（用于日志）
        loss = average_loss_across_dp_cp_ranks(loss, device)
        
        # 更新参数
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        # 重置 DataParallel 的 bucket manager
        if hasattr(model, 'reset'):
            model.reset()

        # 计算性能指标
        step_duration = time.time() - step_start_time
        tokens_per_second = tokens_per_step / step_duration
        tokens_per_second_per_gpu = tokens_per_second / world_size
        mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)
        
        # 打印和记录日志
        if is_wandb_rank:
            max_tokens_str = f"/{to_readable_format(config['training']['max_tokens'])}" if config['training']['max_tokens'] else ""
            
            print(
                f"[rank {pgm.process_group_manager.global_rank}] "
                f"Step: {step:<5d} | "
                f"Loss: {loss:6.4f} | "
                f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                f"Tokens: {to_readable_format(trained_tokens):>7s}{max_tokens_str} | "
                f"MFU: {mfu:5.2f}% | "
                f"Memory: {torch.cuda.memory_reserved() / 1e9:6.2f}GB",
                is_print_rank=is_wandb_rank
            )
        
            if config["logging"]["use_wandb"]:
                wandb.log({
                    "loss": loss,
                    "tokens_per_step": tokens_per_step,
                    "tokens_per_second": tokens_per_second,
                    "mfu": mfu,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "memory_usage": torch.cuda.memory_reserved() / 1e9,
                    "trained_tokens": trained_tokens,
                    "step": step
                })
        
        # 保存检查点
        if step % config["checkpoint"]["save_frequency"] == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, step, trained_tokens, 
                config["checkpoint"]["save_dir"] + f"/{step}"
            )
        
        # 检查是否达到最大训练步数
        if step >= config["training"]["total_train_steps"]:
            break
    
    # 清理
    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.finish()

    dist.destroy_process_group()
    
    print("Training completed successfully!", is_print_rank=is_wandb_rank)
