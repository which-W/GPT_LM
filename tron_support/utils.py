import os
import random
import numpy as np
import builtins
try:
    import fcntl
except ImportError:
    fcntl = None
import glob

import huggingface_hub

import tron_support.process_group_manager as pgm
import torch, torch.distributed as dist

def print(*args, is_print_rank=True, **kwargs):
    "在支持文件锁的平台避免多个进程的打印内容交错。"
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

# 参考来源：
# 参考来源：https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289
# 参考来源：https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L950
def get_mfu(tokens_per_second, num_params, model_config, theoretical_flops = 989.5 * 10 ** 12):
    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100 # 百分比
    return mfu

def get_num_params(model):
    "统计总参数量：张量并行分片乘以进程数，重复保存的参数只计一次，各流水线阶段求和。"
    tp_world_size = pgm.process_group_manager.tp_world_size

    # 统计当前流水线阶段的参数
    local_num_params = 0
    for name, param in model.named_parameters():
        # 这些参数由张量并行进程分片保存
        # 序列并行时需另行处理归一化参数的统计
        if any(tp_keyword in name.lower() for tp_keyword in ['attention', 'mlp', 'embed', 'final_proj']):
            local_num_params += param.numel() * tp_world_size
        else:
            # 归一化和偏置等参数在各张量并行进程中重复保存
            local_num_params += param.numel()

    # 收集各流水线阶段的参数量
    param_counts = torch.tensor(local_num_params, device=next(model.parameters()).device)

    # 对各流水线阶段的参数量求和
    dist.all_reduce(param_counts, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.pp_group)

    return param_counts.item()

def assert_no_meta_tensors(model):
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")

    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")

    assert len(meta_tensors) == 0, f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(meta_tensors)

def average_loss_across_dp_cp_ranks(loss, device):
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    if pgm.process_group_manager.pp_is_last_stage:
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
        reduced_loss /= pgm.process_group_manager.cp_dp_world_size
    return reduced_loss.item()

def model_cache_dir(model_name):
    """按模型名称隔离下载缓存，防止加载另一个模型的权重。"""
    return os.path.join("hf_model_safetensors", model_name.replace("/", "--"))


def download_model(model_name, hf_token=None):
    dst = model_cache_dir(model_name)
    os.makedirs(dst, exist_ok=True)
    files = glob.glob(os.path.join(dst, "*.safetensors"))
    index_path = os.path.join(dst, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path, encoding="utf-8") as f:
            complete = all(os.path.exists(os.path.join(dst, part)) for part in json.load(f)["weight_map"].values())
    else:
        complete = os.path.exists(os.path.join(dst, "model.safetensors"))
    if complete and files and os.path.exists(os.path.join(dst, "config.json")):
        return dst
    huggingface_hub.snapshot_download(model_name, repo_type="model", local_dir=dst,
                                      token=hf_token, allow_patterns=["*.safetensors", "*.json"])
    if not glob.glob(os.path.join(dst, "*.safetensors")):
        raise ValueError(f"模型 {model_name} 没有 safetensors 权重")
    return dst
