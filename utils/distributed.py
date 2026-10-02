"""统一 torchrun 和手动启动时的进程编号、设备及集合通信。"""
import os
from datetime import timedelta
import torch
import torch.distributed as dist


def setup_distributed(rank, world_size, backend="nccl"):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if backend == "nccl":
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("当前环境不支持 NCCL，可使用 --backend gloo 进行 CPU 验证")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    options = {}
    if os.name == "nt":
        # Windows 的部分 PyTorch 构建缺少 libuv；显式建立原生 TCP 存储。
        launched = "TORCHELASTIC_RUN_ID" in os.environ
        options["store"] = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                                        world_size, rank == 0 and not launched,
                                        timeout=timedelta(seconds=120), use_libuv=False)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **options)
    return device


def mean_all_reduce(tensor):
    """所有进程按相同顺序求和，再计算平均值，兼容 Gloo。"""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


@torch.no_grad()
def broadcast_model(model, communication_device):
    """将主进程的初始权重及缓冲区同步到全部模型副本。"""
    for tensor in list(model.parameters()) + list(model.buffers()):
        staged = tensor.detach().to(communication_device).contiguous()
        dist.broadcast(staged, src=0)
        tensor.copy_(staged.to(tensor.device))


def sync_moe_gradients(model, communication_device):
    """未被路由到的专家也参与通信，避免各进程集合调用数量不同。"""
    for param in model.parameters():
        present = torch.tensor(int(param.grad is not None), device=communication_device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if not present.item():
            continue
        staged = (torch.zeros_like(param) if param.grad is None else param.grad).to(communication_device).contiguous()
        mean_all_reduce(staged)
        if param.grad is None:
            param.grad = staged.to(param.device)
        else:
            param.grad.copy_(staged.to(param.device))
