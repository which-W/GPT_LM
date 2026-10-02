import torch.distributed as dist
import torch
import tron_support.process_group_manager as pgm
import torch.nn.functional as F

from typing import Tuple

def merge_first_two_dims(grad_output: torch.Tensor, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    "合并张量的前两个维度。"
    return grad_output.contiguous().view(-1, *grad_output.shape[2:]), input_.contiguous().view(-1, *input_.shape[2:])

def split_tensor_along_last_dim(tensor, num_partitions):
    "沿最后一维将张量均匀切分。"
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)
#用于column切分时，前向copy,反向all-reduce
class CopyToModelParallelRegion(torch.autograd.Function):
    "前向直接返回输入，反向对梯度求和归约；对应论文中的 f：https://arxiv.org/abs/1909.08053"
    @staticmethod
    def forward(ctx, x):
        # 前向传播: 什么都不做,直接返回
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播: all-reduce梯度
        if pgm.process_group_manager.tp_world_size == 1:
          return grad_output # 单GPU不需要通信
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group) #ReduceOp.SUM 求和操作
        return grad_output
#用于row切分时，前向all-reduce,反向copy
class ReduceFromModelParallelRegion(torch.autograd.Function):
    "前向对各分片求和归约，反向直接传递梯度；对应论文中的 g：https://arxiv.org/abs/1909.08053"
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        #前向传播需要reduce
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #反向传播直接输出
        return grad_output
#用于最后一层,需要得到完整的vocab输出，前向gather,反向split
class GatherFromModelParallelRegion(torch.autograd.Function):
    "前向汇集各分片，反向按进程编号切分梯度。"
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        last_dim = x.dim() - 1 ## 最后一维
        # 集合通信要求张量连续，参考：https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        x = x.contiguous() #保持内存连续
        # 创建空的tensor列表
        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x #填入自己的数据
        # all-gather: 每个GPU收集所有GPU的数据
        dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)
        # 现在每个GPU都有 [tensor_gpu0, tensor_gpu1, ...]
        output = torch.cat(tensor_list, dim=last_dim).contiguous()  # 沿最后一维拼接
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # 按张量并行进程数切分梯度
         # 反向: 切分梯度
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        # 保存用于反向传播
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        # 标准的线性层计算
        output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        "先启动输入梯度的异步归约，再计算权重和偏置梯度，以重叠计算和通信；仅用于列并行线性层。"
        input_, weight = ctx.saved_tensors
        grad_input = grad_output @ weight # 梯度计算：(b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
        # 张量形状或计算公式：grad_output: [batch, seq, out_size] = [2, 1024, 2048]
        # 张量形状或计算公式：weight: [out_size, in_size] = [2048, 4096]
        # 张量形状或计算公式：grad_input: [batch, seq, in_size] = [2, 1024, 4096]

        # all-reduce input gradient.  # 2. 立即启动异步all-reduce
        input_gradient_all_reduce_handle = dist.all_reduce(grad_input, group=pgm.process_group_manager.tp_group, async_op=True) #async_op=True异步通信打开

        # 合并前两个维度，以便执行矩阵乘法
        # 3. 趁通信进行,计算权重和偏置梯度
        # 先reshape: 把batch和seq合并
        grad_output, input_ = merge_first_two_dims(grad_output, input_)     # 张量形状：grad_output, input_: (b, s, out_size), (b, s, input_size) -> (b*s, out_size), (b*s, input_size)
        # 张量形状或计算公式：grad_output: [2, 1024, 2048] -> [2048, 2048]
        # 张量形状或计算公式：input_: [2, 1024, 4096] -> [2048, 4096]

        # 计算权重梯度
        grad_weight = grad_output.t() @ input_                              # 权重梯度计算：(out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
        # 计算偏置梯度
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        # 4. 等待all-reduce完成
        input_gradient_all_reduce_handle.wait()
        # 现在grad_input已经包含了所有GPU的梯度和
        return grad_input, grad_weight, grad_bias
#列并行层前向传播
def linear_with_all_reduce(x, weight, bias):
    # 步骤1: 应用CopyToModelParallelRegion
    input_parallel = CopyToModelParallelRegion.apply(x)
    # 前向: input_parallel = x (什么都不做)
    # 反向: 会all-reduce梯度

    # 步骤2: 标准的线性层计算
    output = F.linear(input_parallel, weight, bias) # 当前分片输出公式：Y_i = XW_i^T + b
    return output

def linear_with_async_all_reduce(x, weight, bias):
    return LinearWithAsyncAllReduce.apply(x, weight, bias)
