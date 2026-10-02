import torch
import torch.distributed as dist
import contextlib
from torch import nn
from torch.autograd import Variable

from tron_support.data_parallel_v.bucket import BucketManager
import tron_support.process_group_manager as pgm

class DataParallelNaive(nn.Module):
    "通过归约同步参数梯度的基础数据并行封装，支持临时关闭同步。"
    def __init__(self, module):
        "初始化参数分组、梯度存储和用于同步的通信组。"
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # 控制反向传播时是否同步梯度，累积期间关闭同步
        self.register_backward_hook(self._allreduce_grads)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def register_backward_hook(self, hook):
        "为可训练参数注册梯度累加与同步回调。"
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_post_accumulate_grad_hook(hook)

    def _allreduce_grads(self, grad):
        "在通信组内归约并平均参数梯度。"
        # 只在梯度累积的最后一步进行同步
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad

    @contextlib.contextmanager
    def no_sync(self):
        "在梯度累积期间临时关闭同步。"
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

class DataParallelBucket(nn.Module):
    "将梯度分组到桶中，减少数据并行的通信次数。"
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        "初始化参数分组、梯度存储和用于同步的通信组。"
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # 控制反向传播时是否同步梯度，累积期间关闭同步
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # 单精度梯度的每个元素占四字节
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # 一个桶包含的梯度元素数量
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # 记录是否已经注册等待梯度同步的回调

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)

    def register_backward_hook(self):
        "为可训练参数注册梯度累加与同步回调。"
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # 创建视图以获取梯度函数
                param_tmp = param.expand_as(param)
                # 获取梯度累加节点
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        "创建参数梯度回调，负责累加梯度并标记同步就绪。"
        def param_hook(*unused):
            "累加当前梯度，注册等待同步完成的回调，并将参数标记为就绪。"
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # 累计梯度
                param.grad = None

                # 梯度累积或流水线微批次阶段暂不执行同步
                if self.require_backward_grad_sync:
                    # 每次反向传播只注册一次等待梯度同步的回调
                    # 回调在反向传播后执行，每次反向传播都需要重新注册
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True

                    # 将当前参数标记为梯度同步已就绪
                    bucket_manager.mark_param_as_ready(param)
        return param_hook

    @contextlib.contextmanager
    def no_sync(self):
        "在梯度累积期间临时关闭同步。"
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

    def _post_backward(self):
        "反向传播结束后等待同步完成，并把梯度复制回参数供优化器使用。"
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # 将同步后的梯度复制到参数，供优化器更新
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # 参数与其梯度必须使用相同的数据类型

    def reset(self):
        "清空累计梯度与内部同步状态。"
        self.bucket_manager.reset()
