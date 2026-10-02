from typing import List
import torch
import torch.distributed as dist
from torch import nn

class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        "初始化参数分组、梯度存储和用于同步的通信组。"
        self.params = set(params)    # 当前桶包含的参数集合
        self.params_with_grad_ready = set() # 记录梯度已经就绪的参数，全部就绪后启动归约
        self.grad_data = grad_data  # 保存当前桶中全部参数梯度的张量
        self.process_group = process_group  # 同步梯度使用的通信组
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.handle = None # 异步归约操作的句柄

        self.reset()

    def sync_gradient(self) -> None:
        "启动异步归约以同步当前桶的梯度。"
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)

    def reset(self) -> None:
        "清空累计梯度与内部同步状态。"
        self.handle = None
        self.params_with_grad_ready.clear() # 清空已经就绪的参数集合
        self.grad_data.zero_() # 将梯度张量清零

    def wait(self) -> None:
        "等待所有异步梯度归约完成。"
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        self.handle.wait() # 等待归约操作完成

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        "标记参数的梯度已经就绪，全部就绪后启动同步。"
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        # 桶内所有参数的梯度就绪后同步梯度
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        "初始化参数分组、梯度存储和用于同步的通信组。"
        self.params = list(params) # 将参数迭代器转换为列表
        self.device = self.params[0].device if self.params[0].is_cuda else torch.device("cpu")
        self.buckets = [] # 梯度桶列表
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.params_to_bucket_location = {} # 记录每个参数在梯度桶中的区间和桶编号
        self.bucket_size = bucket_size
        self.bucket_sizes = None # 每个梯度桶的实际大小
        self.grad_data_list = [] # 保存每个桶梯度的张量列表
        self.grad_type = grad_type
        # 按指定大小将梯度分组到各桶
        self._initialize_buckets()


    def _initialize_buckets(self) -> None:
        "按指定容量分配梯度桶及参数的梯度视图。"
        cur_bucket_size = 0
        cur_bucket_idx = 0

        # 将参数分配到梯度桶
        for param in self.params:
            if not param.requires_grad:
                continue

            # 空桶直接接收当前参数
            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
                continue

            # 当前桶容不下此参数时创建新桶
            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
            else:
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                cur_bucket_size += param.numel()

        # 收集桶的大小以及每个桶包含的参数
        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)

        # 分配梯度存储并初始化梯度桶对象
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device=self.device))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))

        # 为每个参数创建对应的梯度视图
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            # 使用 param.main_grad 保存累计梯度
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)

    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        "根据起止索引和形状创建梯度张量视图。"
        return tensor[start:end].view(shape)

    def reset(self) -> None:
        "清空累计梯度与内部同步状态。"
        for bucket in self.buckets:
            bucket.reset()

    def wait(self) -> None:
        "等待所有异步梯度归约完成。"
        for bucket in self.buckets:
            bucket.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        "标记参数的梯度已经就绪，全部就绪后启动同步。"
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)


