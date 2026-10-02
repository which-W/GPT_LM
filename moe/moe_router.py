"""
MoE Gating/Router Module
实现Top-K路由机制，支持负载均衡和辅助损失
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MoERouter(nn.Module):
    """
    MoE路由器/门控网络

    功能:
    1. 为每个token计算对所有专家的得分
    2. 选择Top-K个专家
    3. 计算门控权重
    4. 支持负载均衡损失(类似DeepSeek V2)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 3,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError('top_k must lie between 1 and n_experts')
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight

        # 门控网络: 线性层将hidden state映射到专家得分
        self.gate = nn.Linear(
            d_model,
            n_experts,
            bias=False,
            device=device,
            dtype=dtype
        )

        # 用于存储辅助损失
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数：
            x: [batch_size, seq_len, d_model] 输入张量
        
        返回：
            gates: [batch_size, seq_len, top_k] Top-K专家的归一化权重
            expert_indices: [batch_size, seq_len, top_k] Top-K专家的索引
            aux_loss: 标量,负载均衡损失(如果启用)
        """
        batch_size, seq_len, d_model = x.shape

        # 计算门控logits: [batch_size, seq_len, n_experts]
        gate_logits = self.gate(x)

        # 应用softmax得到概率分布
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 选择Top-K专家
        # 张量形状或计算公式：topk_gates: [batch_size, seq_len, top_k]
        # 张量形状或计算公式：topk_indices: [batch_size, seq_len, top_k]
        topk_gates, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # 对Top-K的权重进行重新归一化
        gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

        # 计算辅助损失(负载均衡)
        if self.use_aux_loss and self.training:
            self.aux_loss = self._compute_aux_loss(gate_probs)
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        return gates, topk_indices, self.aux_loss

    def _compute_aux_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡辅助损失
        
        目标: 让每个专家处理大致相同数量的token
        使用方法类似DeepSeek: 计算专家使用频率和门控概率之间的损失
        
        参数：
            gate_probs: [batch_size, seq_len, n_experts] 所有专家的门控概率
        
        返回：
            aux_loss: 标量损失
        """
        # 计算每个专家被选中的平均概率
        # 张量形状或计算公式：[n_experts]
        mean_gate_probs = gate_probs.mean(dim=[0, 1])
        assignments = F.one_hot(gate_probs.topk(self.top_k, dim=-1).indices,
                                num_classes=self.n_experts).float()
        load = assignments.mean(dim=(0, 1, 2))
        return self.aux_loss_weight * self.n_experts * (load * mean_gate_probs).sum()



class TopKRouter(nn.Module):
    """
    简化版Top-K路由器
    直接选择得分最高的K个专家,不计算辅助损失
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            x: [batch_size, seq_len, d_model]
        
        返回：
            gates: [batch_size, seq_len, top_k] 归一化权重
            expert_indices: [batch_size, seq_len, top_k] 专家索引
        """
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_gates, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

        return gates, topk_indices
