"""
专家并行 MoE Layer
整合路由器和多GPU专家网络
"""
import torch
from torch import nn
from typing import Optional, List
from moe_router import MoERouter
from moe_experts import ExpertParallelMoE


class ExpertParallelMoELayer(nn.Module):
    """
    专家并行MoE层
    
    组成:
    - Router: 在主GPU上，负责为每个token选择专家
    - Experts: 分布在多张GPU上，每个专家在不同的GPU
    
    使用场景:
    - 专家数量 <= GPU数量
    - 想要最大化GPU利用率
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,  # 主设备，路由器放在这里
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            d_model: 隐藏层维度
            d_ff: FFN中间层维度
            n_experts: 专家数量
            top_k: 每个token激活的专家数
            use_aux_loss: 是否使用负载均衡损失
            aux_loss_weight: 辅助损失权重
            device_ids: GPU设备列表，如 [0, 1, 2, 3]
            main_device: 主设备ID，路由器会放在这个GPU上
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.main_device = torch.device(f"cuda:{main_device}")
        
        # 路由器放在主GPU上
        self.router = MoERouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
            device=self.main_device,
            dtype=dtype
        )
        
        # 创建多GPU专家系统
        self.experts = ExpertParallelMoE(
            n_experts=n_experts,
            d_model=d_model,
            d_ff=d_ff,
            device_ids=device_ids,
            dtype=dtype
        )
        
        self.aux_loss = None
        
        print(f"\nExpertParallelMoELayer 初始化完成")
        print(f" 路由器在: cuda:{main_device}")
        print(f" Top-K: {top_k}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 输入张量
        
        Returns:
            output: [batch_size, seq_len, d_model] MoE输出
        """
        batch_size, seq_len, d_model = x.shape
        original_device = x.device
        
        # 确保输入在主设备上进行路由
        if x.device != self.main_device:
            x_route = x.to(self.main_device)
        else:
            x_route = x
        
        # 路由: 为每个token选择Top-K专家
        # Router不需要调整，它只返回索引
        gates, expert_indices, aux_loss = self.router(x_route)
        self.aux_loss = aux_loss
        
        # gates: [batch_size, seq_len, top_k]
        # expert_indices: [batch_size, seq_len, top_k]
        
        # 对每个Top-K位置分别处理
        output = torch.zeros_like(x_route)
        
        for k in range(self.top_k):
            # 获取第k个专家的索引和权重
            k_expert_idx = expert_indices[:, :, k]  # [batch_size, seq_len]
            k_gates = gates[:, :, k:k+1]  # [batch_size, seq_len, 1]
            
            # 通过多GPU专家网络
            # 专家会自动处理跨GPU传输
            k_output = self.experts(x_route, k_expert_idx)  # [batch_size, seq_len, d_model]
            
            # 加权累加
            output = output + k_gates * k_output
        
        # 4. 如果需要，将输出移回原设备
        if original_device != self.main_device:
            output = output.to(original_device)
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失"""
        return self.aux_loss if self.aux_loss is not None else torch.tensor(0.0)


if __name__ == "__main__":
    # 检查GPU
    n_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {n_gpus}")
    
    if n_gpus < 2:
        print("⚠️ 需要至少2张GPU")
    else:
        # 创建MoE层
        moe_layer = ExpertParallelMoELayer(
            d_model=512,
            d_ff=2048,
            n_experts=8,
            top_k=2,
            device_ids=list(range(min(4, n_gpus))),
            main_device=0,
        )
        
        # 测试前向传播
        print("\n测试前向传播:")
        batch_size = 4
        seq_len = 128
        
        x = torch.randn(batch_size, seq_len, 512).cuda(0)
        
        print(f"  输入shape: {x.shape}")
        print(f"  输入device: {x.device}")
        
        with torch.no_grad():
            output = moe_layer(x)
        
        print(f"  输出shape: {output.shape}")
        print(f"  输出device: {output.device}")
        
        aux_loss = moe_layer.get_aux_loss()
        print(f"  辅助损失: {aux_loss.item():.6f}")
        
        print("\n✓ 测试完成!")
        
        # 显存使用情况
        print("\n显存使用:")
        for i in range(min(4, n_gpus)):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB")