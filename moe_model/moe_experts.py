"""
专家并行 MoE - 简化版
每个专家分配到一张GPU上
"""
import torch
from torch import nn
from typing import Optional, List, Dict
from swiGLU import SwiGLU


class Expert(nn.Module):
    """单个专家网络"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ExpertParallelMoE(nn.Module):
    """
    专家并行MoE
    每个专家分配到不同的GPU上
    适合: 专家数 <= GPU数
    示例:
        8个专家 + 4张GPU:
        - GPU 0: Expert 0, 4
        - GPU 1: Expert 1, 5
        - GPU 2: Expert 2, 6
        - GPU 3: Expert 3, 7
    """
    
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        device_ids: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            n_experts: 专家数量
            d_model: 模型维度
            d_ff: FFN维度
            device_ids: GPU设备列表，如 [0, 1, 2, 3]
                       如果为None，自动检测所有可用GPU
        """
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        
        # 检测可用GPU
        if device_ids is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise ValueError("没有可用的GPU设备")
            device_ids = list(range(n_gpus))
        
        self.device_ids = device_ids
        self.n_devices = len(device_ids)
        
        # 为每个专家分配GPU (轮询分配)
        self.expert_to_device = {}
        for expert_id in range(n_experts):
            device_idx = self.device_ids[expert_id % self.n_devices]
            self.expert_to_device[expert_id] = device_idx
        
        # 创建专家，每个专家在指定的GPU上
        self.experts = nn.ModuleList()
        for expert_id in range(n_experts):
            device = torch.device(f"cuda:{self.expert_to_device[expert_id]}")
            expert = Expert(d_model, d_ff, device=device, dtype=dtype)
            self.experts.append(expert)
        
        # 打印分配情况
        print(f"专家并行MoE初始化:")
        print(f"{n_experts}个专家分布在{self.n_devices}张GPU上")
        
        # 统计每张GPU上的专家数
        gpu_expert_count = {}
        for expert_id, device_id in self.expert_to_device.items():
            gpu_expert_count[device_id] = gpu_expert_count.get(device_id, 0) + 1
        
        for device_id in sorted(gpu_expert_count.keys()):
            experts_on_gpu = [e for e, d in self.expert_to_device.items() if d == device_id]
            print(f"  - GPU {device_id}: {gpu_expert_count[device_id]}个专家 {experts_on_gpu}")
    
    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model] 输入 (在某个GPU上)
            expert_idx: [batch_size, seq_len] 每个token分配的专家索引
        
        Returns:
            output: [batch_size, seq_len, d_model] 输出 (在原GPU上)
        """
        batch_size, seq_len, d_model = x.shape
        input_device = x.device
        
        # 展平
        x_flat = x.reshape(-1, d_model)
        expert_idx_flat = expert_idx.reshape(-1)
        
        # 初始化输出 (在输入设备上)
        output = torch.zeros_like(x_flat)
        
        # 对每个专家处理
        for expert_id in range(self.n_experts):
            # 找到分配给当前专家的token
            expert_mask = (expert_idx_flat == expert_id)
            
            if expert_mask.any():
                # 提取token
                expert_input = x_flat[expert_mask]
                
                # 将数据移动到专家所在的GPU
                expert_device = torch.device(f"cuda:{self.expert_to_device[expert_id]}")
                expert_input = expert_input.to(expert_device)
                
                # 在专家GPU上计算
                expert_output = self.experts[expert_id](expert_input)
                
                # 将结果移回输入设备
                expert_output = expert_output.to(input_device)
                
                # 写入输出
                output[expert_mask] = expert_output
        
        # 恢复形状
        output = output.reshape(batch_size, seq_len, d_model)
        return output

if __name__ == "__main__":
    # 检查GPU数量
    n_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {n_gpus}")
    
    if n_gpus < 2:
        print("⚠️ 需要至少2张GPU才能测试专家并行")
    else:
        # 创建专家并行MoE
        moe = ExpertParallelMoE(
            n_experts=8,
            d_model=512,
            d_ff=2048,
            device_ids=list(range(min(4, n_gpus)))  # 最多使用4张GPU
        )
        
        # 测试前向传播
        print("\n测试前向传播:")
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, 512).cuda(0)
        expert_idx = torch.randint(0, 8, (batch_size, seq_len)).cuda(0)
        
        print(f"  输入shape: {x.shape} on {x.device}")
        
        with torch.no_grad():
            output = moe(x, expert_idx)
        
        print(f"  输出shape: {output.shape} on {output.device}")
        print(f"\n✓ 测试通过!")
        
        # 显存使用
        print("\n显存使用:")
        for i in range(min(4, n_gpus)):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB")