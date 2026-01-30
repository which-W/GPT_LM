"""
MoE Transformer Block
将标准Transformer Block中的FFN替换为MoE层
"""
import torch
from torch import nn
from typing import Optional, List
from attention import CauseMutiHeadAttention
from rmsnorm import RMSNorm
from moe_layer import ExpertParallelMoELayer


class MoETransformerBlock(nn.Module):
    """
    MoE版本的Transformer Block
    
    结构:
    1. Pre-norm架构
    2. 因果多头注意力
    3. MoE前馈网络(替代标准FFN)
    4. 残差连接
    
    相比标准Block的改进:
    - 使用MoE层提升模型容量
    - 保持计算效率(仅激活部分专家)
    - 支持负载均衡
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        max_seq_len: int,
        theta: float,
        # MoE特定参数
        n_experts: int = 8,
        top_k: int = 2,
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        # 通用参数
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN中间层维度
            n_head: 注意力头数
            max_seq_len: 最大序列长度
            theta: RoPE的theta参数
            n_experts: 专家数量
            top_k: 每个token激活的专家数
            use_moe_aux_loss: 是否使用MoE负载均衡损失
            moe_aux_loss_weight: MoE辅助损失权重
            device_ids: GPU设备列表
            main_device: 主GPU设备ID
        """
        super().__init__()
        
        main_dev = torch.device(f"cuda:{main_device}")
        
        # Attention在主GPU上
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device=main_dev,
            dtype=dtype
        )
        
        # 两个RMSNorm层
        self.ln1 = RMSNorm(d_model=d_model, device=main_dev, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=main_dev, dtype=dtype)
        
        # MoE在多GPU上
        self.moe = ExpertParallelMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            top_k=top_k,
            use_aux_loss=use_moe_aux_loss,
            aux_loss_weight=moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=main_device,
            dtype=dtype
        )
        
        # 存储配置
        self.n_experts = n_experts
        self.top_k = top_k
    
    def forward(
        self, 
        x: torch.Tensor, 
        x_position: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model] 输入张量
            x_position: [batch_size, seq_len] token位置索引
            use_cache: 是否使用KV缓存
            start_pos: 起始位置
        
        Returns:
            output: [batch_size, seq_len, d_model] 输出张量
        """
        # 1. Attention子层 (pre-norm结构)
        # 残差连接 + LayerNorm + Attention
        x = x + self.attention(self.ln1(x), token_position=x_position, use_cache=use_cache, start_pos=start_pos)
        
        # 2. MoE FFN子层
        # 残差连接 + LayerNorm + MoE
        x = x + self.moe(self.ln2(x))
        
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        获取MoE的辅助损失
        在训练时应该将此损失加到总损失中
        
        Returns:
            aux_loss: 标量张量
        """
        return self.moe.get_aux_loss()
    
    def clear_cache(self):
        """清空该层的 KV Cache"""
        self.attention.clear_cache()
    
    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return self.attention.get_cache_seq_len()


class HybridTransformerBlock(nn.Module):
    """
    混合Transformer Block
    
    特点:
    - 可以在标准FFN和MoE之间切换
    - 用于逐层配置(例如某些层用MoE,某些层用标准FFN)
    - 类似DeepSeek V2的稀疏层策略
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        max_seq_len: int,
        theta: float,
        # FFN类型选择
        use_moe: bool = False,
        # MoE参数
        n_experts: int = 8,
        top_k: int = 2,
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数 (仅MoE需要)
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        # 通用参数
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        self.use_moe = use_moe
        
        # 注意力模块
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        
        # LayerNorm
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # 根据use_moe选择FFN类型
        if use_moe:
            self.ffn = ExpertParallelMoELayer(
                d_model=d_model,
                d_ff=d_ff,
                n_experts=n_experts,
                top_k=top_k,
                use_aux_loss=use_moe_aux_loss,
                aux_loss_weight=moe_aux_loss_weight,
                device_ids=device_ids,
                main_device=main_device,
                dtype=dtype
            )
        else:
            from swiGLU import SwiGLU
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        x_position: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        # Attention子层
        x = x + self.attention(self.ln1(x), token_position=x_position, use_cache=use_cache, start_pos=start_pos)
        
        # FFN子层
        x = x + self.ffn(self.ln2(x))
        
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """只有MoE层才有辅助损失"""
        if self.use_moe and hasattr(self.ffn, 'get_aux_loss'):
            return self.ffn.get_aux_loss()
        return torch.tensor(0.0)

    def clear_cache(self):
        """清空该层的 KV Cache"""
        self.attention.clear_cache()
    
    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return self.attention.get_cache_seq_len()