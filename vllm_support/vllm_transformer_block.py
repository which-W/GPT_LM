"""
支持 vLLM 推理的 Transformer Block
保持原有的 Pre-Norm 结构
"""
import torch
from torch import nn
from vllm_attention import PagedCausalMultiHeadAttention
from rmsnorm import RMSNorm
from swiGLU import SwiGLU

class PagedTransformerBlock(nn.Module):
    """
    支持 PagedAttention 的 Transformer Block
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        max_seq_len: int,
        theta: float,
        # vLLM 参数
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # 注意力模块（支持 PagedAttention）
        self.attention = PagedCausalMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            num_kv_blocks=num_kv_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        
        # RMSNorm 层
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # 前馈网络（SwiGLU）
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        x_position: torch.Tensor,
        # vLLM 参数
        is_prefill: bool = True,
        block_tables: torch.Tensor = None,
        slot_mapping: torch.Tensor = None,
        context_lens: torch.Tensor = None,
        #简单cache
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        """
        Pre-Norm Transformer Block
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            x_position: 位置索引
            is_prefill: 是否为 Prefill 阶段
            block_tables: 物理块映射表
            slot_mapping: Token 到槽位的映射
            context_lens: 每个序列的上下文长度
        """
        # 1. Attention 子层（Pre-Norm）
        x = x + self.attention(
            self.ln1(x),
            token_position=x_position,
            is_prefill=is_prefill,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            use_cache=use_cache,
            start_pos=start_pos
        )
        
        # 2. FFN 子层
        x = x + self.ffn(self.ln2(x))
        
        return x
    
    def clear_cache(self):
        """清空 KV Cache"""
        self.attention.clear_cache()
    
    def truncate_cache(self, length: int):
        """截断 KV Cache"""
        pass
    
    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return 0


# 兼容性别名
TransformerBlock = PagedTransformerBlock