"""
支持 vLLM 推理的 Transformer Block
保持原有的 Pre-Norm 结构
"""
import torch
from torch import nn
from vllm_support.vllm_attention import PagedCausalMultiHeadAttention
from rmsnorm import RMSNorm
from swiGLU import SwiGLU, SiLUFFN

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
        dtype=None,
        use_rms_norm=True, norm_model="pre", ffn_type="swiglu",
    ):
        super().__init__()
        if norm_model not in ("pre", "post") or ffn_type not in ("swiglu", "silu"):
            raise ValueError("无效的归一化位置或前馈类型")
        self.norm_model = norm_model

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
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if use_rms_norm else nn.Identity()
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if use_rms_norm else nn.Identity()

        # 前馈网络（SwiGLU）
        self.ffn = (SwiGLU if ffn_type == "swiglu" else SiLUFFN)(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        x_position: torch.Tensor,
        # vLLM 参数
        is_prefill: bool = True,
        block_tables: torch.Tensor = None,
        slot_mapping: torch.Tensor = None,
        context_lens: torch.Tensor = None,
    ):
        """
        Pre-Norm Transformer Block
        
        参数：
            x: 输入 [batch, seq_len, d_model]
            x_position: 位置索引
            is_prefill: 是否为 Prefill 阶段
            block_tables: 物理块映射表
            slot_mapping: Token 到槽位的映射
            context_lens: 每个序列的上下文长度
        """
        # 1. Attention 子层（Pre-Norm）
        x = x + self.attention(
            self.ln1(x) if self.norm_model == "pre" else x,
            token_position=x_position,
            is_prefill=is_prefill,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
        )

        if self.norm_model == "post":
            x = self.ln1(x)
        # 前馈子层与稠密模型使用相同的归一化顺序。
        x = x + self.ffn(self.ln2(x) if self.norm_model == "pre" else x)
        if self.norm_model == "post":
            x = self.ln2(x)

        return x

    def clear_cache(self):
        """清空 KV Cache"""
        self.attention.clear_cache()

    def truncate_cache(self, length: int):
        """截断 KV Cache"""
        raise NotImplementedError("分页缓存需要通过序列块表管理截断")

    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return 0


# 兼容性别名
TransformerBlock = PagedTransformerBlock
