"提供支持张量并行的注意力计算。"

from typing import Optional, Tuple
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from softmax import StableSoftmax
from rope import RotaryPositionalEmbedding
from tron_support.tensor_parallel_v.tensor_parallel import ColumnParallelLinear,RowParallelLinear
# 导入进程组管理器以支持张量并行
try:
    import tron_support.process_group_manager as pgm
    from tron_support.tensor_parallel_v.tp_communications import (
        ReduceFromModelParallelRegion,
        GatherFromModelParallelRegion,
        linear_with_all_reduce,
        linear_with_async_all_reduce
    )
    TP_AVAILABLE = True
except ImportError:
    TP_AVAILABLE = False
    print("Warning: picotron not available, TP features disabled")

def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask: torch.Tensor = None
):
    "计算缩放点积注意力；查询、键和值的最后两维为序列和通道。"
    #获取d_k
    d_k = Q.size(-1)

    #计算相似度分数，形成打分表
    scores = torch.einsum('...nk,...mk -> ...nm',Q,K) / math.sqrt(d_k)
    #应用mask掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    #计算注意力权重（归一化）
    #dim=-1 对应的是每一个Q对于K的分布
    softmax = StableSoftmax(dim=-1)
    probs = softmax(scores)

    #加权求和得到输出
    output = torch.einsum('...nm, ...mk -> ...nk', probs ,V)

    return output


from attention import KVCache

class FlashAttentionWithTP(nn.Module):
    "支持张量并行、分组查询、旋转位置编码和键值缓存的注意力模块。"

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        max_seq_size: int = 4096,
        bias: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        use_tp: bool = True,
        async_all_reduce: bool = False,
        theta:int = 10000
    ):
        super().__init__()

        # 基础配置
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d_model // n_head
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.bfloat16
        self.use_tp = use_tp and TP_AVAILABLE and hasattr(pgm, "process_group_manager")

        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.head_dim,max_seq_size,device=device,interleaved=False)
        else:
            self.rope = None
        # 张量并行配置
        if self.use_tp:
            self.tp_world_size = pgm.process_group_manager.tp_world_size
            self.tp_rank = pgm.process_group_manager.tp_rank

            assert n_head % self.tp_world_size == 0, "n_head must be divisible by tp_world_size"
            assert self.n_kv_head % self.tp_world_size == 0, "n_kv_head must be divisible by tp_world_size"

            self.num_local_heads = n_head // self.tp_world_size
            self.num_local_kv_heads = self.n_kv_head // self.tp_world_size
        else:
            self.tp_world_size = 1
            self.tp_rank = 0
            self.num_local_heads = n_head
            self.num_local_kv_heads = self.n_kv_head

        # 初始化支持张量并行的投影层
        factory_kwargs = {"device": device, "dtype": dtype}

        if self.use_tp and self.tp_world_size > 1:
            # 使用张量并行的投影层
            self.q_proj = ColumnParallelLinear(
                d_model,
                n_head * self.head_dim,
                bias=bias,
                async_all_reduce=async_all_reduce
            )
            self.k_proj = ColumnParallelLinear(
                d_model,
                self.n_kv_head * self.head_dim,
                bias=bias,
                async_all_reduce=async_all_reduce
            )
            self.v_proj = ColumnParallelLinear(
                d_model,
                self.n_kv_head * self.head_dim,
                bias=bias,
                async_all_reduce=async_all_reduce
            )
            self.out_proj = RowParallelLinear(
                d_model,
                d_model,
                bias=bias
            )
        else:
            # 使用标准线性层
            self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(d_model, self.n_kv_head * self.head_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(d_model, self.n_kv_head * self.head_dim, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        # 键值缓存
        self.kv_cache = KVCache()

        self.reset_parameters()

    def reset_parameters(self):
        "初始化可学习参数。"
        if not self.use_tp or self.tp_world_size == 1:
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    k = 1 / module.in_features
                    bound = math.sqrt(k)
                    torch.nn.init.uniform_(module.weight, -bound, bound)
                    if module.bias is not None:
                        torch.nn.init.uniform_(module.bias, -bound, bound)

            _init_weights(self.q_proj)
            _init_weights(self.k_proj)
            _init_weights(self.v_proj)
            _init_weights(self.out_proj)

    def forward(
        self,
        x: torch.Tensor,
        token_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        "执行前向计算，返回与输入批次和序列维度对应的输出。"
        batch_size, seq_length, _ = x.size()

        # 将输入投影为查询、键和值
        q = self.q_proj(x)  # 张量形状或计算公式：[batch, seq, num_local_heads * head_dim]
        k = self.k_proj(x)  # 张量形状或计算公式：[batch, seq, num_local_kv_heads * head_dim]
        v = self.v_proj(x)  # 张量形状或计算公式：[batch, seq, num_local_kv_heads * head_dim]


        # 标准张量形状为 [batch, heads, seq, dim]
        q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        if self.rope is not None:
            if token_position is None:
                #默认生成从0开始的顺序位置
                #expand处理 Batch维度，不占用额外的内存
                token_position = torch.arange(seq_length,device=x.device).expand(batch_size,seq_length)

        #对Q,K进行旋转，V保持不动
            q = self.rope(q,token_position)
            k = self.rope(k,token_position)

        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)
        # 始终应用因果约束；用户提供的填充掩码只会进一步限制可见键。
        query_positions = torch.arange(start_pos if use_cache else 0,
                                       (start_pos if use_cache else 0) + seq_length, device=x.device)
        mask = torch.arange(k.size(2), device=x.device)[None, :] <= query_positions[:, None]
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape == (batch_size, k.size(2)):
                mask = mask[None, None] & attention_mask[:, None, None, :].bool()
            else:
                mask = mask & attention_mask.bool()

        # 分组查询时重复键和值，使其头数与查询一致
        if self.num_local_heads != self.num_local_kv_heads:
            repeat_factor = self.num_local_heads // self.num_local_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        # 张量形状或计算公式：out: [batch, heads, seq, dim]
        out = out.transpose(1, 2)  # 张量形状或计算公式：[batch, seq, heads, dim]

        # 合并注意力头
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim)

        # 输出投影
        out = self.out_proj(out)

        return out

    def clear_cache(self):
        "清空键值缓存。"
        self.kv_cache.clear()

    def get_cache_seq_len(self) -> int:
        "返回当前缓存的序列长度。"
        return self.kv_cache.get_seq_len()

    def truncate_cache(self, length: int):
        "将键值缓存截断到指定长度。"
        self.kv_cache.truncate(length)


# 保留旧接口别名
CauseMutiHeadAttention = FlashAttentionWithTP


# 使用示例与测试
if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention with Tensor Parallel - Test")
    print("=" * 60)

    # 参数配置
    batch_size = 2
    seq_length = 128
    d_model = 512
    n_head = 8
    n_kv_head = 4  # 分组查询注意力

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print(f"\nDevice: {device}")
    print(f"TP Available: {TP_AVAILABLE}")

    # 创建注意力模块
    attention = FlashAttentionWithTP(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head,
        max_seq_size=2048,
        device=device,
        dtype=dtype,
        use_tp=False,  # 接入张量并行框架时启用此选项
    )

    attention = attention.to(device)

    # 测试输入
    x = torch.randn(batch_size, seq_length, d_model, device=device, dtype=dtype)

    print(f"\nInput shape: {x.shape}")

    # 前向传播
    output = attention(x)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # 验证缓存模式
    print("\n" + "=" * 60)
    print("Testing KV Cache")
    print("=" * 60)

    attention.clear_cache()

    # 提示词预填充
    prefill_seq = 64
    x_prefill = x[:, :prefill_seq, :]
    output_prefill = attention(x_prefill, use_cache=True, start_pos=0)
    print(f"Prefill output shape: {output_prefill.shape}")
    print(f"Cache length after prefill: {attention.get_cache_seq_len()}")

    # 逐令牌生成
    for i in range(5):
        x_new = torch.randn(batch_size, 1, d_model, device=device, dtype=dtype)
        output_new = attention(x_new, use_cache=True, start_pos=prefill_seq + i)
        print(f"Step {i+1} - Output shape: {output_new.shape}, Cache length: {attention.get_cache_seq_len()}")

    print("\n✓ All tests passed!")
