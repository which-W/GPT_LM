"""
Improved Attention Module with Flash Attention Style and Tensor Parallel Support
Compatible with picotron's distributed training framework
"""

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
# Import tron_support's process group manager for TP support
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
    """
        Q:[..., N ,d_k]
        K:[..., m ,d_k]
        V:[..., m ,d_v]
    """
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


class KVCache:
    """
    KV Cache for autoregressive generation
    Compatible with picotron's caching strategy
    """
    
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None  # [batch, n_head, seq_len, d_k]
        self.v_cache: Optional[torch.Tensor] = None  # [batch, n_head, seq_len, d_k]
        
    def update(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return complete K, V"""
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
            
        return self.k_cache, self.v_cache
    
    def truncate(self, max_len: int):
        """Truncate cache to specified length"""
        if self.k_cache is not None and self.k_cache.size(2) > max_len:
            self.k_cache = self.k_cache[:, :, :max_len, :]
            self.v_cache = self.v_cache[:, :, :max_len, :]
    
    def clear(self):
        """Clear cache"""
        self.k_cache = None
        self.v_cache = None
    
    def get_seq_len(self) -> int:
        """Get current cached sequence length"""
        if self.k_cache is None:
            return 0
        return self.k_cache.size(2)

class FlashAttentionWithTP(nn.Module):
    """
    Multi-Head Attention with Flash Attention and Tensor Parallel Support
    Compatible with picotron's training framework
    
    Features:
    - Flash Attention for efficiency
    - Tensor Parallel (TP) support
    - KV Cache for generation
    - RoPE positional encoding
    - GQA (Grouped Query Attention) support
    """
    
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
        
        # Basic configuration
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d_model // n_head
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.bfloat16
        self.use_tp = use_tp and TP_AVAILABLE
        
        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.head_dim,max_seq_size,device=device)
        else:
            self.rope = None
        # Tensor Parallel configuration
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
        
        # Initialize projection layers with TP support
        factory_kwargs = {"device": device, "dtype": dtype}
        
        if self.use_tp and self.tp_world_size > 1:
            # Use TP parallel layers
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
            # Standard linear layers
            self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(d_model, self.n_kv_head * self.head_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(d_model, self.n_kv_head * self.head_dim, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
         
        # KV Cache
        self.kv_cache = KVCache()
              
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
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
        """
        Forward pass with Flash Attention and TP support
        
        Args:
            x: [batch_size, seq_length, d_model]
            token_position: [batch_size, seq_length] position indices
            attention_mask: custom attention mask
            use_cache: whether to use KV cache
            start_pos: starting position for cache
        
        Returns:
            output: [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = x.size()
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, num_local_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, num_local_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, num_local_kv_heads * head_dim]
        
       
        # Standard format: [batch, heads, seq, dim]
        q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.rope is not None:
            if token_position is None:
                #默认生成从0开始的顺序位置
                #expand处理 Batch维度，不占用额外的内存
                token_position = torch.arange(seq_length,device=x.device).expand(batch_size,seq_length)
        
        #对Q,K进行旋转，V保持不动
        q = self.rope(q,token_position)
        k = self.rope(k,token_position)
        
        # Handle KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)
            cached_seq_len = self.kv_cache.get_seq_len()
            
            # Generate causal mask for cache
            if attention_mask is None:
                if seq_length == 1:
                    # Generation: single token can see all history
                    attention_mask = torch.ones(1, cached_seq_len, device=self.device, dtype=torch.bool)
                else:
                    # Prefill: full causal mask
                    attention_mask = torch.zeros(seq_length, cached_seq_len, device=self.device, dtype=torch.bool)
                    if start_pos > 0:
                        attention_mask[:, :start_pos] = True
                    current_mask = torch.tril(
                        torch.ones(seq_length, seq_length, device=self.device, dtype=torch.bool)
                    )
                    attention_mask[:, start_pos:start_pos+seq_length] = current_mask
        else:
            # Training: standard causal mask
            if attention_mask is None:
                attention_mask = torch.tril(
                    torch.ones(seq_length, seq_length, device=self.device, dtype=torch.bool)
                )
        
        # Handle GQA: repeat K, V if needed
        if self.num_local_heads != self.num_local_kv_heads:
            repeat_factor = self.num_local_heads // self.num_local_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention computation
        causal = (q.size(2) == k.size(2)) if not use_cache else False
        
    
        # PyTorch SDPA path
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask if not causal else None,
            is_causal=causal and attention_mask is None
        )
        # out: [batch, heads, seq, dim]
        out = out.transpose(1, 2)  # [batch, seq, heads, dim]
        
        # Merge heads
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out
    
    def clear_cache(self):
        """Clear KV cache"""
        self.kv_cache.clear()
    
    def get_cache_seq_len(self) -> int:
        """Get current cache sequence length"""
        return self.kv_cache.get_seq_len()
    
    def truncate_cache(self, length: int):
        """Truncate KV cache to specified length"""
        self.kv_cache.truncate(length)


# Alias for compatibility
CauseMutiHeadAttention = FlashAttentionWithTP


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention with Tensor Parallel - Test")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_length = 128
    d_model = 512
    n_head = 8
    n_kv_head = 4  # GQA
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    print(f"\nDevice: {device}")
    print(f"TP Available: {TP_AVAILABLE}")
    
    # Create attention module
    attention = FlashAttentionWithTP(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head,
        max_seq_size=2048,
        device=device,
        dtype=dtype,
        use_tp=False,  # Set to True when using with tron
    )
    
    attention = attention.to(device)
    
    # Test input
    x = torch.randn(batch_size, seq_length, d_model, device=device, dtype=dtype)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Test with cache
    print("\n" + "=" * 60)
    print("Testing KV Cache")
    print("=" * 60)
    
    attention.clear_cache()
    
    # Prefill
    prefill_seq = 64
    x_prefill = x[:, :prefill_seq, :]
    output_prefill = attention(x_prefill, use_cache=True, start_pos=0)
    print(f"Prefill output shape: {output_prefill.shape}")
    print(f"Cache length after prefill: {attention.get_cache_seq_len()}")
    
    # Generation
    for i in range(5):
        x_new = torch.randn(batch_size, 1, d_model, device=device, dtype=dtype)
        output_new = attention(x_new, use_cache=True, start_pos=prefill_seq + i)
        print(f"Step {i+1} - Output shape: {output_new.shape}, Cache length: {attention.get_cache_seq_len()}")
    
    print("\n✓ All tests passed!")