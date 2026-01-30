
from typing import Optional, Tuple
import torch 
import math 
from torch import nn
from einops import rearrange
from softmax import StableSoftmax
from rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
):
    d_k = Q.size(-1)
    scores = torch.einsum('...nk,...mk -> ...nm', Q, K) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    softmax = StableSoftmax(dim=-1)
    probs = softmax(scores)
    output = torch.einsum('...nm, ...mk -> ...nk', probs, V)
    
    return output


class PagedKVCache:
    """
    PagedAttention 风格的 KV Cache
    使用物理块管理，支持非连续内存存储
    """
    def __init__(
        self, 
        num_blocks: int, 
        block_size: int, 
        num_heads: int, 
        head_dim: int, 
        device=None, 
        dtype=None
    ):
        """
        Args:
            num_blocks: 物理显存块总数
            block_size: 每个块能存储的 token 数量
            num_heads: 注意力头数
            head_dim: 每个头的维度
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # 物理 KV Cache 池：[num_blocks, block_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
    
    def store(
        self,
        k: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        v: torch.Tensor,
        slot_mapping: torch.Tensor   # [total_tokens]
    ):
        """
        将新计算的 K, V 存入物理块
        
        Args:
            k, v: 当前计算的 Key/Value
            slot_mapping: 每个 token 对应的物理槽位 (block_id * block_size + offset)
        """
        # 重排为 [total_tokens, num_heads, head_dim]
        k_flat = rearrange(k, 'b h s d -> (b s) h d')
        v_flat = rearrange(v, 'b h s d -> (b s) h d')
        
        # 存入对应的物理槽位
        for i, slot in enumerate(slot_mapping):
            if slot >= 0:  # -1 表示 padding，跳过
                block_id = slot // self.block_size
                offset = slot % self.block_size
                if block_id < self.num_blocks:
                    self.k_cache[block_id, offset] = k_flat[i]
                    self.v_cache[block_id, offset] = v_flat[i]
    
    def gather(
        self,
        block_tables: torch.Tensor,  # [batch, max_num_blocks]
        context_lens: torch.Tensor   # [batch]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从物理块中收集 K, V,用于 Decode 阶段
        
        Returns:
            k, v: [batch, num_heads, max_context_len, head_dim]
        """
        batch_size = block_tables.shape[0]
        max_context_len = context_lens.max().item()
        
        # 预分配输出
        k_out = torch.zeros(
            batch_size, self.num_heads, max_context_len, self.head_dim,
            device=self.device, dtype=self.k_cache.dtype
        )
        v_out = torch.zeros_like(k_out)
        
        for b in range(batch_size):
            context_len = context_lens[b].item()
            num_blocks = (context_len + self.block_size - 1) // self.block_size
            
            for i in range(num_blocks):
                if i >= block_tables.shape[1]:
                    break
                block_id = block_tables[b, i].item()
                if block_id < 0 or block_id >= self.num_blocks:  # 无效块
                    break
                
                start = i * self.block_size
                end = min(start + self.block_size, context_len)
                block_len = end - start
                
                # 从物理块复制到输出
                k_out[b, :, start:end] = self.k_cache[block_id, :block_len].permute(1, 0, 2)
                v_out[b, :, start:end] = self.v_cache[block_id, :block_len].permute(1, 0, 2)
        
        return k_out, v_out


class PagedCausalMultiHeadAttention(nn.Module):
    """
    支持 PagedAttention 的因果多头注意力
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_size: int = None,
        # vLLM 相关参数
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        device=None,
        dtype=None,
        theta=None
    ):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.device = device
        
        # Q, K, V 投影层
        factory_par = {"device": device, "dtype": dtype}
        self.q_pro = nn.Linear(d_model, d_model, **factory_par)
        self.k_pro = nn.Linear(d_model, d_model, **factory_par)
        self.v_pro = nn.Linear(d_model, d_model, **factory_par)
        self.output_pro = nn.Linear(d_model, d_model, **factory_par)
        
        # RoPE 位置编码
        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_size, device=device)
        else:
            self.rope = None
        
        # PagedAttention KV Cache
        self.paged_cache = PagedKVCache(
            num_blocks=num_kv_blocks,
            block_size=block_size,
            num_heads=n_head,
            head_dim=self.d_k,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor = None,
        # vLLM 风格参数
        is_prefill: bool = True,
        block_tables: torch.Tensor = None,
        slot_mapping: torch.Tensor = None,
        context_lens: torch.Tensor = None,
        # 简单cache
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        统一的前向传播，支持训练和推理两种模式
        
        训练模式 (is_prefill=True, block_tables=None):
            标准的因果注意力，不使用 Paged Cache
        
        Prefill 推理模式 (is_prefill=True, block_tables!=None):
            使用 Paged Cache 存储 KV，支持前缀缓存
        
        Decode 推理模式 (is_prefill=False):
            从 Paged Cache 读取历史 KV，只计算新 token
        """
        b, s, d = x.shape
        
        # 投影并拆分多头
        q = rearrange(self.q_pro(x), '... s (h d) -> ... h s d', h=self.n_head)
        k = rearrange(self.k_pro(x), '... s (h d) -> ... h s d', h=self.n_head)
        v = rearrange(self.v_pro(x), '... s (h d) -> ... h s d', h=self.n_head)
        
        # 应用 RoPE
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(s, device=x.device).expand(b, s)
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)
        
        # 根据模式选择注意力计算方式
        if block_tables is None and not use_cache:
            # 训练模式:标准因果注意力
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))
            attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        elif is_prefill and block_tables is not None:
            # Prefill - vLLM 风格
            # 存储 KV 到 Paged Cache
            if slot_mapping is not None:
                self.paged_cache.store(k, v, slot_mapping)
            
            # 使用标准注意力
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))
            attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        elif not is_prefill and block_tables is not None:
            # Decode 推理模式,vLLM 风格
            # 从 Paged Cache 收集历史 KV
            k_cache, v_cache = self.paged_cache.gather(block_tables, context_lens)
            
            # 存储当前新 token 的 KV
            if slot_mapping is not None:
                self.paged_cache.store(k, v, slot_mapping)
            
            # 拼接新旧 KV
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
            
            # 注意力计算（新 token 可以看所有历史）
            mask = torch.ones(s, k_full.shape[2], device=self.device, dtype=torch.bool)
            attn_out = scaled_dot_product_attention(q, k_full, v_full, mask=mask)
        
        else:
            # 【兼容旧的 use_cache 模式】
            
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))
            attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 合并多头
        attn_out = rearrange(attn_out, '... h s d -> ... s (h d)')
        return self.output_pro(attn_out)
    
    def clear_cache(self):
        """清空物理 Cache"""
        pass



CauseMutiHeadAttention = PagedCausalMultiHeadAttention