"""
支持 vLLM 推理的完整 Transformer 语言模型
兼容训练模式和高效推理模式
"""
from typing import Optional
import torch
from torch import nn
from vllm_support.vllm_transformer_block import PagedTransformerBlock
from emb import CustomEmbedding
from rmsnorm import RMSNorm

class PagedTransformerLM(nn.Module):
    """
    支持 PagedAttention 的 Transformer 语言模型
    
    使用模式:
    1. 训练模式:
       logits = model(tokens, is_prefill=True, block_tables=None)
    2. vLLM 推理模式:
       # Prefill
       logits = model(prompt_tokens, is_prefill=True, block_tables=..., slot_mapping=...)
       # Decode
       for _ in range(max_tokens):
           logits = model(next_token, is_prefill=False, block_tables=..., slot_mapping=...)
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        vocab_size: int,
        max_seq_len: int,
        d_ff: int,
        theta: float,
        n_layer: int,
        # vLLM 参数
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        device=None,
        dtype=None,
        # 实验参数
        use_rms_norm: bool = True,
        norm_model: str = "pre",
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_kv_blocks = num_kv_blocks
        self.block_size = block_size
        
        factory_pra = {"device": device, "dtype": dtype}
        
        # 初始化 embedding 层
        self.embedding = CustomEmbedding(vocab_size, d_model, **factory_pra)
        
        # 堆叠 transformer block
        self.layers = nn.ModuleList([
            PagedTransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_head=n_head,
                max_seq_len=max_seq_len,
                theta=theta,
                num_kv_blocks=num_kv_blocks,
                block_size=block_size,
                **factory_pra,
            )
            for _ in range(n_layer)
        ])
        
        # 最终输出层
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, **factory_pra)
        else:
            self.ln_final = nn.Identity()
        
        # 输出投影到词表
        self.ln_output = nn.Linear(d_model, vocab_size, **factory_pra)
        
        # 位置计数器(旧cache)
        self._current_pos = 0
    
    def forward(
        self,
        token_ids: torch.Tensor,
        # vLLM 风格参数
        is_prefill: bool = True,
        block_tables: torch.Tensor = None,
        slot_mapping: torch.Tensor = None,
        context_lens: torch.Tensor = None,
    ):
        """
        统一的前向传播
        
        Args:
            token_ids: [batch, seq_len]
            is_prefill: True=Prefill阶段,False=Decode阶段
            block_tables: [batch, max_num_blocks] 物理块映射表
            slot_mapping: [total_tokens] token到槽位的映射
            context_lens: [batch] 每个序列的上下文长度
        """
        b, s = token_ids.shape
        
        # 获取位置编码
        # vLLM 模式或训练模式
        if context_lens is not None and not is_prefill:
            # Decode: 位置 = 当前上下文长度
            token_position = context_lens.unsqueeze(1).expand(b, s)
        else:
            # Prefill 或训练: 从0开始的顺序位置
            token_position = torch.arange(
                s, device=self.device, dtype=torch.long
            ).unsqueeze(0).expand(b, s)
        
        # Embedding
        x = self.embedding(token_ids)
        
        # 逐层通过 block
        for layer in self.layers:
            x = layer(
                x, 
                token_position,
                is_prefill=is_prefill,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
            )
        
        # 最终归一化
        x = self.ln_final(x)
        
        # 返回投射到词表空间的 logits
        return self.ln_output(x)
    
    def clear_cache(self):
        """清空所有层的 KV Cache"""
        for layer in self.layers:
            layer.clear_cache()
        self._current_pos = 0
    
    def truncate_cache(self, length: int):
        """截断 KV Cache 到指定长度"""
        for layer in self.layers:
            layer.truncate_cache(length)
        self._current_pos = length


# 兼容性别名
TransformerLM = PagedTransformerLM