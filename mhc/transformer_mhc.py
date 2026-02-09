from typing import Optional
import torch
from torch import nn
from emb import CustomEmbedding
from rmsnorm import RMSNorm
from mhc.transformer_block_mhc import TransformerBlock

class TransformerLM(nn.Module):
    """
    带 KV Cache 和 mHC 的 Transformer 语言模型
    
    功能:
    1. 训练模式: 标准的自回归训练
    2. 推理模式: 使用 KV Cache 加速生成
    3. mHC 架构: 多分支层次连接
    
    使用示例:
    ```python
    # 训练
    logits = model(tokens, use_cache=False)
    
    # 推理 - Prefill
    model.clear_cache()
    logits = model(prompt_tokens, use_cache=True)
    
    # 推理 - Generation
    for i in range(max_new_tokens):
        logits = model(next_token, use_cache=True)
        next_token = sample(logits)
    ```
    """
    def __init__(self,
                 d_model:int,
                 n_head:int,
                 vocab_size:int,
                 max_seq_len:int,
                 d_ff:int,
                 theta:float,
                 n_layer:int,
                 device=None,
                 dtype=None,
                 # 实验参数
                 use_rms_norm:bool = True,
                 norm_model:str = "pre",
                 ffn_type:str = "swiglu",
                 # mHC 参数
                 n:int = 4,  # mHC 的分支数
                 ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n = n  # 保存 mHC 分支数
        factroy_pra = {"device":device, "dtype":dtype}
        
        # 初始化 embedding 层
        self.embedding = CustomEmbedding(vocab_size, d_model, **factroy_pra)
        
        # 堆叠 transformer block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_head=n_head,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    n=n,  # 传递 mHC 分支数
                    **factroy_pra,
                )
                for _ in range(n_layer)
            ]
        )
        
        # 最终输出层
        # 如果全局禁用 norm 这里的 RMSNorm 要设为 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, **factroy_pra)
        else:
            self.ln_final = nn.Identity()
            
        # 最后一个 Linear 用来返回词表大小
        self.ln_output = nn.Linear(d_model, vocab_size, **factroy_pra)
        
        # 用于跟踪当前生成位置
        self._current_pos = 0
    
    def forward(self,
                token_ids:torch.Tensor,
                use_cache: bool = False,
               ):
        b, s = token_ids.shape
        
        # 获取 RoPE 的位置向量
        if use_cache:
            # 推理模式: 使用缓存位置
            start_pos = self._current_pos
            token_position = torch.arange(
                start_pos, 
                start_pos + s,
                device=self.device,
                dtype=torch.long
            ).unsqueeze(0).expand(b, s)
            
            # 更新位置计数器
            self._current_pos += s
        else:
            # 训练模式: 从0开始的顺序位置
            start_pos = 0
            token_position = torch.arange(
                s, 
                device=self.device, 
                dtype=torch.long
            ).unsqueeze(0).expand(b, s)
        
        # Embedding: [B, L] -> [B, L, d_model]
        x = self.embedding(token_ids)
        
        # 扩展为 mHC 格式: [B, L, d_model] -> [B, L, n, d_model]
        # 初始时,所有分支共享相同的 embedding
        x = x.unsqueeze(2).expand(b, s, self.n, -1)  # [B, L, n, d_model]
        
        # 逐层通过 block
        for layer in self.layers:
            x = layer(x, token_position, use_cache=use_cache, start_pos=start_pos)
        
        # 聚合 mHC 的多个分支: [B, L, n, d_model] -> [B, L, d_model]
        # 使用平均池化 (也可以用其他策略,如加权平均、只取第一个分支等)
        x = x.mean(dim=2)  # [B, L, d_model]
        
        # 最终归一化,如果 use_rms_norm 为 false 则不会通过这一层
        x = self.ln_final(x)
        
        # 返回投射到词表空间的 logits
        return self.ln_output(x)
    
    def clear_cache(self):
        """清空所有层的 KV Cache"""
        for layer in self.layers:
            layer.clear_cache()
        self._current_pos = 0
    
    def truncate_cache(self, length: int):
        """
        截断 KV Cache 到指定长度 (用于投机采样回退)
        Args:
            length: 保留的序列长度 (start_pos)
        """
        # 截断所有层的 cache
        for layer in self.layers:
            layer.truncate_cache(length)
        
        # 重置当前位置指针
        self._current_pos = length