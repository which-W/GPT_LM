import torch
from torch import nn
from attention import CauseMutiHeadAttention
from rmsnorm import RMSNorm
from swiGLU import SwiGLU
from mhc import mHC

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 d_ff:int,
                 n_head:int,
                 max_seq_len:int,
                 theta:float,
                 n:int = 4,
                 device=None,
                 dtype=None):
        super().__init__()
        self.attn_mhc = mHC(d_model=d_model, n=n)
        self.ffn_mhc = mHC(d_model=d_model, n=n)
        
        # 初始化因果注意力模块
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        
        # 初始化两个RMSNorm层，用于attention和FFN
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # 初始化前馈网络（SwiGLU）
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x:torch.Tensor,
                x_position:torch.Tensor,
                use_cache: bool = False,
                start_pos: int = 0):
        """
        Args:
            x: 输入张量 [B, S, n, d_model] (来自 mHC 结构)
            x_position: 位置编码 [B, S]
            use_cache: 是否使用 KV Cache
            start_pos: cache 起始位置
        Returns:
            x: 输出张量 [B, S, n, d_model]
        """
        # === Attention 子层 (with mHC) ===
        # 1. Width connection: 分支间信息交互
        h_pre, h_res, H_post = self.attn_mhc.width_connection(x)
        # h_pre: [B, S, 1, d_model] - 压缩后的表示
        # h_res: [B, S, n, d_model] - 残差分支
        # H_post: [B, S, n] - 后处理权重
        
        # 2. 对压缩表示应用 LayerNorm + Attention
        h_pre_squeezed = h_pre.squeeze(2)  # [B, S, d_model]
        attn_input = self.ln1(h_pre_squeezed)  # [B, S, d_model]
        attn_output = self.attention(
            attn_input, 
            token_position=x_position,
            use_cache=use_cache,
            start_pos=start_pos
        )  # [B, S, d_model]
        
        # 3. Depth connection: 残差连接 + 扩展回 n 个分支
        x = self.attn_mhc.depth_connection(h_res, attn_output, H_post)  # [B, S, n, d_model]
        
        # FFN 子层 (with mHC)
        # 4. Width connection
        h_pre, h_res, H_post = self.ffn_mhc.width_connection(x)
        
        # 5. LayerNorm + FFN
        h_pre_squeezed = h_pre.squeeze(2)  # [B, S, d_model]
        ffn_input = self.ln2(h_pre_squeezed)
        ffn_output = self.ffn(ffn_input)  # [B, S, d_model]
        
        # 6. Depth connection
        x = self.ffn_mhc.depth_connection(h_res, ffn_output, H_post)  # [B, S, n, d_model]
        
        return x
    
    def clear_cache(self):
        """清空该层的 KV Cache"""
        self.attention.clear_cache()
    
    def truncate_cache(self, length: int):
        """传递截断指令给 attention 层"""
        self.attention.truncate_cache(length)
    
    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return self.attention.get_cache_seq_len()