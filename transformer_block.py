import torch
from torch import nn
from attention import CauseMutiHeadAttention
from rmsnorm import RMSNorm
from swiGLU import SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 d_ff:int,
                 n_head:int,
                 max_seq_len:int,
                 theta:float,
                 device=None,
                 dtype=None):
        super().__init__()
        #初始化因果注意力模块
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device = device,
            dtype=dtype,
        )
        #初始化两个RMSNorm层，用于attention和FNN
        self.ln1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        
        #初始化前反馈网络（SWiGLU）
        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)
        
    def forward(self,x:torch.Tensor,
                x_position:torch.Tensor,
                use_cache: bool = False,
                start_pos: int = 0,):
        #1.attention子层(pre-norm结构）
        #x被分成两路，一路直接传走（残差），一路进入norm + attention
        x = x + self.attention(self.ln1(x),token_position = x_position, use_cache=use_cache,start_pos=start_pos)
        #2.FFN子层
        #x被分成两路，一路直接传走（残差），一路进入norm + ffn
        x = x + self.ffn(self.ln2(x))
        
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
    

class TransformerBlock_AttenRes(nn.Module):
    """
    TransformerBlock —— Kimi k1.5 Long-Short Residual Stream 版本

    核心改动（相比传统 Pre-Norm Transformer）：
    ┌──────────────────────────────────────────────────────────────┐
    │  传统:                                                        │
    │    x = x + Attention(Norm(x))                                │
    │    x = x + FFN(Norm(x))                                      │
    │                                                              │
    │  Kimi k1.5:                                                  │
    │    attn_out    = Attention(Norm(x))                          │
    │    attn_stream = attn_stream + attn_out   ← 跨层累积注意力流  │
    │    x           = x + attn_out             ← 短程残差(不变)   │
    │    x           = x + FFN(Norm(x + attn_stream))  ← FFN 双输入│
    └──────────────────────────────────────────────────────────────┘

    attn_stream 需要由外部（模型主干）在层间传递，初始值为全零张量。
    每次 forward 返回 (x, attn_stream)，调用方负责把 attn_stream
    传给下一层。
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_head: int,
                 max_seq_len: int,
                 theta: float,
                 device=None,
                 dtype=None):
        super().__init__()

        # 因果多头注意力
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        # Pre-Norm：两个 RMSNorm
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # 前馈网络（SwiGLU）
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_position: torch.Tensor,
        attn_stream: torch.Tensor,          # ← 新增：跨层注意力累积流
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        """
        Args:
            x            : 输入隐状态 [B, S, D]
            x_position   : token 位置索引 [B, S]
            attn_stream  : 跨层注意力累积流 [B, S, D]，初始为 zeros_like(x)
            use_cache    : 是否使用 KV Cache
            start_pos    : KV Cache 起始位置

        Returns:
            x            : 更新后的隐状态 [B, S, D]
            attn_stream  : 更新后的注意力累积流 [B, S, D]（传给下一层）
        """
        # ── 1. 注意力子层（短程残差） ──────────────────────────────────
        attn_out = self.attention(
            self.ln1(x),
            token_position=x_position,
            use_cache=use_cache,
            start_pos=start_pos,
        )

        # 长程注意力流：跨层累加注意力输出
        attn_stream = attn_stream + attn_out

        # 短程残差：x 照常更新
        x = x + attn_out

        # ── 2. FFN 子层（同时感知短程 x 和长程 attn_stream） ──────────
        # FFN 的输入 = Norm(x + attn_stream)，即让 FFN 看到历史所有层
        # 累积的注意力信号，而不只是当前层的输出
        x = x + self.ffn(self.ln2(x + attn_stream))

        return x, attn_stream

    # ------------------------------------------------------------------
    def clear_cache(self):
        """清空该层的 KV Cache"""
        self.attention.clear_cache()

    def truncate_cache(self, length: int):
        """截断 KV Cache（用于投机采样回退）"""
        self.attention.truncate_cache(length)

    def get_cache_seq_len(self) -> int:
        """获取当前缓存序列长度"""
        return self.attention.get_cache_seq_len()


# ======================================================================
# 使用示例（在模型主干中如何串联多个 TransformerBlock_AttenRes）
# ======================================================================
# 
# class transformer(nn.Module):
#     def __init__(self, ...):
#         self.layers = nn.ModuleList([TransformerBlock_AttenRes(...) for _ in range(n_layers)])
#
#     def forward(self, x, x_position, use_cache=False, start_pos=0):
#         # attn_stream 初始化为全零，形状与 x 相同
#         attn_stream = torch.zeros_like(x)
#
#         for layer in self.layers:
#             x, attn_stream = layer(
#                 x, x_position, attn_stream,
#                 use_cache=use_cache, start_pos=start_pos,
#             )
#         return x