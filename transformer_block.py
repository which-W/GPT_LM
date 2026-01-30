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