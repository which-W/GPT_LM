import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        
        super().__init__() 
        self.d_k = d_k
        
        #计算频率w=theta^(2k /d)
        #我们只需要计算d_k/2的频率，旋转成对进行的
        powers = torch.arange(0,d_k,2,device=device).float() / d_k
        w_freqs = 1.0 / (theta ** powers) #（d_k/2,）
        
        #创建位置序列
        i_t = torch.arange(0,max_seq_len,device=device).float() # (max_seq_len,)
        
        #计算所有位置的所有角度（外积）
        """ w1 w2 w3...
        i=1 i*w ...
        i=2 .....
        """
        w_freqs_t_i_matrix = torch.outer(i_t,w_freqs) #(max_seq_len,d_k/2)
        
        #预计算cos和sin并作为buffer注册，persisten=False 确保不会保存在state_dict之中
        self.register_buffer("cos_cached",w_freqs_t_i_matrix.cos(),persistent=False)
        self.register_buffer("sin_cached",w_freqs_t_i_matrix.sin(),persistent=False)
        
    def forward(self,x:torch.Tensor,token_position:torch.Tensor):
        
        #提取sin,cos（...,seq_len,d_k/2）
        cos = self.cos_cached[token_position]
        sin = self.sin_cached[token_position]
        
        #维度对齐
        if x.ndim >cos.ndim and cos.ndim>=3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        
        #拆分拆成奇偶序列
        x_even =x[..., 0::2]
        x_odd = x[..., 1::2]
        #计算相关的旋转并注册到向量之中
        output = torch.empty_like(x)
        output[..., 0::2] = x_even*cos - x_odd*sin
        output[..., 1::2] = x_even*sin + x_odd*cos
        
        return output