from typing import Optional, Tuple
import torch 
import math 
from softmax import StableSoftmax
from torch import nn
from einops import rearrange
from rope import RotaryPositionalEmbedding

#计算打分表Q*K 并对V进行加权输出
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
    KV Cache 用于存储和管理 Key-Value 缓存
    
    在自回归生成时:
    - 首次输入: 缓存所有 K, V
    - 后续输入: 只计算新token的 K, V，拼接到缓存中
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
        """
        更新缓存并返回完整的 K, V
        
        Args:
            k: 新的 Key [batch, n_head, seq_len, d_k]
            v: 新的 Value [batch, n_head, seq_len, d_k]
            start_pos: 新token在序列中的起始位置
            
        Returns:
            完整的 K, V (包含缓存的历史部分)
        """
        if self.k_cache is None:
            # 首次调用，直接缓存
            self.k_cache = k
            self.v_cache = v
        else:
            # 拼接新的 K, V 到缓存
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
            
        return self.k_cache, self.v_cache
    def truncate(self, max_len: int):
        """
        截断缓存到指定长度 (用于投机采样回退)
        Args:
            max_len: 保留的序列长度
        """
        if self.k_cache is not None and self.k_cache.size(2) > max_len:
            # 维度: [batch, n_head, seq_len, d_k] -> 在第2维截断
            self.k_cache = self.k_cache[:, :, :max_len, :]
            self.v_cache = self.v_cache[:, :, :max_len, :]
    def clear(self):
        """清空缓存"""
        self.k_cache = None
        self.v_cache = None
    
    def get_seq_len(self) -> int:
        """获取当前缓存的序列长度"""
        if self.k_cache is None:
            return 0
        return self.k_cache.size(2)

class CauseMutiHeadAttention(nn.Module):
    def __init__ (self , 
                  d_model:int , 
                  n_head : int , 
                  max_seq_size : int = None, 
                  device = None , 
                  dtype = None , 
                  theta=None):
        super().__init__()
        #判断维度
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.device = device
        #Q,K,V投影层，投影到各自的维度
        factory_par = {"device":device , "dtype":dtype}
        self.q_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.k_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.v_pro = nn.Linear(d_model , d_model ,**factory_par)
        #输出投影层 整合所有信息
        self.output_pro = nn.Linear(d_model , d_model ,**factory_par)
        
        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.d_k,max_seq_size,device=device)
        else:
            self.rope = None
        
        self.k_v_cache = KVCache()
    def forward(self,x:torch.tensor, 
                token_position:torch.tensor = None,
                use_cache: bool = False,
                start_pos: int = 0
                )-> torch.Tensor :
        b,s,d = x.shape
        #将映射拆分为多头
        q = rearrange(self.q_pro(x),'... s (h d) -> ... h s d', h=self.n_head)
        k = rearrange(self.k_pro(x),'... s (h d) -> ... h s d', h=self.n_head)
        v = rearrange(self.v_pro(x),'... s (h d) -> ... h s d', h=self.n_head)

        #应用RoPE 旋转位置编码
        if self.rope is not None:
            if token_position is None:
                #默认生成从0开始的顺序位置
                #expand处理 Batch维度，不占用额外的内存
                token_position = torch.arange(s,device=x.device).expand(b,s)
            
            #对Q,K进行旋转，V保持不动
            q = self.rope(q,token_position)
            k = self.rope(k,token_position)
        
        # 使用 KV Cache
        if use_cache:
            # 更新缓存并获取完整的 K, V
            k, v = self.k_v_cache.update(k, v, start_pos)
            
            # 当前缓存的序列长度
            cached_seq_len = self.k_v_cache.get_seq_len()
            
            # 生成因果掩码
            # Q 的长度是当前输入长度 s
            # K 的长度是缓存长度 cached_seq_len
            if s == 1:
                # 生成阶段：单个新token可以看所有历史token
                # mask shape: [1, cached_seq_len]，全为True
                mask = torch.ones(1, cached_seq_len, device=self.device, dtype=torch.bool)
            else:
                # Prefill阶段：需要完整的因果掩码
                # 创建 [s, cached_seq_len] 的掩码
                mask = torch.zeros(s, cached_seq_len, device=self.device, dtype=torch.bool)
                
                # 历史缓存部分（start_pos之前）全部可见
                if start_pos > 0:
                    mask[:, :start_pos] = True
                
                # 当前输入部分（start_pos到start_pos+s）使用下三角掩码
                current_mask = torch.tril(
                    torch.ones(s, s, device=self.device, dtype=torch.bool)
                )
                mask[:, start_pos:start_pos+s] = current_mask
            
            # 如果是生成阶段(s=1)，Q只需要看所有之前的K
            # mask shape: [1, cached_seq_len]，全为True
        else:
            # 训练模式: 标准因果掩码
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))
        
        #核心注意力计算（SDPA）(bath_size,heads,seq,d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask=mask)
        
        #合并多头
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')

        return self.output_pro(attn_out)
    def clear_cache(self):
        """清空 KV Cache"""
        self.k_v_cache.clear()
    
    def get_cache_seq_len(self) -> int:
        """获取当前缓存的序列长度"""
        return self.k_v_cache.get_seq_len()

    def truncate_cache(self, length: int):
        """截断 KV Cache"""
        self.k_v_cache.truncate(length)