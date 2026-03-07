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
    
    #计算相似度分数,形成打分表
    scores = torch.einsum('...nk,...mk -> ...nm',Q,K) / math.sqrt(d_k)
    #应用mask掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
        
    #计算注意力权重(归一化)
    #dim=-1 对应的是每一个Q对于K的分布
    softmax = StableSoftmax(dim=-1)
    probs = softmax(scores)
    
    #加权求和得到输出
    output = torch.einsum('...nm, ...mk -> ...nk', probs ,V)
    
    return output
#链式KVCache
class KVCache:
    """
    KV Cache 用于存储和管理 Key-Value 缓存
    
    在自回归生成时:
    - 首次输入: 缓存所有 K, V
    - 后续输入: 只计算新token的 K, V,拼接到缓存中
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
            # 首次调用,直接缓存
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
#MHA机制+链表KVCache
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
        #Q,K,V投影层,投影到各自的维度
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
                #expand处理 Batch维度,不占用额外的内存
                token_position = torch.arange(s,device=x.device).expand(b,s)
            
            #对Q,K进行旋转,V保持不动
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
                # 生成阶段:单个新token可以看所有历史token
                # mask shape: [1, cached_seq_len],全为True
                mask = torch.ones(1, cached_seq_len, device=self.device, dtype=torch.bool)
            else:
                # Prefill阶段:需要完整的因果掩码
                # 创建 [s, cached_seq_len] 的掩码
                mask = torch.zeros(s, cached_seq_len, device=self.device, dtype=torch.bool)
                
                # 历史缓存部分(start_pos之前)全部可见
                if start_pos > 0:
                    mask[:, :start_pos] = True
                
                # 当前输入部分(start_pos到start_pos+s)使用下三角掩码
                current_mask = torch.tril(
                    torch.ones(s, s, device=self.device, dtype=torch.bool)
                )
                mask[:, start_pos:start_pos+s] = current_mask
            
            # 如果是生成阶段(s=1),Q只需要看所有之前的K
            # mask shape: [1, cached_seq_len],全为True
        else:
            # 训练模式: 标准因果掩码
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))
        
        #核心注意力计算(SDPA)(bath_size,heads,seq,d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask=mask)
        
        #合并多头
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')
        #返回output以便之后取出logits
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
#GQA
class GroupedQueryAttention(nn.Module):
    """
    Args:
        d_model:     模型维度
        n_head:      Q 的头数(必须被 n_kv_head 整除)
        n_kv_head:   K/V 的头数
        max_seq_size: 最大序列长度(RoPE 需要)
        theta:        RoPE 基频
        use_flash:    是否使用 Flash Attention 内核
        device/dtype: 设备与精度
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_size: int = None,
        device=None,
        dtype=None,
        theta=None,
        use_flash: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须被 n_head 整除"
        assert n_head % n_kv_head == 0, "n_head 必须被 n_kv_head 整除"

        self.d_model   = d_model
        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.n_rep     = n_head // n_kv_head   # 每个 KV 头被几个 Q 头共享
        self.d_k       = d_model // n_head
        self.device    = device
        self.use_flash = use_flash

        factory_par = {"device": device, "dtype": dtype}

        # Q 投影:输出 n_head * d_k = d_model
        self.q_proj = nn.Linear(d_model, n_head * self.d_k, **factory_par)
        # K/V 投影:输出 n_kv_head * d_k(比 MHA 小)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.d_k, **factory_par)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.d_k, **factory_par)
        self.out_proj = nn.Linear(d_model, d_model, **factory_par)

        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_k, max_seq_size, device=device)
            if (theta is not None and max_seq_size is not None)
            else None
        )
        self.kv_cache = KVCache()

    # ----------------------------------------------------------
    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        将 KV 头复制 n_rep 次,对齐 Q 头数。
        输入:  [batch, n_kv_head, seq, d_k]
        输出:  [batch, n_head,    seq, d_k]
        使用 expand + reshape,不额外分配显存。
        """
        if n_rep == 1:
            return x
        b, h, s, d = x.shape
        return (
            x.unsqueeze(2)                     # [b, h, 1, s, d]
             .expand(b, h, n_rep, s, d)        # [b, h, n_rep, s, d]
             .reshape(b, h * n_rep, s, d)      # [b, n_head, s, d]
        )

    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # 投影
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_head)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_kv_head)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_kv_head)

        # RoPE
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(s, device=x.device).expand(b, s)
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)

        # KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)
            cached_len = self.kv_cache.get_seq_len()
            if s == 1:
                mask = torch.ones(1, cached_len, device=self.device, dtype=torch.bool)
            else:
                mask = torch.zeros(s, cached_len, device=self.device, dtype=torch.bool)
                if start_pos > 0:
                    mask[:, :start_pos] = True
                mask[:, start_pos:start_pos + s] = torch.tril(
                    torch.ones(s, s, device=self.device, dtype=torch.bool)
                )
        else:
            cached_len = s
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))

        # 将 KV 头扩展到 Q 头数
        k_exp = self._repeat_kv(k, self.n_rep)   # [b, n_head, cached_len, d_k]
        v_exp = self._repeat_kv(v, self.n_rep)

        # 注意力计算
        attn_out = scaled_dot_product_attention(q, k_exp, v_exp, mask=mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)

    def clear_cache(self):
        self.kv_cache.clear()

    def get_cache_seq_len(self) -> int:
        return self.kv_cache.get_seq_len()

    def truncate_cache(self, length: int):
        self.kv_cache.truncate(length)
#MQA
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention(MQA)
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_size: int = None,
        device=None,
        dtype=None,
        theta=None,
        use_flash: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model   = d_model
        self.n_head    = n_head
        self.d_k       = d_model // n_head
        self.device    = device
        self.use_flash = use_flash

        factory_par = {"device": device, "dtype": dtype}

        self.q_proj   = nn.Linear(d_model, d_model, **factory_par)
        # K/V 只投影到单头维度 d_k(不是 d_model)
        self.k_proj   = nn.Linear(d_model, self.d_k, **factory_par)
        self.v_proj   = nn.Linear(d_model, self.d_k, **factory_par)
        self.out_proj = nn.Linear(d_model, d_model, **factory_par)

        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_k, max_seq_size, device=device)
            if (theta is not None and max_seq_size is not None)
            else None
        )
        self.kv_cache = KVCache()   # 只缓存单头 KV

    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # Q: 多头 [b, n_head, s, d_k]
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_head)
        # K/V: 单头 [b, 1, s, d_k]
        k = self.k_proj(x).unsqueeze(1)   # [b, 1, s, d_k]
        v = self.v_proj(x).unsqueeze(1)

        # RoPE(对 Q 的每一头以及单头 K 施加)
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(s, device=x.device).expand(b, s)
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)

        # KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)
            cached_len = self.kv_cache.get_seq_len()
            if s == 1:
                mask = torch.ones(1, cached_len, device=self.device, dtype=torch.bool)
            else:
                mask = torch.zeros(s, cached_len, device=self.device, dtype=torch.bool)
                if start_pos > 0:
                    mask[:, :start_pos] = True
                mask[:, start_pos:start_pos + s] = torch.tril(
                    torch.ones(s, s, device=self.device, dtype=torch.bool)
                )
        else:
            cached_len = s
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))

        # 将单头 K/V 广播到 n_head(expand 不分配额外显存)
        k_exp = k.expand(b, self.n_head, cached_len, self.d_k)
        v_exp = v.expand(b, self.n_head, cached_len, self.d_k)

       
        attn_out = scaled_dot_product_attention(q, k_exp, v_exp, mask=mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)

    def clear_cache(self):
        self.kv_cache.clear()

    def get_cache_seq_len(self) -> int:
        return self.kv_cache.get_seq_len()

    def truncate_cache(self, length: int):
        self.kv_cache.truncate(length)
#MLA
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention(MLA)
    ────────────────────────────────────
    核心思想(DeepSeek-V2, 2024):
        不直接缓存 K/V,而是缓存一个低维"潜在向量" c_kv,
        在每次 forward 时从 c_kv 上投影出完整 K/V。

        经典 MHA KV Cache 维度:2 x n_head x d_k
        MLA  KV Cache 维度:    d_c(远小于 2 X n_head X d_k)

    额外特性:
        - Q 同样通过低维瓶颈投影,减少参数量
        - 采用 Decoupled RoPE(DeepSeek-V2 Appendix):
            将 d_k 分为 d_rope(施加 RoPE)和 d_nope(不施加 RoPE),
            分别拼接后进行注意力,保证位置信息不受低秩压缩影响
        - KV Cache 只存 c_kv + k_rope,而非完整 K/V

    符号对照(对应论文表述):
        d_c       : KV 压缩维度(latent dim for KV)
        d_cq      : Q  压缩维度(latent dim for Q)
        d_rope    : RoPE 维度(每头)
        d_nope    : 非 RoPE 维度(每头),d_k = d_rope + d_nope

    Args:
        d_model:    模型维度
        n_head:     注意力头数
        d_c:        KV 潜在压缩维度(论文推荐 ≈ d_model/4)
        d_cq:       Q  潜在压缩维度(论文推荐 ≈ d_model/2),None 则等于 d_model
        d_rope:     每头 RoPE 维度(论文推荐 ≈ d_k/2)
        max_seq_size: RoPE 最大序列长度
        theta:      RoPE 基频
        use_flash:  是否使用 Flash Attention
        device/dtype
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_c: int,
        d_cq: int = None,
        d_rope: int = None,
        max_seq_size: int = None,
        device=None,
        dtype=None,
        theta=None,
        use_flash: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model   = d_model
        self.n_head    = n_head
        self.d_k       = d_model // n_head
        self.d_c       = d_c
        self.d_cq      = d_cq if d_cq is not None else d_model
        self.d_rope    = d_rope if d_rope is not None else self.d_k // 2
        self.d_nope    = self.d_k - self.d_rope     # 非 RoPE 部分维度
        self.device    = device
        self.use_flash = use_flash

        factory_par = {"device": device, "dtype": dtype}

        # ── Q 路径(低秩压缩)──────────────────────────────────
        # 下投影:x -> c_q [d_cq]
        self.q_down_proj  = nn.Linear(d_model, self.d_cq, **factory_par)
        self.q_norm       = nn.RMSNorm(self.d_cq)                      # 稳定训练
        # 上投影:c_q -> q_nope + q_rope,拼接后维度为 n_head * d_k
        self.q_up_proj    = nn.Linear(self.d_cq, n_head * self.d_k, **factory_par)

        # ── KV 路径(低秩压缩,共享潜在向量 c_kv)────────────
        # 下投影:x -> c_kv [d_c]
        self.kv_down_proj = nn.Linear(d_model, d_c, **factory_par)
        self.kv_norm      = nn.RMSNorm(d_c)
        # 上投影:c_kv -> k_nope + v_all,一次投影同时生成 K(nope) 和 V
        # 输出维度:n_head * (d_nope + d_k)
        self.kv_up_proj   = nn.Linear(d_c, n_head * (self.d_nope + self.d_k), **factory_par)

        # ── 解耦 RoPE:单独的 k_rope 投影 ─────────────────────
        # k_rope 不经过低秩压缩,直接从 x 投影
        self.k_rope_proj  = nn.Linear(d_model, n_head * self.d_rope, **factory_par)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, **factory_par)

        # RoPE(仅作用于 d_rope 维度)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_rope, max_seq_size, device=device)
            if (theta is not None and max_seq_size is not None)
            else None
        )

        # ── KV Cache:只缓存 c_kv(低维)和 k_rope ────────────
        # 使用两个独立的 KVCache 实例分别存储
        self.c_kv_cache    = KVCache()   # 存 c_kv: [b, 1, s, d_c](视为单"头")
        self.k_rope_cache  = KVCache()   # 存 k_rope: [b, n_head, s, d_rope]

    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # ── Q 计算 ─────────────────────────────────────────────
        c_q = self.q_norm(self.q_down_proj(x))                 # [b, s, d_cq]
        q   = rearrange(
            self.q_up_proj(c_q), "b s (h d) -> b h s d", h=self.n_head
        )                                                       # [b, n_head, s, d_k]
        # 拆分为 nope 和 rope 部分
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)

        # ── KV 计算 ────────────────────────────────────────────
        c_kv = self.kv_norm(self.kv_down_proj(x))              # [b, s, d_c]
        kv   = rearrange(
            self.kv_up_proj(c_kv),
            "b s (h d) -> b h s d",
            h=self.n_head
        )                                                       # [b, n_head, s, d_nope+d_k]
        k_nope, v = kv.split([self.d_nope, self.d_k], dim=-1)  # k_nope/v 各 n_head 头

        # 解耦 k_rope(直接从 x 投影,不经过低秩压缩)
        k_rope = rearrange(
            self.k_rope_proj(x), "b s (h d) -> b h s d", h=self.n_head
        )                                                       # [b, n_head, s, d_rope]

        # ── 应用 RoPE(仅 rope 维度)──────────────────────────
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(s, device=x.device).expand(b, s)
            q_rope  = self.rope(q_rope,  token_position)
            k_rope  = self.rope(k_rope, token_position)

        # 拼接得到完整 Q 和 K
        # Q: [b, n_head, s, d_nope + d_rope] = [b, n_head, s, d_k]
        q = torch.cat([q_nope, q_rope], dim=-1)
        # K: [b, n_head, s, d_nope + d_rope] = [b, n_head, s, d_k]
        k = torch.cat([k_nope, k_rope], dim=-1)

        # ── KV Cache ──────────────────────────────────────────
        # 缓存策略:
        #   - c_kv_cache: 存低维 c_kv(节省显存),需要时再上投影
        #   - k_rope_cache: 存 k_rope(解耦 RoPE,必须分开缓存)
        #
        # 为保持 KVCache 接口兼容(其 update 要求 [b, h, s, d]),
        # 将 c_kv 视为 h=1 的单头张量
        if use_cache:
            # 缓存 c_kv(reshape 为 [b, 1, s, d_c])
            c_kv_4d = c_kv.unsqueeze(1)                         # [b, 1, s, d_c]
            c_kv_4d, _ = self.c_kv_cache.update(c_kv_4d, c_kv_4d, start_pos)
            # 从完整历史 c_kv 重新上投影 K/V(包含历史 + 当前)
            c_kv_full = c_kv_4d.squeeze(1)                      # [b, cached_s, d_c]
            kv_full   = rearrange(
                self.kv_up_proj(c_kv_full),
                "b s (h d) -> b h s d",
                h=self.n_head
            )
            k_nope_full, v_full = kv_full.split([self.d_nope, self.d_k], dim=-1)

            # 缓存 k_rope
            k_rope, _ = self.k_rope_cache.update(k_rope, k_rope, start_pos)
            # k_rope 现在包含历史,形状 [b, n_head, cached_s, d_rope]

            # 重组完整 K
            k = torch.cat([k_nope_full, k_rope], dim=-1)        # [b, n_head, cached_s, d_k]
            v = v_full

            cached_len = self.c_kv_cache.get_seq_len()
            if s == 1:
                mask = torch.ones(1, cached_len, device=self.device, dtype=torch.bool)
            else:
                mask = torch.zeros(s, cached_len, device=self.device, dtype=torch.bool)
                if start_pos > 0:
                    mask[:, :start_pos] = True
                mask[:, start_pos:start_pos + s] = torch.tril(
                    torch.ones(s, s, device=self.device, dtype=torch.bool)
                )
        else:
            mask = torch.tril(torch.ones(s, s, device=self.device, dtype=torch.bool))

        # ── 注意力计算 ────────────────────────────────────────
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)

    def clear_cache(self):
        self.c_kv_cache.clear()
        self.k_rope_cache.clear()

    def get_cache_seq_len(self) -> int:
        return self.c_kv_cache.get_seq_len()

    def truncate_cache(self, length: int):
        self.c_kv_cache.truncate(length)
        self.k_rope_cache.truncate(length)