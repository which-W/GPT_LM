"""
Engram + MoE Transformer Block
结合条件记忆(Engram)和混合专家(MoE)的Transformer架构
"""
import torch
from torch import nn
from typing import Optional, List
from attention import CauseMutiHeadAttention
from rmsnorm import RMSNorm
from moe_model.moe_layer import ExpertParallelMoELayer
from engram.engram_module import EngramLayer


class EngramMoETransformerBlock(nn.Module):
    """
    Engram + MoE Transformer Block
    
    结构 (Pre-norm架构):
    1. Engram 条件记忆模块 (可选, 仅在特定层)
    2. RMSNorm + 多头注意力 + 残差
    3. RMSNorm + MoE FFN + 残差
    
    设计原则:
    - Engram放在早期层(如第2层)以卸载静态模式重构
    - MoE处理动态推理
    - 最优分配: ~75-80% MoE专家, ~20-25% Engram记忆
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        # Engram参数
        use_engram: bool = False,
        engram_max_ngram: int = 3,
        engram_n_heads: int = 8,
        engram_embed_dim: int = 1280,
        # MoE参数
        n_experts: int = 8,
        top_k: int = 2,
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        # 通用参数
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN中间层维度
            n_head: 注意力头数
            max_seq_len: 最大序列长度
            theta: RoPE的theta参数
            vocab_size: 词汇表大小 (Engram需要)
            use_engram: 是否在此层使用Engram
            engram_max_ngram: Engram的最大N-gram (2或3)
            engram_n_heads: Engram哈希头数
            engram_embed_dim: Engram嵌入维度
            n_experts: MoE专家数量
            top_k: 每个token激活的专家数
            use_moe_aux_loss: 是否使用MoE负载均衡损失
            moe_aux_loss_weight: MoE辅助损失权重
            device_ids: GPU设备列表
            main_device: 主GPU设备ID
        """
        super().__init__()
        
        self.use_engram = use_engram
        main_dev = torch.device(f"cuda:{main_device}")
        
        # Engram条件记忆 (可选)
        if use_engram:
            self.engram = EngramLayer(
                d_model=d_model,
                vocab_size=vocab_size,
                max_ngram=engram_max_ngram,
                n_heads=engram_n_heads,
                embed_dim=engram_embed_dim,
                device=main_dev,
                dtype=dtype
            )
            self.ln_engram = RMSNorm(d_model=d_model, device=main_dev, dtype=dtype)
        else:
            self.engram = None
            self.ln_engram = None
        
        # 注意力在主GPU上
        self.attention = CauseMutiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            max_seq_size=max_seq_len,
            theta=theta,
            device=main_dev,
            dtype=dtype
        )
        
        # LayerNorm层
        self.ln1 = RMSNorm(d_model=d_model, device=main_dev, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=main_dev, dtype=dtype)
        
        # MoE在多GPU上
        self.moe = ExpertParallelMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            top_k=top_k,
            use_aux_loss=use_moe_aux_loss,
            aux_loss_weight=moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=main_device,
            dtype=dtype
        )
        
        self.n_experts = n_experts
        self.top_k = top_k
    
    def forward(
        self, 
        x: torch.Tensor,
        token_ids: torch.Tensor,
        x_position: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model] 输入张量
            token_ids: [batch_size, seq_len] token ID (Engram需要)
            x_position: [batch_size, seq_len] token位置索引
            use_cache: 是否使用KV缓存
            start_pos: 起始位置
        
        Returns:
            output: [batch_size, seq_len, d_model] 输出张量
        """
        # 0. Engram子层 (如果启用)
        # 论文建议放在早期层,在attention之前
        if self.use_engram:
            x = self.engram(self.ln_engram(x), token_ids)
        
        # 1. Attention子层 (pre-norm结构)
        x = x + self.attention(
            self.ln1(x), 
            token_position=x_position, 
            use_cache=use_cache, 
            start_pos=start_pos
        )
        
        # 2. MoE FFN子层
        x = x + self.moe(self.ln2(x))
        
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        获取MoE的辅助损失
        
        Returns:
            aux_loss: 标量张量
        """
        return self.moe.get_aux_loss()
    
    def clear_cache(self):
        """清空该层的 KV Cache"""
        self.attention.clear_cache()
    
    def get_cache_seq_len(self) -> int:
        """获取缓存序列长度"""
        return self.attention.get_cache_seq_len()


class AdaptiveEngramMoEBlock(nn.Module):
    """
    自适应 Engram + MoE Block
    
    特点:
    - 可以根据层深度自动决定是否使用Engram
    - 遵循论文中的放置策略: 早期层使用Engram效果最好
    - 支持多层Engram配置 (如第2层和第15层)
    """
    
    def __init__(
        self,
        layer_idx: int,
        engram_layer_indices: List[int],  # 使用Engram的层索引
        d_model: int,
        d_ff: int,
        n_head: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        # Engram参数
        engram_max_ngram: int = 3,
        engram_n_heads: int = 8,
        engram_embed_dim: int = 1280,
        # MoE参数
        n_experts: int = 8,
        top_k: int = 2,
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            layer_idx: 当前层的索引
            engram_layer_indices: 哪些层使用Engram (如 [2, 15])
        """
        super().__init__()
        
        # 判断当前层是否使用Engram
        use_engram = layer_idx in engram_layer_indices
        
        self.block = EngramMoETransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            n_head=n_head,
            max_seq_len=max_seq_len,
            theta=theta,
            vocab_size=vocab_size,
            use_engram=use_engram,
            engram_max_ngram=engram_max_ngram,
            engram_n_heads=engram_n_heads,
            engram_embed_dim=engram_embed_dim,
            n_experts=n_experts,
            top_k=top_k,
            use_moe_aux_loss=use_moe_aux_loss,
            moe_aux_loss_weight=moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=main_device,
            dtype=dtype
        )
        
        self.layer_idx = layer_idx
        self.use_engram = use_engram
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        x_position: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        return self.block(x, token_ids, x_position, use_cache, start_pos)
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.block.get_aux_loss()
    
    def clear_cache(self):
        self.block.clear_cache()
    
    def get_cache_seq_len(self) -> int:
        return self.block.get_cache_seq_len()


if __name__ == "__main__":
    # 测试 Engram + MoE Block
    print("测试 Engram + MoE Transformer Block...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 参数设置
    batch_size = 2
    seq_len = 16
    d_model = 512
    vocab_size = 128
    
    # 创建带Engram的Block
    block_with_engram = EngramMoETransformerBlock(
        d_model=d_model,
        d_ff=2048,
        n_head=8,
        max_seq_len=128,
        theta=10000,
        vocab_size=vocab_size,
        use_engram=True,
        engram_max_ngram=3,
        engram_n_heads=8,
        engram_embed_dim=256,
        n_experts=4,
        top_k=2,
        device_ids=[0] if torch.cuda.is_available() else None,
        main_device=0
    )
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    x_position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(device)
    
    print(f"输入shape: {x.shape}")
    print(f"Token IDs shape: {token_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = block_with_engram(x, token_ids, x_position)
    
    print(f"输出shape: {output.shape}")
    
    aux_loss = block_with_engram.get_aux_loss()
    print(f"MoE辅助损失: {aux_loss.item():.6f}")
    
    # 统计参数
    total_params = sum(p.numel() for p in block_with_engram.parameters())
    engram_params = sum(p.numel() for p in block_with_engram.engram.parameters()) if block_with_engram.use_engram else 0
    moe_params = sum(p.numel() for p in block_with_engram.moe.parameters())
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  Engram参数: {engram_params:,} ({engram_params/total_params*100:.1f}%)")
    print(f"  MoE参数: {moe_params:,} ({moe_params/total_params*100:.1f}%)")
    
    print("\n Engram + MoE Block 测试通过!")
