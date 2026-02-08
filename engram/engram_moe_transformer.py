"""
Engram + MoE Transformer 语言模型
结合条件记忆和混合专家的完整Transformer架构

基于 DeepSeek Engram 论文:
- 最优容量分配: 75-80% MoE, 20-25% Engram
- Engram放置策略: 早期层效果最好 (如第2层和第15层)
- U型缩放定律: 混合架构优于纯MoE
"""
import torch
from torch import nn
from typing import Optional, List
from emb import CustomEmbedding
from rmsnorm import RMSNorm
from engram.engram_moe_transformer_block import EngramMoETransformerBlock, AdaptiveEngramMoEBlock


class EngramMoETransformerLM(nn.Module):
    """
    Engram + MoE Transformer 语言模型
    
    特点:
    1. 在特定层插入Engram条件记忆模块
    2. 所有层使用MoE进行动态推理
    3. 遵循论文的最优分配比例
    4. 聚合所有层的辅助损失
    
    论文配置示例 (Engram-27B):
    - 30层Transformer
    - Engram在第2层和第15层
    - 55个routed experts + 2个shared experts (top-6)
    - 5.7B Engram参数
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
        # Engram配置
        engram_layer_indices: List[int] = [2, 15],  # 使用Engram的层
        engram_max_ngram: int = 3,
        engram_n_heads: int = 8,
        engram_embed_dim: int = 1280,
        # MoE参数
        n_experts: int = 55,  # 论文中减少了专家数以分配给Engram
        top_k: int = 6,  # DeepSeek配置
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        # 其他参数
        dtype: Optional[torch.dtype] = None,
        use_rms_norm: bool = True,
    ):
        """
        Args:
            d_model: 模型维度
            n_head: 注意力头数
            vocab_size: 词表大小
            max_seq_len: 最大序列长度
            d_ff: FFN中间层维度
            theta: RoPE theta参数
            n_layer: Transformer层数
            engram_layer_indices: 使用Engram的层索引列表
            engram_max_ngram: Engram的最大N-gram (2或3)
            engram_n_heads: Engram哈希头数
            engram_embed_dim: Engram嵌入维度
            n_experts: MoE专家数量 (论文: 从72减到55以分配给Engram)
            top_k: 每个token激活的专家数 (论文: 6)
            use_moe_aux_loss: 是否使用负载均衡损失
            moe_aux_loss_weight: 辅助损失权重
            device_ids: GPU设备列表
            main_device: 主GPU设备ID
        """
        super().__init__()
        self.n_layer = n_layer
        self.use_moe_aux_loss = use_moe_aux_loss
        self.engram_layer_indices = engram_layer_indices
        
        # 设置主设备
        self.main_device = torch.device(f"cuda:{main_device}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = dtype
        
        # Embedding层 (在主GPU上)
        self.embedding = CustomEmbedding(
            vocab_size, d_model, device=self.main_device, dtype=dtype
        )
        
        # 堆叠 Engram + MoE Transformer blocks
        self.layers = nn.ModuleList()
        for layer_idx in range(n_layer):
            block = AdaptiveEngramMoEBlock(
                layer_idx=layer_idx,
                engram_layer_indices=engram_layer_indices,
                d_model=d_model,
                d_ff=d_ff,
                n_head=n_head,
                max_seq_len=max_seq_len,
                theta=theta,
                vocab_size=vocab_size,
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
            self.layers.append(block)
        
        # 最终输出层 (在主GPU上)
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=self.main_device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()
        
        # 输出投影到词表 (在主GPU上)
        self.ln_output = nn.Linear(d_model, vocab_size, device=self.main_device, dtype=dtype)
        
        # 存储总辅助损失
        self.total_aux_loss = None
        
        # 位置计数器 (用于KV缓存)
        self._current_pos = 0
        
        # 打印架构信息
        self._print_architecture()
    
    def _print_architecture(self):
        """打印模型架构信息"""
        print(f"\n{'='*60}")
        print(f"Engram + MoE Transformer LM 架构")
        print(f"{'='*60}")
        print(f"总层数: {self.n_layer}")
        print(f"Engram层数: {len(self.engram_layer_indices)}")
        print(f"Engram层索引: {self.engram_layer_indices}")
        print(f"MoE专家数: {self.layers[0].block.n_experts}")
        print(f"Top-K: {self.layers[0].block.top_k}")
        
        # 计算参数分配
        total_params = self.get_num_params(non_embedding=True)
        engram_params = sum(
            sum(p.numel() for p in layer.block.engram.parameters())
            for layer in self.layers if layer.use_engram
        )
        moe_params = sum(
            sum(p.numel() for p in layer.block.moe.parameters())
            for layer in self.layers
        )
        
        print(f"\n参数分配:")
        print(f"  总参数 (不含embedding): {total_params:,}")
        print(f"  Engram参数: {engram_params:,} ({engram_params/(engram_params+moe_params)*100:.1f}%)")
        print(f"  MoE参数: {moe_params:,} ({moe_params/(engram_params+moe_params)*100:.1f}%)")
        print(f"  论文推荐分配: MoE 75-80%, Engram 20-25%")
        print(f"{'='*60}\n")
    
    def forward(self, token_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_ids: [batch_size, seq_len] token索引
            use_cache: 是否使用KV缓存
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] 输出logits
        """
        b, s = token_ids.shape
        
        # 确保输入在主GPU上
        if token_ids.device != self.main_device:
            token_ids = token_ids.to(self.main_device)
        
        # 生成位置编码
        if use_cache:
            start_pos = self._current_pos
            token_position = torch.arange(
                start_pos, start_pos + s,
                device=self.main_device, dtype=torch.long
            ).unsqueeze(0).expand(b, s)
            self._current_pos += s
        else:
            start_pos = 0
            token_position = torch.arange(
                s, device=self.main_device, dtype=torch.long
            ).unsqueeze(0).expand(b, s)
        
        # Embedding
        x = self.embedding(token_ids)
        
        # 收集所有层的辅助损失
        aux_losses = []
        
        # 逐层前向传播
        for layer in self.layers:
            x = layer(x, token_ids, token_position, use_cache=use_cache, start_pos=start_pos)
            if self.use_moe_aux_loss and self.training:
                aux_losses.append(layer.get_aux_loss())
        
        # 聚合辅助损失
        if aux_losses:
            self.total_aux_loss = torch.stack(aux_losses).sum()
        else:
            self.total_aux_loss = torch.tensor(0.0, device=x.device)
        
        # 最终归一化
        x = self.ln_final(x)
        
        # 投影到词表
        logits = self.ln_output(x)
        
        return logits
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        获取聚合的辅助损失
        训练时应该加到主损失中: total_loss = lm_loss + aux_loss
        
        Returns:
            total_aux_loss: 所有层的辅助损失之和
        """
        return self.total_aux_loss if self.total_aux_loss is not None else torch.tensor(0.0)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        计算模型参数量
        
        Args:
            non_embedding: 是否排除embedding层参数
        
        Returns:
            n_params: 参数数量
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params
    
    def clear_cache(self):
        """清空所有层的KV Cache"""
        self._current_pos = 0
        for layer in self.layers:
            layer.clear_cache()
    
    def get_architecture_info(self) -> dict:
        """获取架构信息"""
        total_params = self.get_num_params(non_embedding=True)
        engram_params = sum(
            sum(p.numel() for p in layer.block.engram.parameters())
            for layer in self.layers if layer.use_engram
        )
        moe_params = sum(
            sum(p.numel() for p in layer.block.moe.parameters())
            for layer in self.layers
        )
        
        return {
            'total_layers': self.n_layer,
            'engram_layers': len(self.engram_layer_indices),
            'engram_layer_indices': self.engram_layer_indices,
            'n_experts': self.layers[0].block.n_experts,
            'top_k': self.layers[0].block.top_k,
            'total_params': total_params,
            'engram_params': engram_params,
            'moe_params': moe_params,
            'engram_ratio': engram_params / (engram_params + moe_params),
            'moe_ratio': moe_params / (engram_params + moe_params)
        }


class FlexibleEngramMoELM(nn.Module):
    """
    灵活配置的 Engram + MoE 语言模型
    
    支持:
    1. 自定义Engram放置策略
    2. 不同层可以有不同的MoE配置
    3. 渐进式Engram深度策略
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
        # Engram策略
        engram_placement_strategy: str = 'early',  # 'early', 'distributed', 'custom'
        custom_engram_layers: Optional[List[int]] = None,
        n_engram_layers: int = 2,  # 使用多少个Engram层
        # MoE参数
        n_experts: int = 55,
        top_k: int = 6,
        # 其他参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            engram_placement_strategy: Engram放置策略
                - 'early': 在早期层 (如 [2])
                - 'distributed': 均匀分布 (如 [2, 15])
                - 'custom': 自定义位置
            custom_engram_layers: 自定义Engram层索引
            n_engram_layers: 使用的Engram层数量
        """
        super().__init__()
        
        # 确定Engram层位置
        if engram_placement_strategy == 'custom' and custom_engram_layers:
            engram_indices = custom_engram_layers
        elif engram_placement_strategy == 'early':
            engram_indices = [2] if n_engram_layers == 1 else [2, 5]
        elif engram_placement_strategy == 'distributed':
            # 均匀分布
            step = n_layer // (n_engram_layers + 1)
            engram_indices = [step * (i + 1) for i in range(n_engram_layers)]
        else:
            engram_indices = [2]  # 默认
        
        # 创建主模型
        self.model = EngramMoETransformerLM(
            d_model=d_model,
            n_head=n_head,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_ff=d_ff,
            theta=theta,
            n_layer=n_layer,
            engram_layer_indices=engram_indices,
            n_experts=n_experts,
            top_k=top_k,
            device_ids=device_ids,
            main_device=main_device,
            dtype=dtype
        )
    
    def forward(self, token_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        return self.model(token_ids, use_cache)
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.model.get_aux_loss()
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        return self.model.get_num_params(non_embedding)
    
    def clear_cache(self):
        self.model.clear_cache()


if __name__ == "__main__":
    print("测试 Engram + MoE Transformer LM...")
    
    # 检查GPU
    n_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {n_gpus}")
    
    # 创建模型 (小规模测试)
    model = EngramMoETransformerLM(
        d_model=512,
        n_head=8,
        vocab_size=10000,
        max_seq_len=256,
        d_ff=2048,
        theta=10000,
        n_layer=4,
        engram_layer_indices=[1],  # 第1层使用Engram
        engram_max_ngram=2,
        engram_n_heads=2,
        engram_embed_dim=64,
        n_experts=8,
        top_k=2,
        device_ids=list(range(min(2, max(1, n_gpus)))),
        main_device=0
    )
    
    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 2
    seq_len = 32
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    token_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    
    print(f"输入shape: {token_ids.shape}")
    
    with torch.no_grad():
        logits = model(token_ids)
    
    print(f"输出logits shape: {logits.shape}")
    
    aux_loss = model.get_aux_loss()
    print(f"MoE辅助损失: {aux_loss.item():.6f}")
    
    # 架构信息
    info = model.get_architecture_info()
    print(f"\n架构验证:")
    print(f"  实际分配 - MoE: {info['moe_ratio']*100:.1f}%, Engram: {info['engram_ratio']*100:.1f}%")
    print(f"  论文推荐 - MoE: 75-80%, Engram: 20-25%")
    
    print("\n Engram + MoE Transformer LM 测试通过!")
