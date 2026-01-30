"""
MoE Transformer Language Model
完整的基于MoE的Transformer语言模型
"""
import torch
from torch import nn
from typing import Optional, List
from emb import CustomEmbedding
from rmsnorm import RMSNorm
from moe_transformer_block import MoETransformerBlock, HybridTransformerBlock


class MoETransformerLM(nn.Module):
    """
    MoE版本的Transformer语言模型
    
    特点:
    1. 所有层都使用MoE(比较普遍的用法)
    2. 统一的专家配置
    3. 聚合所有层的辅助损失
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
        # MoE参数
        n_experts: int = 8,
        top_k: int = 2,
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
            n_experts: 每层的专家数量
            top_k: 每个token激活的专家数
            use_moe_aux_loss: 是否使用负载均衡损失
            moe_aux_loss_weight: 辅助损失权重
            device_ids: GPU设备列表
            main_device: 主GPU设备ID
        """
        super().__init__()
        self.n_layer = n_layer
        self.use_moe_aux_loss = use_moe_aux_loss
        
        # 设置主设备
        self.main_device = torch.device(f"cuda:{main_device}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = dtype
        
        # Embedding层 (在主GPU上)
        self.embedding = CustomEmbedding(
            vocab_size, d_model, device=self.main_device, dtype=dtype
        )
        
        # 堆叠MoE Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            block = MoETransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_head=n_head,
                max_seq_len=max_seq_len,
                theta=theta,
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
            x = layer(x, token_position, use_cache=use_cache, start_pos=start_pos)
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


class HybridMoETransformerLM(nn.Module):
    """
    混合MoE Transformer语言模型
    
    特点:
    1. 可以指定哪些层使用MoE,哪些层使用标准FFN
    2. 灵活的架构配置
    3. 类似DeepSeek V2的稀疏激活策略
    
    例如: 可以配置为每4层中有2层是MoE层
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
        # MoE配置
        moe_layer_indices: Optional[List[int]] = None,  # 哪些层使用MoE
        n_experts: int = 8,
        top_k: int = 2,
        use_moe_aux_loss: bool = True,
        moe_aux_loss_weight: float = 0.01,
        # 多GPU参数
        device_ids: Optional[List[int]] = None,
        main_device: int = 0,
        # 其他参数
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rms_norm: bool = True,
    ):
        """
        Args:
            moe_layer_indices: 使用MoE的层索引列表
                               例如 [2, 5, 8, 11] 表示第2,5,8,11层使用MoE
                               如果为None,则所有层都使用MoE
        """
        super().__init__()
        self.n_layer = n_layer
        
        # 设置主设备
        self.main_device = torch.device(f"cuda:{main_device}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = dtype
        
        # 如果未指定,默认所有层都用MoE
        if moe_layer_indices is None:
            moe_layer_indices = list(range(n_layer))
        
        self.moe_layer_indices = set(moe_layer_indices)
        
        # Embedding
        self.embedding = CustomEmbedding(vocab_size, d_model, device=self.main_device, dtype=dtype)
        
        # 构建混合层
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            use_moe = (i in self.moe_layer_indices)
            
            block = HybridTransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_head=n_head,
                max_seq_len=max_seq_len,
                theta=theta,
                use_moe=use_moe,
                n_experts=n_experts,
                top_k=top_k,
                use_moe_aux_loss=use_moe_aux_loss,
                moe_aux_loss_weight=moe_aux_loss_weight,
                device_ids=device_ids if use_moe else None,
                main_device=main_device if use_moe else 0,
                device=self.main_device,
                dtype=dtype,
            )
            self.layers.append(block)
        
        # 输出层
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=self.main_device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()
        
        self.ln_output = nn.Linear(d_model, vocab_size, device=self.main_device, dtype=dtype)
        
        self.total_aux_loss = None
    
    def forward(self, token_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        b, s = token_ids.shape
        
        # 确保输入在主GPU上
        if token_ids.device != self.main_device:
            token_ids = token_ids.to(self.main_device)
        
        # 位置编码
        token_position = torch.arange(
            s, device=self.main_device, dtype=torch.long
        ).unsqueeze(0).expand(b, s)
        
        # Embedding
        x = self.embedding(token_ids)
        
        # 收集辅助损失
        aux_losses = []
        
        # 逐层传播
        for layer in self.layers:
            x = layer(x, token_position, use_cache=use_cache)
            if self.training:
                aux_loss = layer.get_aux_loss()
                if aux_loss.item() > 0:
                    aux_losses.append(aux_loss)
        
        # 聚合辅助损失
        if aux_losses:
            self.total_aux_loss = torch.stack(aux_losses).sum()
        else:
            self.total_aux_loss = torch.tensor(0.0, device=x.device)
        
        # 输出
        x = self.ln_final(x)
        logits = self.ln_output(x)
        
        return logits
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.total_aux_loss if self.total_aux_loss is not None else torch.tensor(0.0)
    
    def print_architecture(self):
        """打印模型架构信息"""
        print(f"Total layers: {self.n_layer}")
        print(f"MoE layers: {len(self.moe_layer_indices)}")
        print(f"Standard FFN layers: {self.n_layer - len(self.moe_layer_indices)}")
        print(f"MoE layer indices: {sorted(self.moe_layer_indices)}")
    
    def clear_cache(self):
        """清空所有层的KV Cache"""
        for layer in self.layers:
            layer.clear_cache()