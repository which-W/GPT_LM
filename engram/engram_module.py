"""
Engram 条件记忆模块
基于 DeepSeek Engram 论文实现

核心特性:
1. N-gram 哈希嵌入 (2-gram 和 3-gram)
2. 词汇压缩 (tokenizer compression)
3. 多头哈希以减少碰撞
4. 上下文感知门控
5. 短卷积增强
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from rmsnorm import RMSNorm
from engram.tokeniercompress import TokenizerCompressor, FastTokenizerCompressor
    
class MultiHeadHashEmbedding(nn.Module):
    """
    多头哈希嵌入表
    使用多个哈希函数减少碰撞
    """
    
    def __init__(
        self,
        n_gram_size: int,
        n_heads: int,
        embed_dim_per_head: int,
        vocab_size: int,
        table_size: int = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            n_gram_size: N-gram的大小 (2或3)
            n_heads: 哈希头数量
            embed_dim_per_head: 每个头的嵌入维度
            vocab_size: 词汇表大小
            table_size: 哈希表大小 (建议使用质数)
        """
        super().__init__()
        self.n_gram_size = n_gram_size
        self.n_heads = n_heads
        self.embed_dim_per_head = embed_dim_per_head
        self.vocab_size = vocab_size
        
        # 使用质数作为表大小以减少碰撞
        if table_size is None:
            # 默认使用较大的质数
            table_size = self._find_prime(vocab_size ** n_gram_size // 100)
        
        self.table_size = table_size
        
        # 每个头一个嵌入表
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(
                table_size, 
                embed_dim_per_head,
                device=device,
                dtype=dtype
            ) for _ in range(n_heads)
        ])
        
        # 哈希系数 (用于多项式哈希)
        self.register_buffer(
            'hash_coeffs',
            torch.randint(1, table_size, (n_heads, n_gram_size), device=device)
        )
    
    def _find_prime(self, n: int) -> int:
        """找到大于n的最小质数"""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def hash_ngram(self, ngram_ids: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        使用多项式哈希计算N-gram的哈希值
        
        Args:
            ngram_ids: [batch_size, seq_len, n_gram_size] N-gram token IDs
            head_idx: 哈希头索引
            
        Returns:
            hash_indices: [batch_size, seq_len] 哈希索引
        """
        # 多项式哈希: hash = (c0*id0 + c1*id1 + ... + cn*idn) % table_size
        coeffs = self.hash_coeffs[head_idx]  # [n_gram_size]
        
        # 计算加权和
        hash_val = (ngram_ids * coeffs.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        # 取模
        hash_indices = hash_val % self.table_size
        
        return hash_indices
    
    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """
        从多个哈希头检索嵌入并拼接
        
        Args:
            ngram_ids: [batch_size, seq_len, n_gram_size]
            
        Returns:
            embeddings: [batch_size, seq_len, n_heads * embed_dim_per_head]
        """
        head_embeddings = []
        
        for head_idx in range(self.n_heads):
            # 计算哈希索引
            hash_indices = self.hash_ngram(ngram_ids, head_idx)
            
            # 检索嵌入
            emb = self.embedding_tables[head_idx](hash_indices)
            head_embeddings.append(emb)
        
        # 拼接所有头
        output = torch.cat(head_embeddings, dim=-1)
        
        return output


class EngramModule(nn.Module):
    """
    Engram 条件记忆模块
    
    包含:
    1. 词汇压缩
    2. 多个N-gram级别的哈希嵌入 (2-gram, 3-gram)
    3. 上下文感知门控
    4. 短深度卷积
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_ngram: int = 3,
        n_heads: int = 8,
        embed_dim: int = 1280,
        table_sizes: Optional[Dict[int, int]] = None,
        conv_kernel_size: int = 4,
        tokenizer_compressor: Optional[Union[TokenizerCompressor, FastTokenizerCompressor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            d_model: Transformer隐藏层维度
            vocab_size: 词汇表大小
            max_ngram: 最大N-gram大小 (论文中使用3)
            n_heads: 每个N-gram级别的哈希头数
            embed_dim: 每个头的嵌入维度
            table_sizes: 每个N-gram级别的哈希表大小
            conv_kernel_size: 卷积核大小
            tokenizer_compressor: 词汇压缩器实例(可选)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.n_heads = n_heads
        
        # 词汇压缩器
        if tokenizer_compressor is not None:
            self.compressor = tokenizer_compressor
            # 使用压缩后的词汇表大小构建哈希表
            effective_vocab_size = self.compressor.compressed_vocab_size
        else:
            self.compressor = TokenizerCompressor(vocab_size=vocab_size)
            effective_vocab_size = vocab_size
        
        # 为每个N-gram级别创建多头哈希嵌入
        self.ngram_embeddings = nn.ModuleDict()
        
        total_mem_dim = 0
        for n in range(2, max_ngram + 1):
            ts = table_sizes.get(n) if table_sizes else None
            self.ngram_embeddings[str(n)] = MultiHeadHashEmbedding(
                n_gram_size=n,
                n_heads=n_heads,
                embed_dim_per_head=embed_dim // n_heads,
                vocab_size=effective_vocab_size,  # 使用压缩后的词汇表大小
                table_size=ts,
                device=device,
                dtype=dtype
            )
            total_mem_dim += n_heads * (embed_dim // n_heads)
        
        self.total_mem_dim = total_mem_dim
        
        # 上下文感知门控
        # Query: 从当前隐藏状态
        self.query_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Key & Value: 从检索的记忆
        self.key_proj = nn.Linear(total_mem_dim, d_model, bias=False, device=device, dtype=dtype)
        self.value_proj = nn.Linear(total_mem_dim, d_model, bias=False, device=device, dtype=dtype)
        
        self.key_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 短深度卷积 (depthwise causal convolution),这个不是很重要可以去掉
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel_size,
            groups=d_model,  # depthwise
            padding=conv_kernel_size - 1,  # causal
            device=device,
            dtype=dtype
        )
        
        self.conv_norm = RMSNorm(d_model, device=device, dtype=dtype)
    
    def extract_ngrams(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        """
        提取N-gram序列
        
        Args:
            token_ids: [batch_size, seq_len]
            n: N-gram大小
            
        Returns:
            ngrams: [batch_size, seq_len, n]
        """
        batch_size, seq_len = token_ids.shape
        
        # 压缩token IDs
        compressed_ids = self.compressor.compress(token_ids)
        
        # 为前n-1个位置填充
        padded = F.pad(compressed_ids, (n - 1, 0), value=0)
        
        # 提取N-gram
        ngrams = []
        for i in range(n):
            ngrams.append(padded[:, i:i + seq_len])
        
        ngrams = torch.stack(ngrams, dim=-1)  # [batch_size, seq_len, n]
        
        return ngrams
    
    def retrieve_memory(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        从所有N-gram级别检索记忆嵌入
        
        Args:
            token_ids: [batch_size, seq_len]
            
        Returns:
            memory: [batch_size, seq_len, total_mem_dim]
        """
        memory_parts = []
        
        for n in range(2, self.max_ngram + 1):
            # 提取N-gram
            ngrams = self.extract_ngrams(token_ids, n)
            
            # 检索嵌入
            emb = self.ngram_embeddings[str(n)](ngrams)
            memory_parts.append(emb)
        
        # 拼接所有N-gram级别
        memory = torch.cat(memory_parts, dim=-1)
        
        return memory
    
    def context_aware_gating(
        self, 
        hidden_states: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        使用上下文感知门控调制记忆
        
        Args:
            hidden_states: [batch_size, seq_len, d_model] 当前隐藏状态
            memory: [batch_size, seq_len, total_mem_dim] 检索的记忆
            
        Returns:
            gated_values: [batch_size, seq_len, d_model]
        """
        # 计算query, key, value
        query = self.query_norm(hidden_states)  # [B, L, d_model]
        key = self.key_norm(self.key_proj(memory))  # [B, L, d_model]
        value = self.value_proj(memory)  # [B, L, d_model]
        
        # 计算门控分数 (缩放点积)
        gate_scores = (query * key).sum(dim=-1) / (self.d_model ** 0.5)  # [B, L]
        gate_weights = torch.sigmoid(gate_scores).unsqueeze(-1)  # [B, L, 1]
        
        # 门控调制
        gated_values = gate_weights * value
        
        return gated_values
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model] 当前层输入
            token_ids: [batch_size, seq_len] token索引
            
        Returns:
            output: [batch_size, seq_len, d_model] Engram输出
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 1. 检索记忆
        memory = self.retrieve_memory(token_ids)
        
        # 2. 上下文感知门控
        gated_values = self.context_aware_gating(hidden_states, memory)
        
        # 3. 短卷积 (with residual)
        # 转换维度用于Conv1d: [B, L, D] -> [B, D, L]
        conv_input = self.conv_norm(gated_values).transpose(1, 2)
        
        # 因果卷积 + 截断
        conv_output = self.conv(conv_input)[:, :, :seq_len]
        
        # 转回: [B, D, L] -> [B, L, D]
        conv_output = conv_output.transpose(1, 2)
        
        # SiLU激活 + 残差
        output = F.silu(conv_output) + gated_values
        
        return output


class EngramLayer(nn.Module):
    """
    可以插入到Transformer Block中的Engram层
    包含残差连接
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_ngram: int = 3,
        n_heads: int = 8,
        embed_dim: int = 1280,
        table_sizes: Optional[Dict[int, int]] = None,
        tokenizer_compressor: Optional[Union[TokenizerCompressor, FastTokenizerCompressor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.engram = EngramModule(
            d_model=d_model,
            vocab_size=vocab_size,
            max_ngram=max_ngram,
            n_heads=n_heads,
            embed_dim=embed_dim,
            table_sizes=table_sizes,
            tokenizer_compressor=tokenizer_compressor,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            token_ids: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Engram with residual connection
        engram_output = self.engram(hidden_states, token_ids)
        output = hidden_states + engram_output
        
        return output


if __name__ == "__main__":
    # 测试Engram模块
    print("测试 Engram 模块...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 参数
    batch_size = 2
    seq_len = 32
    d_model = 512
    vocab_size = 128
    
    # 创建Engram层
    engram_layer = EngramLayer(
        d_model=d_model,
        vocab_size=vocab_size,
        max_ngram=3,
        n_heads=8,
        embed_dim=32,
        device=device
    )
    
    # 测试输入
    hidden_states = torch.randn(batch_size, seq_len, d_model).to(device)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"输入 hidden_states: {hidden_states.shape}")
    print(f"输入 token_ids: {token_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = engram_layer(hidden_states, token_ids)
    
    print(f"输出: {output.shape}")
    print("✓ Engram模块测试通过!")
    
    # 计算参数量
    total_params = sum(p.numel() for p in engram_layer.parameters())
    print(f"\nEngram层参数量: {total_params:,}")