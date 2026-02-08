"""
Tokenizer Compressor - 词汇压缩器
基于 DeepSeek Engram 论文实现

功能:
1. 将语义等价的token映射到规范形式
2. NFKC归一化
3. 大小写归一化
4. 去除前导空格/特殊字符的归一化
5. 支持HuggingFace Tokenizer

论文中提到对128k词汇表可以减少23%的有效词汇量
"""
import torch
import unicodedata
from typing import Dict, Optional, List, Union
from collections import defaultdict
import json


class TokenizerCompressor:
    """
    词汇压缩器
    
    核心思想:
    - 将语义等价但tokenizer认为不同的tokens归一化到相同ID
    - 减少N-gram嵌入表的有效大小
    - 提高哈希表的利用率
    
    示例:
        'Apple' → 'apple'
        ' apple' → 'apple'
        '␣␣apple' → 'apple'
        'APPLE' → 'apple'
    """
    
    def __init__(
        self,
        tokenizer=None,
        vocab_size: Optional[int] = None,
        enable_nfkc: bool = True,
        enable_lowercase: bool = True,
        enable_strip: bool = True,
        custom_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer对象
            vocab_size: 词汇表大小(如果不提供tokenizer则必须指定)
            enable_nfkc: 是否启用NFKC Unicode归一化
            enable_lowercase: 是否启用小写归一化
            enable_strip: 是否去除前导/尾随空格
            custom_mapping: 自定义token ID映射
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size or (len(tokenizer) if tokenizer else None)
        
        if self.vocab_size is None:
            raise ValueError("必须提供tokenizer或vocab_size")
        
        self.enable_nfkc = enable_nfkc
        self.enable_lowercase = enable_lowercase
        self.enable_strip = enable_strip
        
        # 构建压缩映射表: token_id → canonical_token_id
        self.canonical_map = None
        self.compressed_vocab_size = None
        
        # 如果提供了tokenizer,自动构建映射
        if tokenizer is not None:
            self._build_canonical_map(custom_mapping)
        elif custom_mapping is not None:
            self.canonical_map = custom_mapping
            self.compressed_vocab_size = len(set(custom_mapping.values()))
    
    def _normalize_text(self, text: str) -> str:
        """
        对文本进行归一化
        
        Args:
            text: 原始文本
            
        Returns:
            normalized_text: 归一化后的文本
        """
        # 1. NFKC归一化 (兼容性分解+兼容性组合)
        if self.enable_nfkc:
            text = unicodedata.normalize('NFKC', text)
        
        # 2. 小写化
        if self.enable_lowercase:
            text = text.lower()
        
        # 3. 去除前导/尾随空格和特殊空白字符
        if self.enable_strip:
            # 保留单个前导空格(因为很多tokenizer区分" word"和"word")
            # 但将多个空格归一化为单个空格
            leading_space = text.startswith(' ') or text.startswith('▁')
            text = text.strip()
            if leading_space and text:
                text = ' ' + text
        
        return text
    
    def _build_canonical_map(self, custom_mapping: Optional[Dict[int, int]] = None):
        """
        构建规范映射表
        
        策略:
        1. 对每个token进行归一化
        2. 将归一化后相同的tokens映射到最小的token ID
        3. 保留特殊tokens不变
        """
        if self.tokenizer is None:
            raise ValueError("需要tokenizer来构建映射表")
        
        print(f"构建Token压缩映射表...")
        print(f"原始词汇表大小: {self.vocab_size}")
        
        # 获取词汇表
        vocab = self.tokenizer.get_vocab()
        
        # 反向映射: id → token
        id_to_token = {idx: token for token, idx in vocab.items()}
        
        # 归一化后的文本 → 最小token ID列表
        normalized_to_ids = defaultdict(list)
        
        # 特殊token (不进行压缩)
        special_tokens = set()
        if hasattr(self.tokenizer, 'all_special_tokens'):
            special_tokens = set(self.tokenizer.all_special_ids)
        
        # 遍历所有tokens
        for token_id in range(self.vocab_size):
            token_str = id_to_token.get(token_id, "")
            
            # 特殊token保持不变
            if token_id in special_tokens:
                normalized_to_ids[f"__SPECIAL_{token_id}__"].append(token_id)
                continue
            
            # 归一化
            normalized = self._normalize_text(token_str)
            normalized_to_ids[normalized].append(token_id)
        
        # 构建映射: 每组相同归一化文本的tokens映射到组内最小ID
        self.canonical_map = {}
        for normalized_text, token_ids in normalized_to_ids.items():
            canonical_id = min(token_ids)  # 使用最小ID作为规范ID
            for token_id in token_ids:
                self.canonical_map[token_id] = canonical_id
        
        # 应用自定义映射(如果提供)
        if custom_mapping is not None:
            self.canonical_map.update(custom_mapping)
        
        # 计算压缩后的词汇表大小
        self.compressed_vocab_size = len(set(self.canonical_map.values()))
        
        compression_rate = (1 - self.compressed_vocab_size / self.vocab_size) * 100
        
        print(f"压缩后词汇表大小: {self.compressed_vocab_size}")
        print(f"压缩率: {compression_rate:.2f}%")
        print(f"映射表构建完成\n")
        
        # 显示一些示例
        self._show_compression_examples(id_to_token)
    
    def _show_compression_examples(self, id_to_token: Dict[int, str], n_examples: int = 10):
        """显示压缩示例"""
        print("压缩示例:")
        
        # 找到被压缩的token组
        canonical_to_group = defaultdict(list)
        for token_id, canonical_id in self.canonical_map.items():
            if token_id != canonical_id:  # 只显示被压缩的
                canonical_to_group[canonical_id].append(token_id)
        
        shown = 0
        for canonical_id, group in list(canonical_to_group.items())[:n_examples]:
            if shown >= n_examples:
                break
            
            canonical_token = id_to_token.get(canonical_id, "")
            group_tokens = [id_to_token.get(tid, "") for tid in group[:5]]  # 最多显示5个
            
            print(f"  [{canonical_id}] '{canonical_token}' ← {group_tokens}")
            shown += 1
        
        if len(canonical_to_group) > n_examples:
            print(f"  ... 还有 {len(canonical_to_group) - n_examples} 组")
        print()
    
    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        将token IDs压缩到规范形式
        
        Args:
            token_ids: [...] 任意形状的token ID张量
            
        Returns:
            compressed_ids: [...] 压缩后的token ID张量
        """
        if self.canonical_map is None:
            # 没有映射表,直接返回
            return token_ids
        
        # 保存原始形状和设备
        original_shape = token_ids.shape
        original_device = token_ids.device
        
        # 展平
        flat_ids = token_ids.flatten().cpu().numpy()
        
        # 应用映射
        compressed_flat = []
        for token_id in flat_ids:
            token_id = int(token_id)
            canonical_id = self.canonical_map.get(token_id, token_id)
            compressed_flat.append(canonical_id)
        
        # 转回张量并恢复形状
        compressed_ids = torch.tensor(compressed_flat, dtype=token_ids.dtype)
        compressed_ids = compressed_ids.reshape(original_shape).to(original_device)
        
        return compressed_ids
    
    def save_mapping(self, filepath: str):
        """保存映射表到文件"""
        mapping_data = {
            'vocab_size': self.vocab_size,
            'compressed_vocab_size': self.compressed_vocab_size,
            'canonical_map': {str(k): v for k, v in self.canonical_map.items()},
            'config': {
                'enable_nfkc': self.enable_nfkc,
                'enable_lowercase': self.enable_lowercase,
                'enable_strip': self.enable_strip
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"映射表已保存到: {filepath}")
    
    @classmethod
    def load_mapping(cls, filepath: str, tokenizer=None) -> 'TokenizerCompressor':
        """从文件加载映射表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # 转换映射表
        canonical_map = {int(k): v for k, v in mapping_data['canonical_map'].items()}
        
        config = mapping_data.get('config', {})
        
        compressor = cls(
            tokenizer=tokenizer,
            vocab_size=mapping_data['vocab_size'],
            enable_nfkc=config.get('enable_nfkc', True),
            enable_lowercase=config.get('enable_lowercase', True),
            enable_strip=config.get('enable_strip', True),
            custom_mapping=canonical_map
        )
        
        compressor.compressed_vocab_size = mapping_data['compressed_vocab_size']
        
        print(f"  映射表已从 {filepath} 加载")
        print(f"  原始词汇: {compressor.vocab_size}")
        print(f"  压缩后: {compressor.compressed_vocab_size}")
        
        return compressor
    
    def get_compression_stats(self) -> Dict:
        """获取压缩统计信息"""
        if self.canonical_map is None:
            return {}
        
        # 统计有多少token被压缩
        compressed_count = sum(1 for k, v in self.canonical_map.items() if k != v)
        
        return {
            'original_vocab_size': self.vocab_size,
            'compressed_vocab_size': self.compressed_vocab_size,
            'compression_rate': (1 - self.compressed_vocab_size / self.vocab_size) * 100,
            'num_compressed_tokens': compressed_count,
            'num_canonical_tokens': self.compressed_vocab_size
        }


class FastTokenizerCompressor:
    """
    快速版本的TokenizerCompressor
    使用预计算的查找表,在GPU上进行快速映射
    """
    
    def __init__(self, canonical_map: Dict[int, int], vocab_size: int, device: str = 'cpu'):
        """
        Args:
            canonical_map: 预计算的映射表
            vocab_size: 词汇表大小
            device: 映射表存储设备
        """
        self.vocab_size = vocab_size
        self.device = device
        
        # 构建查找表张量: lookup_table[token_id] = canonical_id
        self.lookup_table = torch.arange(vocab_size, dtype=torch.long, device=device)
        
        for token_id, canonical_id in canonical_map.items():
            if token_id < vocab_size:
                self.lookup_table[token_id] = canonical_id
        
        self.compressed_vocab_size = len(set(canonical_map.values()))
    
    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        快速压缩token IDs
        
        Args:
            token_ids: [...] 任意形状的token ID张量
            
        Returns:
            compressed_ids: [...] 压缩后的token ID张量
        """
        original_device = token_ids.device
        
        # 移动到lookup table所在设备
        if original_device != self.lookup_table.device:
            token_ids = token_ids.to(self.lookup_table.device)
        
        # 直接索引查找
        compressed_ids = self.lookup_table[token_ids]
        
        # 移回原设备
        if original_device != self.lookup_table.device:
            compressed_ids = compressed_ids.to(original_device)
        
        return compressed_ids


# 便利函数
def build_compressor_from_tokenizer(
    tokenizer,
    enable_nfkc: bool = True,
    enable_lowercase: bool = True,
    enable_strip: bool = True,
    use_fast: bool = True,
    device: str = 'cpu'
) -> Union[TokenizerCompressor, FastTokenizerCompressor]:
    """
    从HuggingFace tokenizer构建压缩器
    
    Args:
        tokenizer: HuggingFace tokenizer
        enable_nfkc: 启用NFKC归一化
        enable_lowercase: 启用小写化
        enable_strip: 启用空格处理
        use_fast: 使用快速版本(推荐用于训练)
        device: 快速版本的设备
        
    Returns:
        compressor: TokenizerCompressor或FastTokenizerCompressor
    """
    # 先构建标准版本获取映射
    standard_compressor = TokenizerCompressor(
        tokenizer=tokenizer,
        enable_nfkc=enable_nfkc,
        enable_lowercase=enable_lowercase,
        enable_strip=enable_strip
    )
    
    if use_fast:
        # 转换为快速版本
        return FastTokenizerCompressor(
            canonical_map=standard_compressor.canonical_map,
            vocab_size=standard_compressor.vocab_size,
            device=device
        )
    else:
        return standard_compressor


if __name__ == "__main__":
    print("TokenizerCompressor 测试")
    
    # 测试1: 使用HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        
        print("测试1: 使用GPT2 Tokenizer")
        print("-"*60)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # 构建压缩器
        compressor = TokenizerCompressor(
            tokenizer=tokenizer,
            enable_nfkc=True,
            enable_lowercase=True,
            enable_strip=True
        )
        
        # 测试压缩
        test_text = "Hello World! HELLO world"
        test_ids = tokenizer.encode(test_text, return_tensors='pt')
        
        print(f"原始文本: {test_text}")
        print(f"原始IDs: {test_ids}")
        print(f"原始tokens: {tokenizer.convert_ids_to_tokens(test_ids[0])}")
        
        compressed_ids = compressor.compress(test_ids)
        print(f"压缩后IDs: {compressed_ids}")
        print(f"压缩后tokens: {tokenizer.convert_ids_to_tokens(compressed_ids[0])}")
        
        # 统计信息
        stats = compressor.get_compression_stats()
        print(f"\n压缩统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # 保存和加载测试
        print("\n测试2: 保存和加载映射表")
        print("-"*60)
        
        compressor.save_mapping("tokenizer_mapping.json")
        loaded_compressor = TokenizerCompressor.load_mapping(
            "tokenizer_mapping.json",
            tokenizer=tokenizer
        )
        
        # 验证加载的压缩器
        recompressed_ids = loaded_compressor.compress(test_ids)
        assert torch.equal(compressed_ids, recompressed_ids), "加载的映射表不一致!"
        print(" 映射表保存和加载验证通过")
        
        # 测试3: 快速版本
        print("\n测试3: FastTokenizerCompressor")
        print("-"*60)
        
        fast_compressor = FastTokenizerCompressor(
            canonical_map=compressor.canonical_map,
            vocab_size=compressor.vocab_size,
            device='cpu'
        )
        
        fast_compressed_ids = fast_compressor.compress(test_ids)
        assert torch.equal(compressed_ids, fast_compressed_ids), "快速版本结果不一致!"
        print(" 快速版本验证通过")
        
        # 性能测试
        import time
        
        large_ids = torch.randint(0, len(tokenizer), (100, 512))
        
        # 标准版本
        start = time.time()
        for _ in range(100):
            _ = compressor.compress(large_ids)
        standard_time = time.time() - start
        
        # 快速版本
        start = time.time()
        for _ in range(100):
            _ = fast_compressor.compress(large_ids)
        fast_time = time.time() - start
        
        print(f"\n性能对比 (100次迭代, shape={large_ids.shape}):")
        print(f"  标准版本: {standard_time:.4f}s")
        print(f"  快速版本: {fast_time:.4f}s")
        print(f"  加速比: {standard_time/fast_time:.2f}x")
        
    except ImportError:
        print("需要安装transformers库: pip install transformers")
        print("跳过HuggingFace tokenizer测试\n")
        
        # 使用简单示例
        print("测试4: 简单示例(不使用HuggingFace)")
        print("-"*60)
        
        # 创建简单的映射
        simple_mapping = {
            0: 0,   # <pad>
            1: 1,   # <unk>
            100: 50,  # 'Apple' -> 'apple'
            101: 50,  # 'APPLE' -> 'apple'
            102: 50,  # ' apple' -> 'apple'
            200: 200, # 'hello' -> 'hello'
            201: 200, # 'Hello' -> 'hello'
        }
        
        compressor = TokenizerCompressor(
            vocab_size=1000,
            custom_mapping=simple_mapping
        )
        
        test_ids = torch.tensor([[100, 101, 102, 200, 201]])
        compressed = compressor.compress(test_ids)
        
        print(f"原始IDs: {test_ids}")
        print(f"压缩后: {compressed}")
        print(f"期望结果: tensor([[50, 50, 50, 200, 200]])")
        
        assert torch.equal(compressed, torch.tensor([[50, 50, 50, 200, 200]])), "简单映射测试失败!"
        print(" 简单映射测试通过")
    
    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)