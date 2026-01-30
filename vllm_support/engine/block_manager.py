"""
物理块管理器
负责分配和回收 KV Cache 的物理显存块
"""
from collections import deque
import xxhash
import numpy as np
from vllm_support.engine.sequence import Sequence


class Block:
    """单个物理块"""
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 引用计数
        self.hash = -1  # 用于前缀缓存
        self.token_ids = []
    
    def update(self, hash: int, token_ids: list):
        """更新块的哈希和内容"""
        self.hash = hash
        self.token_ids = token_ids
    
    def reset(self):
        """重置块"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    物理块管理器
    负责 KV Cache 显存块的分配、回收和前缀缓存
    """
    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 总物理块数
            block_size: 每个块的大小（token 数）
        """
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}  # 前缀哈希到块ID的映射
        self.free_block_ids = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids = set()  # 已使用的块集合
    
    @classmethod
    def compute_hash(cls, token_ids: list, prefix: int = -1):
        """计算 token 序列的哈希（用于前缀缓存）"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_block(self, block_id: int) -> Block:
        """分配一个块"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _deallocate_block(self, block_id: int):
        """回收一个块"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    
    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块"""
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate(self, seq: Sequence):
        """
        为序列分配物理块
        支持前缀缓存：如果发现相同的前缀，直接复用
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            
            # 只对完整的块计算哈希
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1
            
            # 检查是否命中缓存
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # 缓存未命中，分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块已被使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块未使用，分配它
                    block = self._allocate_block(block_id)
            
            # 更新块信息
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)
    
    def deallocate(self, seq: Sequence):
        """回收序列占用的所有块"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """检查是否能为序列追加新 token"""
        # 如果当前块满了，需要新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    
    def may_append(self, seq: Sequence):
        """
        可能需要为序列分配新块（当当前块满时）
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # 当前块刚满，需要新块
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        
        elif len(seq) % self.block_size == 0:
            # 块刚好填满，计算哈希
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        
        else:
            # 块未满，无需操作
            assert last_block.hash == -1