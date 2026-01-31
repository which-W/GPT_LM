"""
物理块管理器
负责分配和回收 KV Cache 的物理显存块
"""
from collections import deque
from typing import Tuple
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

class LazyBlockManager:
    """
    懒计算块管理器
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
    
    @staticmethod
    def compute_hash(token_ids: list, prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()
    
    def can_allocate(self, seq) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    def _get_or_compute_block_hash(self, seq, block_idx: int, prev_hash: int) -> Tuple[int, list]:
        """
        获取或计算块的哈希
        使用缓存避免重复计算
        """
        # 检查是否已缓存
        if hasattr(seq, '_block_hashes') and block_idx < len(seq._block_hashes):
            return seq._block_hashes[block_idx]
        
        # 初始化缓存
        if not hasattr(seq, '_block_hashes'):
            seq._block_hashes = []
        
        # 计算新哈希
        token_ids = seq.block(block_idx)
        
        if len(token_ids) == self.block_size:
            curr_hash = self.compute_hash(token_ids, prev_hash)
        else:
            curr_hash = -1
        
        result = (curr_hash, token_ids)
        seq._block_hashes.append(result)
        
        return result
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
    def allocate(self, seq):
        """
        懒计算版本的分配
        只在需要时计算哈希
        """
        assert not seq.block_table
        
        prev_hash = -1
        cache_hit_until = 0
        
        #查找阶段：懒计算哈希
        for i in range(seq.num_blocks):
            curr_hash, token_ids = self._get_or_compute_block_hash(seq, i, prev_hash)
            
            if curr_hash == -1:
                break  # 不完整块，停止查找
            
            # 查找缓存
            block_id = self.hash_to_block_id.get(curr_hash)
            if block_id is None:
                break  # 未命中，停止
            
            # 验证内容
            if self.blocks[block_id].token_ids != token_ids:
                break
            
            cache_hit_until = i + 1
            prev_hash = curr_hash
        
        # 分配阶段：复用已计算的哈希
        prev_hash = -1
        
        for i in range(seq.num_blocks):
            if i >= len(seq._block_hashes):
                prev_hash = seq._block_hashes[i-1][0] if i > 0 else -1
                self._get_or_compute_block_hash(seq, i, prev_hash)

            curr_hash, token_ids = seq._block_hashes[i]
            
            if i < cache_hit_until:
                # 使用缓存
                block_id = self.hash_to_block_id[curr_hash]
                block = self.blocks[block_id]
                
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.reset()
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                
                seq.num_cached_tokens += self.block_size
            
            else:
                # 分配新块
                block_id = self.free_block_ids[0]
                block = self.blocks[block_id]
                block.reset()
                block.token_ids = token_ids
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                
                if curr_hash != -1:
                    self.hash_to_block_id[curr_hash] = block_id
            
            seq.block_table.append(block_id)
    
    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        
        # 保留哈希缓存，下次可复用
        #if hasattr(seq, '_block_hashes'):
         #    seq._block_hashes.clear()
    
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

class BlockManager(LazyBlockManager):
    #启用懒启动
    pass
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
        assert not seq.block_table, "序列已经有块表了"
        
        num_blocks = seq.num_blocks
        
        #批量计算所有块的哈希
        block_info = []  # [(block_index, hash, token_ids), ...]
        prev_hash = -1
        
        for i in range(num_blocks):
            token_ids = seq.block(i)
            
            # 只对完整块计算哈希
            if len(token_ids) == self.block_size:
                curr_hash = self.compute_hash(token_ids, prev_hash)
                block_info.append((i, curr_hash, token_ids))
                prev_hash = curr_hash
            else:
                # 不完整的块，哈希设为 -1
                block_info.append((i, -1, token_ids))
                prev_hash = -1
        
        # 查找缓存（提前终止）
        cache_hit_until = 0  # 缓存命中到第几个块
        
        for i, curr_hash, token_ids in block_info:
            if curr_hash == -1:
                # 不完整的块，无法缓存
                break
            
            # 快速查找哈希表
            block_id = self.hash_to_block_id.get(curr_hash)
            
            if block_id is None:
                # 哈希未命中，立即终止
                break
            
            # 验证内容（防止哈希冲突）
            cached_tokens = self.blocks[block_id].token_ids
            
            # 快速检查：先比较长度
            if len(cached_tokens) != len(token_ids):
                break
            
            # 再比较内容
            if cached_tokens != token_ids:
                break
            
            # 缓存命中
            cache_hit_until = i + 1
        
        # 分配块 
        for i, curr_hash, token_ids in block_info:
            
            if i < cache_hit_until:
                # 使用缓存的块
                block_id = self.hash_to_block_id[curr_hash]
                block = self.blocks[block_id]
                
                # 增加引用计数或重新分配
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    # 块在缓存中但未被使用，重新分配
                    block.reset()
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                
                # 更新缓存计数
                seq.num_cached_tokens += self.block_size
            
            else:
                #分配新块
                block_id = self.free_block_ids[0]
                block = self.blocks[block_id]
                
                # 分配块
                block.reset()
                block.token_ids = token_ids
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                
                # 更新哈希表（只对完整块）
                if curr_hash != -1:
                    self.hash_to_block_id[curr_hash] = block_id
            
            # 添加到块表
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