"""管理分页缓存的引用计数、前缀复用及空闲块回收。"""
from collections import deque
import numpy as np
import xxhash


class Block:
    """记录物理块的内容及引用次数。"""
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash, token_ids):
        self.hash = hash
        self.token_ids = list(token_ids)

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """只有完整的块参与前缀共享，部分块始终由一个序列独占。"""
    def __init__(self, num_blocks, block_size):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("缓存块数量和大小必须为正数")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
        self.hash_to_block_id = {}

    @staticmethod
    def compute_hash(token_ids, prefix=-1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def can_allocate(self, seq):
        # 保守估算能够避免在复用过程中回收后续将被复用的块。
        return len(self.free_block_ids) >= seq.num_blocks

    def _allocate_block(self, block_id):
        block = self.blocks[block_id]
        if block.ref_count:
            raise RuntimeError("不能分配仍被使用的缓存块")
        if self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.reset()
        return block

    def allocate(self, seq):
        if seq.block_table:
            raise RuntimeError("序列已经分配缓存块")
        prefix = -1
        for i in range(seq.num_blocks):
            tokens = seq.block(i)
            hash_value = self.compute_hash(tokens, prefix) if len(tokens) == self.block_size else -1
            hit = self.hash_to_block_id.get(hash_value) if hash_value != -1 else None
            if hit is not None and self.blocks[hit].token_ids == tokens:
                block = self.blocks[hit]
                if block.ref_count == 0:
                    self.free_block_ids.remove(hit)
                    self.used_block_ids.add(hit)
                block.ref_count += 1
                seq.num_cached_tokens += self.block_size
            else:
                hit = self.free_block_ids[0]
                block = self._allocate_block(hit)
                block.update(hash_value, tokens)
                if hash_value != -1:
                    self.hash_to_block_id[hash_value] = hit
            seq.block_table.append(hit)
            prefix = hash_value

    def deallocate(self, seq):
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq):
        return len(self.free_block_ids) >= max(0, seq.num_blocks - len(seq.block_table))

    def may_append(self, seq):
        if seq.num_blocks > len(seq.block_table):
            seq.block_table.append(self._allocate_block(self.free_block_ids[0]).block_id)
        if len(seq) % self.block_size == 0:
            block = self.blocks[seq.block_table[-1]]
            prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
            tokens = seq.block(seq.num_blocks - 1)
            value = self.compute_hash(tokens, prefix)
            block.update(value, tokens)
            self.hash_to_block_id[value] = block.block_id


# 保留旧接口名称，使用相同的正确性实现。
LazyBlockManager = BlockManager
