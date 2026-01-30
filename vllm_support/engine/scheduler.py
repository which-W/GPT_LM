"""
调度器
负责决定每一轮推理运行哪些序列
"""
from collections import deque
import torch
from sequence import Sequence, SequenceStatus
from block_manager import BlockManager


class Scheduler:
    """
    vLLM 风格的调度器
    负责：
    1. 管理等待和运行队列
    2. 根据显存块可用性调度任务
    3. 处理抢占（当显存不足时）
    """
    def __init__(
        self,
        num_kv_blocks: int,
        block_size: int,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 2048,
        eos_token_id: int = 0
    ):
        """
        Args:
            num_kv_blocks: KV Cache 物理块总数
            block_size: 每个块的大小
            max_num_seqs: 单批次最大序列数
            max_num_batched_tokens: 单批次最大 token 数
            eos_token_id: 结束符 ID
        """
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        
        # 块管理器
        self.block_manager = BlockManager(num_kv_blocks, block_size)
        
        # 任务队列
        self.waiting = deque()  # 等待队列
        self.running = deque()  # 运行队列
    
    def is_finished(self):
        """是否所有任务都完成"""
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        """添加新请求"""
        self.waiting.append(seq)
    
    def schedule(self):
        """
        核心调度逻辑
        
        Returns:
            scheduled_seqs: 本轮要运行的序列列表
            is_prefill: True=Prefill阶段,False=Decode阶段
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # 优先处理 Prefill（新请求)
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 检查资源
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
                not self.block_manager.can_allocate(seq)):
                break
            
            # 分配资源
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            
            # 更新状态
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果有 Prefill 任务，本轮只做 Prefill
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # 处理 Decode（继续生成）
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否能追加新 token
            while not self.block_manager.can_append(seq):
                # 显存不足，需要抢占
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # 有足够显存
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs
        # 把取出的序列放回队列头部
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    
    def preempt(self, seq: Sequence):
        """抢占：暂停序列，回收显存"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)  # 插队到等待队列最前面
    
    def postprocess(self, seqs: list, token_ids: list):
        """
        后处理：更新序列状态
        
        Args:
            seqs: 本轮运行的序列
            token_ids: 生成的 token ID
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # 检查是否结束
            if ((not seq.ignore_eos and token_id == self.eos) or 
                seq.num_completion_tokens >= seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)