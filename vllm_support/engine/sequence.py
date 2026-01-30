"""
序列状态管理
基于vllm，适配到你的模型
"""
from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass


class SequenceStatus(Enum):
    """序列状态"""
    WAITING = auto()   # 等待中
    RUNNING = auto()   # 运行中
    FINISHED = auto()  # 已完成


@dataclass
class SamplingParams:
    """采样参数"""
    temperature: float = 1.0
    max_tokens: int = 512
    ignore_eos: bool = True
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty:float= 1.2
    eos_token_id: int = 0 

class Sequence:
    """
    序列管理类
    管理每个生成请求的状态和数据
    """
    # 静态变量
    block_size = 16  # 每个物理块能存储的 token 数
    counter = count()  # 全局ID计数器
    
    def __init__(self, token_ids: list, sampling_params: SamplingParams = None):
        """
        Args:
            token_ids: 初始 token 列表（prompt）
            sampling_params: 采样参数
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        # 基础信息
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        
        # Token 数据
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        
      
        # PagedAttention 相关
        self.num_cached_tokens = 0  # 已缓存的 token 数
        self.block_table = []  # 物理块 ID 列表
        
        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.repetition_penalty = sampling_params.repetition_penalty
        self.eos_token_id = sampling_params.eos_token_id
    def __len__(self):
        """返回当前序列长度"""
        return self.num_tokens
    
    def __getitem__(self, key):
        """支持索引访问"""
        return self.token_ids[key]
    
    @property
    def is_finished(self):
        """是否已完成"""
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        """已生成的 token 数"""
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self):
        """原始 prompt"""
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self):
        """生成的内容"""
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_cached_blocks(self):
        """已缓存的完整块数"""
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
        """需要的总块数"""
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        """最后一个块中的 token 数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def block(self, i):
        """获取第 i 个块的 token"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]
    
    def append_token(self, token_id: int):
        """添加新生成的 token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1