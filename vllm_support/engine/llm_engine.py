"""
LLM 推理引擎
整合调度器、模型运行器和采样逻辑
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Union
from vllm_support.vllm_transformer import PagedTransformerLM
from vllm_support.engine.scheduler import Scheduler
from vllm_support.engine.sequence import Sequence, SamplingParams


class ModelRunner:
    """
    模型运行器
    负责准备输入、执行模型、采样输出
    """
    def __init__(self, model: PagedTransformerLM, device: str = "cuda"):
        self.model = model
        self.device = device
        self.block_size = 16  # 与 Sequence 保持一致
    
    def prepare_inputs(self, seqs: List[Sequence], is_prefill: bool):
        """
        准备模型输入
        
        Returns:
            input_tokens: [total_tokens] 或 [batch, seq_len]
            block_tables: [batch, max_num_blocks]
            slot_mapping: [total_tokens]
            context_lens: [batch]
        """
        batch_size = len(seqs)
        
        if is_prefill:
            # Prefill 阶段：需要处理完整的 prompt
            # 为了简化，这里使用 padding（实际 vLLM 使用变长处理）
            max_len = max(len(seq) for seq in seqs)
            input_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            
            for i, seq in enumerate(seqs):
                tokens = torch.tensor(seq.token_ids, dtype=torch.long, device=self.device)
                input_tokens[i, :len(seq)] = tokens
            
            # 构建 slot_mapping
            slot_mapping = []
            for seq in seqs:
                for i, block_id in enumerate(seq.block_table):
                    num_tokens = min(self.block_size, len(seq) - i * self.block_size)
                    for j in range(num_tokens):
                        slot_mapping.append(block_id * self.block_size + j)
            
            # Padding slots
            while len(slot_mapping) < batch_size * max_len:
                slot_mapping.append(-1)
            
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
            
            # 构建 block_tables
            max_num_blocks = max(len(seq.block_table) for seq in seqs)
            block_tables = torch.full(
                (batch_size, max_num_blocks), -1, dtype=torch.long, device=self.device
            )
            for i, seq in enumerate(seqs):
                block_tables[i, :len(seq.block_table)] = torch.tensor(
                    seq.block_table, dtype=torch.long, device=self.device
                )
            
            context_lens = torch.tensor(
                [len(seq) for seq in seqs], dtype=torch.long, device=self.device
            )
        
        else:
            # Decode 阶段：只处理最后一个 token
            input_tokens = torch.tensor(
                [seq.last_token for seq in seqs],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(1)  # [batch, 1]
            
            # slot_mapping: 每个新 token 的存储位置
            slot_mapping = []
            for seq in seqs:
                block_idx = (len(seq) - 1) // self.block_size
                offset = (len(seq) - 1) % self.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)
            
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
            
            # block_tables
            max_num_blocks = max(len(seq.block_table) for seq in seqs)
            block_tables = torch.full(
                (batch_size, max_num_blocks), -1, dtype=torch.long, device=self.device
            )
            for i, seq in enumerate(seqs):
                block_tables[i, :len(seq.block_table)] = torch.tensor(
                    seq.block_table, dtype=torch.long, device=self.device
                )
            
            context_lens = torch.tensor(
                [len(seq) - 1 for seq in seqs], dtype=torch.long, device=self.device
            )
        
        return input_tokens, block_tables, slot_mapping, context_lens
    
    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """
        运行模型并采样
        
        Returns:
            token_ids: 每个序列生成的 token ID
        """
        # 准备输入
        input_tokens, block_tables, slot_mapping, context_lens = self.prepare_inputs(
            seqs, is_prefill
        )
        
        # 执行模型
        with torch.no_grad():
            logits = self.model(
                input_tokens,
                is_prefill=is_prefill,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
            )
        
        # 取最后一个位置的 logits
        if is_prefill:
            # [batch, seq_len, vocab_size] -> [batch, vocab_size]
            last_logits = logits[:, -1, :]
        else:
            # [batch, 1, vocab_size] -> [batch, vocab_size]
            last_logits = logits.squeeze(1)
        
        # 采样
        next_tokens = self.sample(last_logits, seqs)
        
        return next_tokens
    
    def sample(self, logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
        """
        从 logits 采样下一个 token
        
        Args:
            logits: [batch, vocab_size]
            seqs: 序列列表（用于获取采样参数）
        
        Returns:
            token_ids: 采样得到的 token ID 列表
        """
        token_ids = []
        
        for i, seq in enumerate(seqs):
            # 应用温度
            logit = logits[i] / seq.temperature
            eos_id = seq.eos_token_id
            # 第一 token 绝对禁止 EOS
            if seq.num_completion_tokens == 0:
                logit[eos_id] = torch.finfo(logit.dtype).min
            # 重复惩罚
            if seq.repetition_penalty != 1.0:
                penalty = seq.repetition_penalty

                # 历史 token（prompt + 已生成）
                seen_tokens = set(seq.token_ids + seq.completion_token_ids)

                for t in seen_tokens:
                    if t < 0 or t >= logit.size(0):
                        continue
                    if logit[t] > 0:
                        logit[t] /= penalty
                    else:
                        logit[t] *= penalty

            # Top-k 采样
            if seq.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logit, min(seq.top_k, logit.size(-1)))
                logit = torch.full_like(logit, float('-inf'))
                logit.scatter_(0, top_k_indices, top_k_logits)
            
            # Top-p (nucleus) 采样
            if seq.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logit, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                
                # 找到累积概率超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > seq.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logit[indices_to_remove] = float('-inf')
            
            # Softmax 并采样
            probs = F.softmax(logit, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            token_ids.append(token_id)
        
        return token_ids


class LLMEngine:
    """
    LLM 推理引擎
    整合调度器和模型运行器
    """
    def __init__(
        self,
        model: PagedTransformerLM,
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 2048,
        eos_token_id: int = 0,
        device: str = "cuda"
    ):
        """
        Args:
            model: Transformer 模型
            num_kv_blocks: KV Cache 物理块数
            block_size: 每个块的大小
            max_num_seqs: 最大并发序列数
            max_num_batched_tokens: 最大批次 token 数
            eos_token_id: 结束符 ID
        """
        self.model = model.to(device)
        self.model.eval()
        
        # 调度器
        self.scheduler = Scheduler(
            num_kv_blocks=num_kv_blocks,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos_token_id=eos_token_id
        )
        
        # 模型运行器
        self.model_runner = ModelRunner(model, device)
        
        # 设置 Sequence 的 block_size
        Sequence.block_size = block_size
    
    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams = None
    ):
        """
        添加生成请求
        
        Args:
            prompt: token ID 列表（暂不支持字符串）
            sampling_params: 采样参数
        """
        if isinstance(prompt, str):
            raise NotImplementedError("暂不支持字符串输入，请直接传入 token ID 列表")
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    
    def step(self):
        """
        执行一步推理
        
        Returns:
            outputs: 完成的序列列表 [(seq_id, completion_token_ids), ...]
            num_tokens: 本轮处理的 token 数（正数=Prefill，负数=Decode）
        """
        # 调度
        seqs, is_prefill = self.scheduler.schedule()
        
        # 执行
        token_ids = self.model_runner.run(seqs, is_prefill)
        
        # 后处理
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集完成的序列
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs if seq.is_finished
        ]
        
        # 统计 token 数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens
    
    def is_finished(self):
        """是否所有请求都完成"""
        return self.scheduler.is_finished()
    
    def generate(
        self,
        prompts: List[List[int]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        use_tqdm: bool = True
    ) -> List[dict]:
        """
        批量生成
        
        Args:
            prompts: token ID 列表的列表
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条
        
        Returns:
            outputs: 生成结果列表，每个元素包含 'token_ids' 和可选的 'text'
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        # 初始化进度条
        if use_tqdm:
           pbar = tqdm(
                total=sampling_params[0].max_tokens,
                desc="Decoding",
                dynamic_ncols=True
            )

        
        # 处理采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 执行生成
        import time
        while not self.is_finished():
            t = time.time()
            output, num_tokens = self.step()
            if use_tqdm and num_tokens < 0:
                 pbar.update(-num_tokens)
            # 更新吞吐量
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (time.time() - t)
                else:
                    decode_throughput = -num_tokens / (time.time() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 保存完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # 排序并格式化输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        
        return outputs