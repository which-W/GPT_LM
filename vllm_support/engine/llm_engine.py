"""
LLM 推理引擎
整合调度器、模型运行器和采样逻辑
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Union
from collections import Counter
from utils.generation import sampling_probs
from vllm_support.vllm_transformer import PagedTransformerLM
from vllm_support.engine.scheduler import Scheduler
from vllm_support.engine.sequence import Sequence, SamplingParams


class ModelRunner:
    """
    模型运行器
    负责准备输入、执行模型、采样输出
    """
    def __init__(self, model: PagedTransformerLM, device: str = "cuda", tokenizer=None):
        self.model = model
        self.device = device
        self.block_size = model.block_size
        self.tokenizer = tokenizer

    def prepare_inputs(self, seqs: List[Sequence], is_prefill: bool):
        """
        准备模型输入
        
        返回：
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

            # 每行分别填充槽位，保持与输入展平顺序完全一致。
            slot_mapping = []
            for seq in seqs:
                slots = [seq.block_table[j // self.block_size] * self.block_size + j % self.block_size
                         for j in range(len(seq))]
                slot_mapping.extend(slots + [-1] * (max_len - len(seq)))

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
            ).unsqueeze(1)  # 张量形状或计算公式：[batch, 1]

            # slot_mapping: 每个新 token 的存储位置
            slot_mapping = []
            for seq in seqs:
                block_idx = (len(seq) - 1) // self.block_size
                offset = (len(seq) - 1) % self.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)

            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)

            # 物理块映射表
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
        
        返回：
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
            # 张量形状或计算公式：[batch, seq_len, vocab_size] -> [batch, vocab_size]
            last_logits = logits[torch.arange(len(seqs), device=logits.device), context_lens - 1]
        else:
            # 张量形状或计算公式：[batch, 1, vocab_size] -> [batch, vocab_size]
            last_logits = logits.squeeze(1)

        # 采样
        next_tokens = self.sample(last_logits, seqs)

        return next_tokens

    def sample(self, logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
        """
        从 logits 采样下一个 token
        
        参数：
            logits: [batch, vocab_size]
            seqs: 序列列表（用于获取采样参数）
        
        返回：
            token_ids: 采样得到的 token ID 列表
        """
        return [int(torch.multinomial(sampling_probs(
            logits[i], tokenizer=self.tokenizer, temperature=seq.temperature,
            top_k=seq.top_k, top_p=seq.top_p, repetition_penalty=seq.repetition_penalty,
            token_counts=Counter(seq.token_ids)), 1).item()) for i, seq in enumerate(seqs)]


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
        device: str = "cuda",
        tokenizer=None,
    ):
        """
        参数：
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
        self.model_runner = ModelRunner(model, device, tokenizer)

        # 设置 Sequence 的 block_size
        Sequence.block_size = block_size

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams = None
    ):
        """
        添加生成请求
        
        参数：
            prompt: token ID 列表（暂不支持字符串）
            sampling_params: 采样参数
        """
        if isinstance(prompt, str):
            raise NotImplementedError("暂不支持字符串输入，请直接传入 token ID 列表")

        if sampling_params is None:
            sampling_params = SamplingParams()

        if not prompt or len(prompt) >= self.model.config["max_seq_len"]:
            raise ValueError("提示词不能为空，且必须为生成保留上下文空间")
        if len(prompt) > self.scheduler.max_num_batched_tokens:
            raise ValueError("提示词长度超过批次令牌上限")
        capacity = len(self.scheduler.block_manager.blocks) * self.scheduler.block_manager.block_size
        if len(prompt) + sampling_params.max_tokens - 1 > min(capacity, self.model.config["max_seq_len"]):
            raise ValueError("请求长度超过模型上下文或单请求缓存容量")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理
        
        返回：
            outputs: 完成的序列列表 [(seq_id, completion_token_ids), ...]
            num_tokens: 本轮处理的 token 数（正数=Prefill，负数=Decode）
        """
        # 调度
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

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
        
        参数：
            prompts: token ID 列表的列表
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条
        
        返回：
            outputs: 生成结果列表，每个元素包含 'token_ids' 和可选的 'text'
        """
        if not prompts:
            return []
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        # 初始化进度条
        if use_tqdm:
           pbar = tqdm(
                total=sum(sp.max_tokens for sp in sampling_params),
                desc="Decoding",
                dynamic_ncols=True
            )


        # 处理采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)

        if len(sampling_params) != len(prompts):
            raise ValueError("提示词数量和采样参数数量不一致")
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
