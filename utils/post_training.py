"""为后训练提供共享模型评估，以及可选的独立 vLLM 评估。"""
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
import importlib.util
import os
import torch


@dataclass
class SamplingParams:
    """两个评估后端共同支持的采样设置。"""
    temperature: float = 0.0
    max_tokens: int = 512
    min_tokens: int = 0
    n: int = 1
    stop: list[str] = field(default_factory=list)
    include_stop_str_in_output: bool = True


class TransformersEvaluator:
    """直接复用训练模型，适用于单显卡、CPU 和 Windows。"""
    def __init__(self, policy, tokenizer):
        self.policy = policy
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompts, sampling_params, **kwargs):
        if sampling_params.n < 1 or sampling_params.max_tokens < 1:
            raise ValueError("候选数量和生成长度必须为正数")
        was_training = self.policy.training
        self.policy.eval()
        results = []
        try:
            for prompt in prompts:
                context = getattr(self.policy.config, "max_position_embeddings", 2048)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, return_token_type_ids=False,
                                        max_length=max(1, context - sampling_params.max_tokens))
                inputs = {k: v.to(self.policy.device) for k, v in inputs.items()}
                count = min(sampling_params.max_tokens, context - inputs["input_ids"].shape[1])
                candidates = []
                for _ in range(sampling_params.n):
                    options = dict(max_new_tokens=count, min_new_tokens=min(sampling_params.min_tokens, count),
                                   do_sample=sampling_params.temperature > 0, use_cache=True,
                                   pad_token_id=self.tokenizer.pad_token_id,
                                   eos_token_id=self.tokenizer.eos_token_id)
                    if sampling_params.temperature > 0:
                        options["temperature"] = sampling_params.temperature
                    generated = self.policy.generate(**inputs, **options)
                    ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    endings = [(text.find(stop), stop) for stop in sampling_params.stop if stop in text]
                    if endings:
                        index, stop = min(endings)
                        text = text[:index + (len(stop) if sampling_params.include_stop_str_in_output else 0)]
                    candidates.append(SimpleNamespace(text=text, token_ids=ids))
                results.append(SimpleNamespace(prompt=prompt, outputs=candidates))
            return results
        finally:
            self.policy.train(was_training)

    def sync_weights(self, policy):
        if self.policy is not policy:
            raise ValueError("共享评估后端必须使用同一个训练模型")


class VLLMEvaluator:
    """使用 V0 执行器的独立评估模型；不修改分布式或显存检查。"""
    def __init__(self, args):
        os.environ["VLLM_USE_V1"] = "0"
        from vllm import LLM, SamplingParams as VLLMSamplingParams
        self.params_class = VLLMSamplingParams
        self.llm = LLM(model=args.model_id, device=args.vllm_device, dtype="bfloat16",
                       enable_prefix_caching=True, gpu_memory_utilization=args.vllm_gpu_util,
                       seed=args.seed, max_model_len=2048)

    def generate(self, prompts, sampling_params, **kwargs):
        return self.llm.generate(prompts, self.params_class(**asdict(sampling_params)), **kwargs)

    def sync_weights(self, policy):
        worker = self.llm.llm_engine.model_executor.driver_worker
        worker = getattr(worker, "worker", worker)
        if not hasattr(worker, "model_runner"):
            raise RuntimeError("当前 vLLM 执行器不支持直接同步，请使用 --eval_backend transformers")
        worker.model_runner.model.load_weights(policy.state_dict().items())
        # 权重改变后，前缀缓存中的键值也必须失效。
        self.llm.llm_engine.reset_prefix_cache()


def init_evaluator(policy, tokenizer, args):
    """默认复用训练模型；显式指定 vLLM 时才创建第二份模型。"""
    if args.eval_backend == "transformers":
        return TransformersEvaluator(policy, tokenizer)
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("未安装 vLLM，请使用 --eval_backend transformers")
    device = torch.device(args.vllm_device)
    if device.type != "cuda" or (device.index or 0) >= torch.cuda.device_count():
        raise ValueError("vllm_device 指向不可用的显卡")
    return VLLMEvaluator(args)


def load_policy_into_vllm_instance(policy, evaluator):
    """兼容原训练入口的同步函数名。"""
    evaluator.sync_weights(policy)
