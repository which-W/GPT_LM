from pathlib import Path
from typing import List
import torch
import argparse
from checkpoint_use import get_checkpoint_config, load_tokenizer
from utils.generation import MODEL_KEYS
from vllm_support.vllm_transformer import PagedTransformerLM
from vllm_support.engine.llm_engine import LLMEngine
from vllm_support.engine.sequence import  SamplingParams
"""
vLLM 风格的推理脚本
完全兼容原训练检查点，无需重新训练

使用方法:
    # 交互模式
    python -m inference.vllm_inference --model_path checkpoints/checkpoint_final.pt --tokenizer_path tokenizer.json

    # 单次生成
    python -m inference.vllm_inference --model_path checkpoints/checkpoint_final.pt --tokenizer_path tokenizer.json --prompt "你好"

    # 批量生成
    python -m inference.vllm_inference --model_path checkpoints/checkpoint_final.pt --tokenizer_path tokenizer.json --batch_mode
"""
class VLLMTextGenerator:
    """使用项目自研分页缓存引擎的文本生成器。"""

    def __init__(self, model_path, tokenizer_path, device='cuda', num_kv_blocks=1024, dtype=torch.float16):
        """
        初始化生成器
        
        参数：
            model_path: 模型 checkpoint 路径
            tokenizer_path: tokenizer 文件路径
            device: 运行设备
            num_kv_blocks: KV Cache 物理块数（根据显存调整）
            dtype: 数据类型（推荐 float16）
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.config = get_checkpoint_config(checkpoint)
        if self.config.get("model_type", "dense") != "dense":
            raise ValueError("分页推理当前仅支持基础 Transformer 检查点")
        self.tokenizer = load_tokenizer(checkpoint, tokenizer_path, self.config["vocab_size"])
        self.vocab_size = self.config["vocab_size"]
        self.model = PagedTransformerLM(**{key: self.config[key] for key in MODEL_KEYS},
                                       num_kv_blocks=num_kv_blocks, block_size=16)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)

        # 转换并移动到目标设备
        self.model = self.model.to(dtype=dtype, device=self.device)
        self.model.eval()

        # 创建 vLLM 引擎
        self.engine = LLMEngine(
            model=self.model,
            num_kv_blocks=num_kv_blocks,
            block_size=16,
            max_num_seqs=16,
            device=self.device,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.token_to_id("<|endoftext|>") or 0,
        )

        print("模型加载完成!")
        print(f"  参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  KV Cache 块数: {num_kv_blocks}")
        print(f"  最大并发: 16 sequences")

    def generate(self, prompt: str, max_new_tokens=500, temperature=0.8,
                 top_k=20, top_p=0.9, repetition_penalty=1.2):
        """
        生成文本
        
        参数：
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: top-k 采样
            top_p: nucleus 采样
            repetition_penalty: 重复惩罚系数
        
        返回：
            生成的完整文本
        """
        # 编码输入
        encoding = self.tokenizer.encode(prompt)
        prompt_tokens = encoding.ids
        if max_new_tokens < 0:
            raise ValueError("生成长度不能为负数")
        if max_new_tokens == 0:
            return self.tokenizer.decode(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("提示词不能为空")
        prompt_tokens = prompt_tokens[-max(1, self.config["max_seq_len"] - max_new_tokens):]
        max_new_tokens = min(max_new_tokens, self.config["max_seq_len"] - len(prompt_tokens))

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.engine.scheduler.eos,
        )

        # 生成
        outputs = self.engine.generate([prompt_tokens], sampling_params)

        # 解码
        generated_ids = outputs[0]["token_ids"]

        eos_id = self.engine.scheduler.eos
        generated_ids = [
            t for t in generated_ids if t != eos_id
        ]
        full_tokens = prompt_tokens + generated_ids
        generated_text = self.tokenizer.decode(full_tokens)

        return generated_text

    def batch_generate(self, prompts: List[str], **kwargs):
        """批量生成"""
        if not prompts:
            return []
        max_new_tokens = kwargs.get("max_new_tokens", 250)
        if max_new_tokens < 0:
            raise ValueError("生成长度不能为负数")
        if max_new_tokens == 0:
            return [self.tokenizer.decode(self.tokenizer.encode(p).ids) for p in prompts]
        budget = min(max_new_tokens, self.config["max_seq_len"] - 1)
        token_lists = [self.tokenizer.encode(p).ids[-(self.config["max_seq_len"] - budget):] for p in prompts]
        params = SamplingParams(temperature=kwargs.get("temperature", 0.8), max_tokens=budget,
                                top_p=kwargs.get("top_p", 0.9), top_k=kwargs.get("top_k", 20),
                                repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                                eos_token_id=self.engine.scheduler.eos)
        outputs = self.engine.generate(token_lists, params)
        return [self.tokenizer.decode(tokens + output["token_ids"]) for tokens, output in zip(token_lists, outputs)]

    def interactive_mode(self):
        """交互式生成模式"""
        print("vLLM 交互模式 (输入 'quit' 退出)")


        while True:
            try:
                prompt = input("\n请输入提示文本: ").strip()

                if prompt.lower() == 'quit':
                    print("退出交互模式")
                    break

                if not prompt:
                    continue

                print("\n生成中...")
                generated_text = self.generate(
                    prompt=prompt,
                    max_new_tokens=250,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.2
                )

                print("\n生成结果:")
                print(generated_text)

            except KeyboardInterrupt:
                print("\n\n退出交互模式")
                break
            except Exception as e:
                print(f"生成出错: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='vLLM 推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json', help='tokenizer 文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda/cpu)')
    parser.add_argument('--num_kv_blocks', type=int, default=1024, help='KV Cache 物理块数')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'float16', 'bfloat16'])

    # 生成参数
    parser.add_argument('--prompt', type=str, default=None, help='输入提示文本')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top_k', type=int, default=20, help='top-k 采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='nucleus 采样')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='重复惩罚系数')

    # 批量模式
    parser.add_argument('--batch_mode', action='store_true', help='批量生成模式')
    parser.add_argument('--prompts_file', type=str, help='批量 prompts 文件（每行一个 prompt）')

    args = parser.parse_args()

    # 检查文件
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    if not Path(args.tokenizer_path).exists():
        print(f"错误: Tokenizer 文件不存在: {args.tokenizer_path}")
        return

    # 数据类型映射
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # 初始化生成器
    generator = VLLMTextGenerator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        num_kv_blocks=args.num_kv_blocks,
        dtype=dtype
    )

    # 批量模式
    if args.batch_mode:
        if args.prompts_file and Path(args.prompts_file).exists():
            with open(args.prompts_file) as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = [
                "Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house.",
            ]

        print(f"\n批量生成 {len(prompts)} 个 prompts...\n")
        results = generator.batch_generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n{'='*60}")
            print(f"Prompt {i+1}: {prompt}")
            print(f"生成: {result}")
        print(f"\n{'='*60}\n")

    # 单次生成
    elif args.prompt:
        print(f"\n输入提示: {args.prompt}\n")
        print("生成中...\n")

        generated_text = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

        print("生成结果:")
        print(generated_text)

    # 交互模式
    else:
        generator.interactive_mode()


if __name__ == "__main__":
    main()
