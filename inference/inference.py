import torch
from tokenizers import Tokenizer
from transformer import TransformerLM
import argparse
from pathlib import Path
from utils.generation import load_dense_model, sampling_probs


class TextGenerator:
    """文本生成器类"""

    def __init__(self, model_path, tokenizer_path, device='cuda', config_overrides=None):
        """
        初始化文本生成器
        
        参数：
            model_path: 模型checkpoint路径
            tokenizer_path: tokenizer文件路径
            device: 运行设备 ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        self.model, self.tokenizer, self.config = load_dense_model(
            model_path, tokenizer_path, self.device, config_overrides
        )
        self.vocab_size = self.config['vocab_size']

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=250, temperature=1.0,
                 top_k=None, top_p=0.9, repetition_penalty=1.0):
        """先计算提示词，再利用缓存逐个生成 token。"""
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens 必须非负')
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            raise ValueError('提示词至少需要一个 token')
        if max_new_tokens == 0:
            return self.tokenizer.decode(ids)
        max_len = self.config['max_seq_len']
        ids = ids[-max(1, max_len - min(max_new_tokens, max_len - 1)):]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        counts = {}
        eos = self.tokenizer.token_to_id('<|endoftext|>')
        self.model.clear_cache()
        try:
            logits = self.model(input_ids, use_cache=True)[:, -1, :]
            for _ in range(min(max_new_tokens, max_len - len(ids))):
                probs = sampling_probs(logits, temperature, top_k, top_p, counts,
                                       repetition_penalty, self.tokenizer)
                token = torch.multinomial(probs, 1)
                token_id = token.item()
                ids.append(token_id)
                counts[token_id] = counts.get(token_id, 0) + 1
                if token_id == eos or len(ids) >= max_len:
                    break
                logits = self.model(token, use_cache=True)[:, -1, :]
            return self.tokenizer.decode(ids)
        finally:
            self.model.clear_cache()

    def interactive_mode(self):
        """交互式生成模式"""
        print("进入交互模式 (输入 'quit' 退出)")
        print("提示: 每次生成都会自动清空KV Cache\n")

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
                    max_new_tokens=258,
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
    parser = argparse.ArgumentParser(description='Transformer模型推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer_tinystories.json',
                        help='tokenizer文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='输入提示文本（如果不提供则进入交互模式）')
    parser.add_argument('--max_new_tokens', type=int, default=250,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='温度参数')
    parser.add_argument('--top_k', type=int, default=None,
                        help='top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='nucleus采样')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='重复惩罚系数')

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    if not Path(args.tokenizer_path).exists():
        print(f"错误: Tokenizer文件不存在: {args.tokenizer_path}")
        return

    # 初始化生成器
    generator = TextGenerator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )

    # 如果提供了prompt，直接生成；否则进入交互模式
    if args.prompt:
        print(f"\n输入提示: {args.prompt}")
        print("\n生成中...\n")

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
    else:
        # 进入交互模式
        generator.interactive_mode()


if __name__ == "__main__":
    main()
