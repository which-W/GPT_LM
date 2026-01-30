import torch
from tokenizers import Tokenizer
from transformer import TransformerLM
import argparse
from pathlib import Path


class TextGenerator:
    """文本生成器类"""
    
    def __init__(self, model_path, tokenizer_path, device='cuda'):
        """
        初始化文本生成器
        
        Args:
            model_path: 模型checkpoint路径
            tokenizer_path: tokenizer文件路径
            device: 运行设备 ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        print(f"加载tokenizer: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # 加载模型checkpoint
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint中获取模型配置
        self.config = checkpoint.get('config', {})
        
        # 初始化模型
        self.model = TransformerLM(
            d_model=self.config.get('d_model', 512),
            n_head=self.config.get('n_head', 8),
            vocab_size=self.vocab_size,
            max_seq_len=self.config.get('max_seq_len', 512),
            d_ff=self.config.get('d_ff', 2048),
            theta=self.config.get('theta', 10000.0),
            n_layer=self.config.get('n_layer', 6),
            device=self.device,
            dtype=torch.float32
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("模型加载完成!")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("KV Cache 已启用")
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=250, temperature=1.0, 
                 top_k=None, top_p=0.9, repetition_penalty=1.0):
        """
        生成文本 - 修复版
        
        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数，越高越随机
            top_k: top-k采样
            top_p: nucleus采样
            repetition_penalty: 重复惩罚系数
            
        Returns:
            生成的完整文本
        """
        # 编码输入
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        
        # 检查prompt长度
        max_seq_len = self.config.get('max_seq_len', 512)
        if input_ids.shape[1] > max_seq_len - max_new_tokens:
            print(f"警告: Prompt太长 ({input_ids.shape[1]} tokens)，截断到 {max_seq_len - max_new_tokens}")
            input_ids = input_ids[:, -(max_seq_len - max_new_tokens):]
        
        # 用于重复惩罚的token计数
        token_counts = {}
        
        # 清空缓存
        self.model.clear_cache()
        
        # Prefill阶段 - 只处理一次prompt
        logits = self.model(input_ids, use_cache=True)
        next_token_logits = logits[:, -1, :]
        
        # Generation循环 - 每次只处理新token
        for i in range(max_new_tokens):
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for token_id, count in token_counts.items():
                    next_token_logits[:, token_id] /= (repetition_penalty ** count)
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # Top-k采样
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus)采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 更新token计数
            token_id = next_token.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            # 拼接到序列（用于最终解码）
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查序列长度
            if input_ids.shape[1] >= max_seq_len:
                print(f"\n达到最大序列长度 {max_seq_len}，停止生成")
                break
            
            # 只传入新token，不是整个序列 
            logits = self.model(next_token, use_cache=True)
            next_token_logits = logits[:, -1, :]
            
            # 检查是否生成了结束符（如果有的话）
            # 这里可以根据实际情况添加EOS token的检查
        
        # 解码生成的文本
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text
    
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
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json',
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