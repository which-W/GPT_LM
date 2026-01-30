import torch
import torch.nn.functional as F
import argparse
import time
from pathlib import Path
from tokenizers import Tokenizer
from transformer import TransformerLM

class SpeculativeGenerator:
    """投机采样生成器 - 支持交互模式"""
    def __init__(self, draft_model_path, target_model_path, tokenizer_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 1. 加载 Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # 2. 加载模型逻辑
        print(f"正在准备模型 (设备: {self.device})...")
        self.draft_model, self.draft_config = self._load_model(draft_model_path)
        self.target_model, self.target_config = self._load_model(target_model_path)
        
        # 使用目标模型的配置作为主配置
        self.max_seq_len = self.target_config.get('max_seq_len', 512)
        
        print("系统就绪。大模型负责质量，小模型负责速度。")

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get('config', {})
        model = TransformerLM(
            d_model=config.get('d_model', 512),
            n_head=config.get('n_head', 8),
            vocab_size=self.vocab_size,
            max_seq_len=config.get('max_seq_len', 512),
            d_ff=config.get('d_ff', 2048),
            theta=config.get('theta', 10000.0),
            n_layer=config.get('n_layer', 6),
            device=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, config

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, gamma=5, temperature=0.8, 
                 top_k=None, top_p=0.9, repetition_penalty=1.0):
        """核心投机采样逻辑 - 修复版"""
        encoding = self.tokenizer.encode(prompt)
        prefix = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        
        # 用于重复惩罚的token计数
        token_counts = {}
        
        # 初始清理
        self.draft_model.clear_cache()
        self.target_model.clear_cache()
        
        # Prefill 阶段
        _ = self.draft_model(prefix, use_cache=True)
        target_logits = self.target_model(prefix, use_cache=True)
        
        # 修复：正确传递所有参数
        next_token = self._sample(target_logits[:, -1, :], temperature, top_k, top_p, 
                                  token_counts, repetition_penalty)
        prefix = torch.cat([prefix, next_token], dim=1)
        
        # 更新token计数
        token_id = next_token.item()
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        total_gen = 1  # 已经生成了第一个token
        accepted_count = 0
        
        while total_gen < max_new_tokens:
            # Draft - 小模型猜 gamma 步
            draft_seq = []
            curr_input = next_token
            for _ in range(gamma):
                d_logits = self.draft_model(curr_input, use_cache=True)
                # 修复：传递所有参数
                d_token = self._sample(d_logits[:, -1, :], temperature, top_k, top_p, 
                                      token_counts, repetition_penalty)
                draft_seq.append(d_token)
                curr_input = d_token
            
            draft_tokens = torch.cat(draft_seq, dim=1)
            
            # Verify - 大模型一次性并行验证
            verify_input = torch.cat([next_token, draft_tokens], dim=1)
            t_logits_seq = self.target_model(verify_input, use_cache=True)
            
            # Check - 比较采样结果
            n_matches = 0
            for i in range(gamma):
                # 修复：传递所有参数
                t_token = self._sample(t_logits_seq[:, i, :], temperature, top_k, top_p, 
                                      token_counts, repetition_penalty)
                d_token = draft_tokens[:, i:i+1]
                
                if t_token.item() == d_token.item():
                    n_matches += 1
                    prefix = torch.cat([prefix, t_token], dim=1)
                    total_gen += 1
                    # 更新token计数
                    token_id = t_token.item()
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
                else:
                    # 猜错了，接受大模型的正确 token 并回滚
                    next_token = t_token
                    prefix = torch.cat([prefix, next_token], dim=1)
                    total_gen += 1
                    # 更新token计数
                    token_id = next_token.item()
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
                    # 执行回滚
                    self.draft_model.truncate_cache(prefix.shape[1])
                    self.target_model.truncate_cache(prefix.shape[1])
                    break
            else:
                # 如果 gamma 步全部猜对，额外拿一个大模型的奖励 token
                next_token = self._sample(t_logits_seq[:, -1, :], temperature, top_k, top_p, 
                                         token_counts, repetition_penalty)
                prefix = torch.cat([prefix, next_token], dim=1)
                total_gen += 1
                # 更新token计数
                token_id = next_token.item()
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            accepted_count += n_matches
            
            # 修复：使用配置中的 max_seq_len 而不是硬编码
            if prefix.shape[1] >= self.max_seq_len:
                print(f"\n达到最大序列长度 {self.max_seq_len}，停止生成")
                break

        return self.tokenizer.decode(prefix[0].tolist()), accepted_count, total_gen

    def _sample(self, logits, temperature=0.8, top_k=None, top_p=0.9, 
                token_counts=None, repetition_penalty=1.0):
        """
        采样函数 - 修复版
        
        Args:
            logits: 模型输出的logits
            temperature: 温度参数
            top_k: top-k采样参数
            top_p: nucleus采样参数
            token_counts: token计数字典（用于重复惩罚）
            repetition_penalty: 重复惩罚系数
        """
        # 应用重复惩罚
        if repetition_penalty != 1.0 and token_counts is not None:
            for token_id, count in token_counts.items():
                logits[:, token_id] /= (repetition_penalty ** count)
        
        # 应用温度
        logits = logits / temperature
        
        # Top-k采样
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus)采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def chat_mode(self, gamma=5, temperature=0.8, top_k=None, top_p=0.9, repetition_penalty=1.0):
        """交互式聊天模式 - 修复版"""
        print("\n=== 欢迎进入投机采样对话模式 ===")
        print(f"模式参数: gamma(每次预测步数)={gamma}, temperature={temperature}, "
              f"top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}")
        print("输入 'exit' 退出程序\n")
        
        while True:
            prompt = input("User > ").strip()
            if prompt.lower() in ['exit', 'quit']: break
            if not prompt: continue
            
            start_time = time.time()
            response, accepted, total = self.generate(
                prompt, 
                gamma=gamma,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            end_time = time.time()
            
            duration = end_time - start_time
            acc_rate = (accepted / total) * 100 if total > 0 else 0
            
            print(f"\nAssistant > {response}")
            print(f"\n[统计: 耗时 {duration:.2f}s | 接受率 {acc_rate:.1f}% | 生成 {total} tokens]\n")

def main():
    parser = argparse.ArgumentParser(description='投机采样推理')
    parser.add_argument('--draft', type=str, required=True, help='小模型路径')
    parser.add_argument('--target', type=str, required=True, help='大模型路径')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json')
    parser.add_argument('--gamma', type=int, default=5, help='投机步数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top_k', type=int, default=None, help='top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='nucleus采样')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='重复惩罚系数')
    args = parser.parse_args()

    generator = SpeculativeGenerator(args.draft, args.target, args.tokenizer)
    generator.chat_mode(
        gamma=args.gamma,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

if __name__ == "__main__":
    main()