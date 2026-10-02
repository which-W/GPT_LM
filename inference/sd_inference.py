import torch
import torch.nn.functional as F
import argparse
import time
from pathlib import Path
from tokenizers import Tokenizer
from transformer import TransformerLM
from utils.generation import load_dense_model, sampling_probs

class SpeculativeGenerator:
    """投机采样生成器 - 支持交互模式"""
    def __init__(self, draft_model_path, target_model_path, tokenizer_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 1. 加载 Tokenizer
        self.tokenizer_path = tokenizer_path
        target_checkpoint = torch.load(target_model_path, map_location='cpu', weights_only=True)
        from checkpoint_use import get_checkpoint_config, load_tokenizer
        target_config = get_checkpoint_config(target_checkpoint)
        self.tokenizer = load_tokenizer(target_checkpoint, tokenizer_path, target_config['vocab_size'])
        del target_checkpoint
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 2. 加载模型逻辑
        print(f"正在准备模型 (设备: {self.device})...")
        self.draft_model, self.draft_config = self._load_model(draft_model_path)
        self.target_model, self.target_config = self._load_model(target_model_path)
        if self.draft_config['vocab_size'] != self.target_config['vocab_size']:
            raise ValueError('草稿模型和目标模型的输出词表维度必须相同')
        if not all(hasattr(model, 'truncate_cache') for model in (self.draft_model, self.target_model)):
            raise ValueError('投机采样需要支持缓存回退的模型')

        # 使用目标模型的配置作为主配置
        self.max_seq_len = self.target_config.get('max_seq_len', 512)

        print("系统就绪。大模型负责质量，小模型负责速度。")

    def _load_model(self, path):
        model, tokenizer, config = load_dense_model(path, self.tokenizer_path, self.device)
        if tokenizer.to_str() != self.tokenizer.to_str():
            raise ValueError('草稿模型和目标模型必须使用相同分词器')
        return model, config

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, gamma=5, temperature=0.8,
                 top_k=None, top_p=0.9, repetition_penalty=1.0):
        """使用概率接受率和拒绝后的残差分布执行投机采样。"""
        if gamma < 1 or max_new_tokens < 0:
            raise ValueError('gamma 必须为正数，max_new_tokens 必须非负')
        ids = self.tokenizer.encode(prompt).ids
        max_len = min(self.draft_config['max_seq_len'], self.target_config['max_seq_len'])
        if not ids or len(ids) > max_len:
            raise ValueError('提示词为空或超过模型上下文长度')
        if max_new_tokens == 0 or len(ids) == max_len:
            return self.tokenizer.decode(ids), 0, 0
        prefix = torch.tensor([ids], device=self.device)
        counts = {}
        accepted_count = total_gen = 0
        eos = self.tokenizer.token_to_id('<|endoftext|>')
        self.draft_model.clear_cache()
        self.target_model.clear_cache()
        try:
            q_logits = self.draft_model(prefix, use_cache=True)[:, -1, :]
            p_logits = self.target_model(prefix, use_cache=True)[:, -1, :]
            while total_gen < max_new_tokens and len(ids) < max_len:
                start = len(ids)
                steps = min(gamma, max_new_tokens - total_gen, max_len - start)
                drafts, q_probs = [], []
                draft_counts = dict(counts)
                for _ in range(steps):
                    q = sampling_probs(q_logits, temperature, top_k, top_p, draft_counts,
                                       repetition_penalty, self.tokenizer)
                    token = torch.multinomial(q, 1)
                    drafts.append(token)
                    q_probs.append(q)
                    draft_counts[token.item()] = draft_counts.get(token.item(), 0) + 1
                    q_logits = self.draft_model(token, use_cache=True)[:, -1, :]
                verified = self.target_model(torch.cat(drafts, dim=1), use_cache=True)
                replacement = None
                matched = 0
                for index, token in enumerate(drafts):
                    logits = p_logits if index == 0 else verified[:, index - 1, :]
                    p = sampling_probs(logits, temperature, top_k, top_p, counts,
                                       repetition_penalty, self.tokenizer)
                    token_id = token.item()
                    accept = min(1.0, (p[0, token_id] / q_probs[index][0, token_id]).item())
                    if torch.rand((), device=p.device).item() > accept:
                        residual = (p - q_probs[index]).clamp(min=0)
                        residual /= residual.sum(dim=-1, keepdim=True)
                        replacement = torch.multinomial(residual, 1)
                        break
                    ids.append(token_id)
                    counts[token_id] = counts.get(token_id, 0) + 1
                    matched += 1
                    accepted_count += 1
                    total_gen += 1
                    if token_id == eos:
                        return self.tokenizer.decode(ids), accepted_count, total_gen
                # 回退到真正接受的前缀，替换 token 在下一步写入两个缓存。
                if replacement is not None:
                    self.draft_model.truncate_cache(start + matched)
                    self.target_model.truncate_cache(start + matched)
                elif total_gen < max_new_tokens and len(ids) < max_len:
                    p = sampling_probs(verified[:, -1, :], temperature, top_k, top_p,
                                       counts, repetition_penalty, self.tokenizer)
                    replacement = torch.multinomial(p, 1)
                else:
                    break
                token_id = replacement.item()
                ids.append(token_id)
                counts[token_id] = counts.get(token_id, 0) + 1
                total_gen += 1
                if token_id == eos or len(ids) >= max_len or total_gen >= max_new_tokens:
                    break
                q_logits = self.draft_model(replacement, use_cache=True)[:, -1, :]
                p_logits = self.target_model(replacement, use_cache=True)[:, -1, :]
            return self.tokenizer.decode(ids), accepted_count, total_gen
        finally:
            self.draft_model.clear_cache()
            self.target_model.clear_cache()

    def _sample(self, logits, temperature=0.8, top_k=None, top_p=0.9,
                token_counts=None, repetition_penalty=1.0):
        """按统一的温度、词表和重复惩罚规则采样。"""
        return torch.multinomial(sampling_probs(logits, temperature, top_k, top_p,
                                token_counts, repetition_penalty, self.tokenizer), 1)

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
    parser.add_argument('--tokenizer', type=str, default='tokenizer_tinystories.json')
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
