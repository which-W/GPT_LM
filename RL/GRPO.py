"""
GRPO (Group Relative Policy Optimization) 完整实现

这是一个更完整的实现，包括：
1. 正确的 logit_probs 重新计算
2. 多轮 PPO 更新
3. 更详细的统计信息
4. 支持自定义奖励函数
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import copy

from RL.DPO import (
    HFTransformerLM,
    TransformerLMConfig,
    create_tokenizer
)


@dataclass
class GRPOExperience:
    """存储一次采样的经验"""
    prompt: str
    prompt_ids: torch.Tensor
    response: str
    response_ids: torch.Tensor
    reward: float
    old_logit_prob: float
    advantage: float
    ref_logit_prob: Optional[float] = None
    old_token_log_probs: Optional[torch.Tensor] = None
    ref_token_log_probs: Optional[torch.Tensor] = None


class ExperienceBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.experiences: List[GRPOExperience] = []

    def add(self, experience: GRPOExperience):
        self.experiences.append(experience)

    def add_batch(self, experiences: List[GRPOExperience]):
        self.experiences.extend(experiences)

    def clear(self):
        self.experiences = []

    def get_batches(self, batch_size: int):
        """生成批次"""
        for i in range(0, len(self.experiences), batch_size):
            yield self.experiences[i:i + batch_size]

    def __len__(self):
        return len(self.experiences)


class AdvancedRewardModel:
    """
    高级奖励模型
    支持多种奖励组合：
    1. 长度奖励
    2. 格式奖励
    3. 多样性奖励
    此处的组合奖励仅用于教学，不代表数学答案的正确率。
    """

    def __init__(
        self,
        reward_type: str = 'combined',
        length_weight: float = 0.3,
        format_weight: float = 0.3,
        diversity_weight: float = 0.2,
        learned_weight: float = 0.2,
    ):
        self.reward_type = reward_type
        self.length_weight = length_weight
        self.format_weight = format_weight
        self.diversity_weight = diversity_weight
        self.learned_weight = learned_weight

        # 用于计算多样性的历史回答
        self.response_history = []

    def compute_reward(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs
    ) -> torch.Tensor:
        """计算组合奖励"""

        if self.reward_type == 'length_penalty':
            return self._length_reward(responses)

        elif self.reward_type == 'combined':
            # 组合多个奖励
            rewards = torch.zeros(len(responses))

            if self.length_weight > 0:
                rewards += self.length_weight * self._length_reward(responses)

            if self.format_weight > 0:
                rewards += self.format_weight * self._format_reward(responses)

            if self.diversity_weight > 0:
                rewards += self.diversity_weight * self._diversity_reward(responses)

            return rewards

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _length_reward(self, responses: List[str]) -> torch.Tensor:
        """长度奖励：鼓励适中的长度"""
        rewards = []

        for response in responses:
            length = len(response.split())

            # 目标长度：50-150词
            target_min, target_max = 50, 150

            if target_min <= length <= target_max:
                reward = 1.0
            elif length < target_min:
                # 太短：线性惩罚
                reward = length / target_min
            else:
                # 太长：线性惩罚
                reward = max(0.1, 1.0 - (length - target_max) / target_max)

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _format_reward(self, responses: List[str]) -> torch.Tensor:
        """格式奖励：鼓励良好的格式"""
        rewards = []

        for response in responses:
            reward = 0.5  # 基础分

            # 检查是否有段落结构
            if '\n\n' in response or '\n' in response:
                reward += 0.2

            # 检查是否有标点符号
            if any(p in response for p in ['.', '!', '?']):
                reward += 0.2

            # 检查是否以完整句子结尾
            if response.strip().endswith(('.', '!', '?')):
                reward += 0.1

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _diversity_reward(self, responses: List[str]) -> torch.Tensor:
        """多样性奖励：鼓励不同的回答"""
        rewards = []

        for response in responses:
            # 计算与历史回答的相似度
            if not self.response_history:
                reward = 1.0  # 第一个回答总是新颖的
            else:
                # 简单的词重叠度量
                response_words = set(response.lower().split())

                max_similarity = 0.0
                for hist_response in self.response_history[-10:]:  # 只看最近10个
                    hist_words = set(hist_response.lower().split())

                    if len(response_words) == 0 or len(hist_words) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(response_words & hist_words)
                        union = len(response_words | hist_words)
                        similarity = intersection / union if union > 0 else 0.0

                    max_similarity = max(max_similarity, similarity)

                # 奖励新颖性
                reward = 1.0 - max_similarity

            # 添加到历史
            self.response_history.append(response)
            self.response_history = self.response_history[-100:]

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


class GRPOTrainerV2:
    """按回答中的每个令牌计算策略比率，支持不同长度的提示词及回答。"""
    def __init__(self, model, ref_model, train_dataset, eval_dataset, tokenizer, reward_model, config):
        if min(config.batch_size, config.ppo_batch_size, config.num_samples_per_prompt, config.num_ppo_updates) < 1:
            raise ValueError("训练批次、候选数量和更新次数必须为正数")
        if config.num_epochs < 1 or config.max_steps < 0:
            raise ValueError("训练轮数必须为正，总步数不能为负")
        if min(config.logging_steps, config.save_steps, config.max_gen_length, config.max_prompt_length) < 1:
            raise ValueError("日志间隔、保存间隔和生成长度必须为正")
        self.model = model.to(config.device)
        self.ref_model = ref_model.to(config.device).eval().requires_grad_(False)
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.tokenizer, self.reward_model, self.config = tokenizer, reward_model, config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        if not len(self.train_dataloader):
            raise ValueError("训练数据不能为空")
        from transformers import get_scheduler
        self.scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=config.warmup_steps,
                                       num_training_steps=config.max_steps or len(self.train_dataloader) * config.num_epochs)
        self.global_step = self.epoch = 0
        self.buffer = ExperienceBuffer()

    def _token_log_probs(self, model, prompt_ids, response_ids):
        full = torch.cat([prompt_ids, response_ids], dim=1)
        logits = model(full).logits.float()
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        selected = log_probs.gather(-1, full[:, 1:].unsqueeze(-1)).squeeze(-1)
        return selected[:, prompt_ids.size(1) - 1:]

    def compute_logit_probs(self, prompt_ids, response_ids):
        """保留旧接口名称，返回回答的条件对数概率总和。"""
        return self._token_log_probs(self.model, prompt_ids, response_ids).sum(-1)

    @torch.no_grad()
    def compute_ref_logit_probs(self, prompt_ids, response_ids):
        return self._token_log_probs(self.ref_model, prompt_ids, response_ids).sum(-1)

    @torch.no_grad()
    def _generate_with_logitprobs(self, prompt_ids, max_new_tokens):
        generated = self.model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=True,
                                        temperature=self.config.temperature, top_k=self.config.top_k,
                                        top_p=self.config.top_p, eos_token_id=self.tokenizer.eos_token_id)
        response = generated[:, prompt_ids.size(1):]
        if response.size(1) == 0:
            raise ValueError("提示词占满上下文，无法生成回答")
        return response, self.compute_logit_probs(prompt_ids, response)

    def _compute_group_advantages(self, rewards):
        centered = rewards - rewards.mean()
        return centered / rewards.std(unbiased=False).clamp_min(1e-8) if self.config.advantage_normalization else centered

    @torch.no_grad()
    def sample_trajectories(self, prompts, num_samples_per_prompt):
        self.model.eval()
        experiences = []
        try:
            for prompt in prompts:
                room = self.model.config.max_seq_len - min(self.config.max_gen_length, self.model.config.max_seq_len - 1)
                prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                            max_length=min(self.config.max_prompt_length, room))["input_ids"].to(self.config.device)
                responses, texts, old_probs = [], [], []
                for _ in range(num_samples_per_prompt):
                    ids, _ = self._generate_with_logitprobs(prompt_ids, self.config.max_gen_length)
                    responses.append(ids)
                    texts.append(self.tokenizer.decode(ids[0], skip_special_tokens=True))
                    old_probs.append(self._token_log_probs(self.model, prompt_ids, ids).detach())
                rewards = self.reward_model.compute_reward([prompt] * len(texts), texts)
                advantages = self._compute_group_advantages(rewards)
                for i, text in enumerate(texts):
                    exp = GRPOExperience(prompt, prompt_ids.detach(), text, responses[i].detach(),
                                         rewards[i].item(), old_probs[i].sum().item(), advantages[i].item())
                    exp.old_token_log_probs = old_probs[i]
                    exp.ref_token_log_probs = self._token_log_probs(self.ref_model, prompt_ids, responses[i]).detach()
                    experiences.append(exp)
            return experiences
        finally:
            self.model.train()

    def _batch_experiences(self, experiences, batch_size):
        for i in range(0, len(experiences), batch_size):
            yield experiences[i:i + batch_size]

    def _compute_ppo_loss(self, logit_probs, old_logit_probs, advantages, ref_logit_probs):
        ratio = torch.exp(logit_probs - old_logit_probs)
        clipped = ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range)
        policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        # 非负的采样 KL 估计；使用逐令牌概率，不将整条序列概率当成一个动作。
        delta = ref_logit_probs - logit_probs
        kl_div = (delta.clamp(max=20).exp() - delta - 1).mean()
        loss = policy_loss + self.config.kl_coef * kl_div
        return {"loss": loss, "policy_loss": policy_loss.detach().item(),
                "kl_div": kl_div.detach().item(), "ratio_mean": ratio.detach().mean().item()}

    def ppo_update(self, experiences, num_updates=4):
        import random
        if not experiences:
            raise ValueError("没有可训练的采样轨迹")
        stats = {key: [] for key in ("policy_loss", "kl_div", "ratio_mean")}
        for _ in range(num_updates):
            random.shuffle(experiences)
            for batch in self._batch_experiences(experiences, self.config.ppo_batch_size):
                self.optimizer.zero_grad()
                # 每条轨迹单独计算后求批次均值，避免拼接变长张量引入填充误差。
                for exp in batch:
                    current = self._token_log_probs(self.model, exp.prompt_ids, exp.response_ids)
                    item = self._compute_ppo_loss(current, exp.old_token_log_probs,
                                                  torch.tensor(exp.advantage, device=current.device), exp.ref_token_log_probs)
                    (item["loss"] / len(batch)).backward()
                    for key in stats:
                        stats[key].append(item[key])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        return {key: float(np.mean(values)) for key, values in stats.items()}

    def train(self):
        budget = self.config.max_steps or len(self.train_dataloader) * self.config.num_epochs
        epochs = (budget + len(self.train_dataloader) - 1) // len(self.train_dataloader)
        for epoch in range(epochs):
            self.epoch = epoch
            for batch in tqdm(self.train_dataloader, desc=f"轮次 {epoch + 1}"):
                experiences = self.sample_trajectories(batch["prompt"], self.config.num_samples_per_prompt)
                stats = self.ppo_update(experiences, self.config.num_ppo_updates)
                self.scheduler.step()
                self.global_step += 1
                if self.global_step % self.config.logging_steps == 0:
                    print(f"训练步 {self.global_step}，平均奖励 {np.mean([e.reward for e in experiences]):.4f}，损失 {stats['policy_loss']:.4f}")
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                if self.global_step >= budget:
                    return

    def save_checkpoint(self):
        from pathlib import Path
        output = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
        torch.save(dict(optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(),
                        global_step=self.global_step, epoch=self.epoch), output / "trainer_state.pt")


@dataclass
class GRPOConfigV2:
    """采样及逐令牌策略更新的配置。"""
    output_dir: str = "grpo_output"
    num_epochs: int = 2
    max_steps: int = 0
    batch_size: int = 2
    ppo_batch_size: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    num_samples_per_prompt: int = 4
    num_ppo_updates: int = 3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    kl_coef: float = 0.1
    clip_range: float = 0.2
    advantage_normalization: bool = True
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_gen_length: int = 200
    max_prompt_length: int = 256
    logging_steps: int = 10
    save_steps: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PromptDataset(Dataset):
    """读取仅需要 prompt 文本字段的 JSON 或 JSONL 数据。"""
    def __init__(self, data_path, tokenizer=None):
        with open(data_path, encoding="utf-8-sig") as f:
            data = [json.loads(line) for line in f if line.strip()] if str(data_path).lower().endswith(".jsonl") else json.load(f)
        self.prompts = [item["prompt"] for item in data]
        if not self.prompts or any(not isinstance(prompt, str) or not prompt for prompt in self.prompts):
            raise ValueError("提示词数据不能为空，且每项必须包含非空 prompt 文本")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return {"prompt": self.prompts[index]}


def main():
    import argparse
    from RL.DPO import load_policy
    parser = argparse.ArgumentParser(description="使用项目模型执行 GRPO 教学实验")
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--val_data_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--tokenizer_path", default="tokenizer_tinystories.json")
    parser.add_argument("--output_dir", default="grpo_output")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ppo_batch_size", type=int, default=4)
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--num_ppo_updates", type=int, default=3)
    parser.add_argument("--max_gen_length", type=int, default=200)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = GRPOConfigV2(**{key: value for key, value in vars(args).items()
                             if key in GRPOConfigV2.__dataclass_fields__})
    model, tokenizer = load_policy(args.checkpoint_path, args.tokenizer_path, max_seq_len=args.max_seq_len)
    ref_model = copy.deepcopy(model)
    dataset = PromptDataset(args.train_data_path)
    validation = PromptDataset(args.val_data_path) if args.val_data_path else None
    trainer = GRPOTrainerV2(model, ref_model, dataset, validation, tokenizer,
                            AdvancedRewardModel(reward_type="combined"), config)
    trainer.train()
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
