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
    4. 学习到的奖励模型
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
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)


class GRPOTrainerV2:
    """
    GRPO 训练器 V2 - 完整实现
    
    改进：
    1. 正确的 logit_probs 计算
    2. 多轮 PPO 更新
    3. 经验回放
    4. 更详细的日志
    """
    
    def __init__(
        self,
        model: HFTransformerLM,
        ref_model: HFTransformerLM,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer,
        reward_model: AdvancedRewardModel,
        config,
    ):
        self.model = model.to(config.device)
        self.ref_model = ref_model.to(config.device)
        self.ref_model.eval()
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
        
        # 经验缓冲
        self.buffer = ExperienceBuffer()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        from transformers import get_scheduler
        total_steps = (
            len(train_dataset) // config.batch_size
        ) * config.num_epochs
        
        self.scheduler = get_scheduler(
            'cosine',
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # 统计
        self.global_step = 0
        self.epoch = 0
        
        # DataLoader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
    
    @torch.no_grad()
    def sample_trajectories(
        self,
        prompts: List[str],
        num_samples_per_prompt: int,
    ) -> List[GRPOExperience]:
        """
        采样轨迹（生成回答并计算奖励）
        """
        self.model.eval()
        
        all_experiences = []
        
        for prompt in prompts:
            group_responses = []
            group_response_ids = []
            group_logit_probs = []
            
            # 为这个 prompt 生成多个样本（一组）
            for _ in range(num_samples_per_prompt):
                # Tokenize prompt
                prompt_tokens = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                ).to(self.config.device)
                
                prompt_ids = prompt_tokens['input_ids']
                
                # 生成回答
                response_ids, logit_prob = self._generate_with_logitprobs(
                    prompt_ids,
                    max_new_tokens=self.config.max_gen_length,
                )
                
                # 解码
                full_text = self.tokenizer.decode(
                    torch.cat([prompt_ids, response_ids], dim=1)[0],
                    skip_special_tokens=True
                )
                response = full_text[len(prompt):].strip()
                
                group_responses.append(response)
                group_response_ids.append(response_ids)
                group_logit_probs.append(logit_prob.item())
            
            # 计算这一组的奖励
            rewards = self.reward_model.compute_reward([prompt] * len(group_responses), group_responses)
            
            # 计算组内相对优势
            advantages = self._compute_group_advantages(rewards)
            
            # 创建经验
            for i in range(num_samples_per_prompt):
                exp = GRPOExperience(
                    prompt=prompt,
                    prompt_ids=prompt_tokens['input_ids'],
                    response=group_responses[i],
                    response_ids=group_response_ids[i],
                    reward=rewards[i].item(),
                    old_logit_prob=group_logit_probs[i],
                    advantage=advantages[i].item(),
                )
                all_experiences.append(exp)
        
        self.model.train()
        
        return all_experiences
    
    def _generate_with_logitprobs(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成并记录 logit 概率
        
        Returns:
            response_ids: [1, response_len] 只包含生成的部分
            logit_prob: [1] 总 logit 概率
        """
        self.model.model.clear_cache()
        
        generated = prompt_ids.clone()
        response_logit_probs = []
        
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.model(generated, use_cache=True)
            logitits = outputs.logitits[:, -1, :]
            
            # 应用温度
            logitits = logitits / self.config.temperature
            
            # Top-k & Top-p
            logitits = self._apply_sampling_filters(logitits)
            
            # 采样
            probs = F.softmax(logitits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 记录 logit_prob
            logit_prob = torch.logit(probs.gather(1, next_token) + 1e-10)
            response_logit_probs.append(logit_prob)
            
            # 拼接
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查结束
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        self.model.model.clear_cache()
        
        # 提取 response 部分
        response_ids = generated[:, prompt_ids.size(1):]
        
        # 总 logit_prob
        if response_logit_probs:
            total_logit_prob = torch.stack(response_logit_probs).sum()
        else:
            total_logit_prob = torch.tensor(0.0, device=self.config.device)
        
        return response_ids, total_logit_prob.unsqueeze(0)
    
    def _apply_sampling_filters(self, logitits: torch.Tensor) -> torch.Tensor:
        """应用 top-k 和 top-p 过滤"""
        
        # Top-k
        if self.config.top_k > 0:
            indices_to_remove = logitits < torch.topk(logitits, self.config.top_k)[0][..., -1, None]
            logitits[indices_to_remove] = float('-inf')
        
        # Top-p
        if self.config.top_p < 1.0:
            sorted_logitits, sorted_indices = torch.sort(logitits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logitits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logitits[indices_to_remove] = float('-inf')
        
        return logitits
    
    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算组内归一化优势"""
        
        if self.config.advantage_normalization:
            mean = rewards.mean()
            std = rewards.std() + 1e-8
            advantages = (rewards - mean) / std
        else:
            advantages = rewards - rewards.mean()
        
        return advantages
    
    def compute_logit_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算给定序列的 logit 概率
        
        Args:
            prompt_ids: [batch, prompt_len]
            response_ids: [batch, response_len]
        
        Returns:
            logit_probs: [batch] 每个样本的总 logit 概率
        """
        # 拼接完整序列
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # 前向传播
        outputs = self.model(full_ids)
        logitits = outputs.logitits
        
        # Shift for autoregressive
        shift_logitits = logitits[:, :-1, :].contiguous()
        shift_labels = full_ids[:, 1:].contiguous()
        
        # 计算 logit_probs
        logit_probs = F.logit_softmax(shift_logitits, dim=-1)
        token_logit_probs = torch.gather(
            logit_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只计算 response 部分
        prompt_len = prompt_ids.size(1)
        response_logit_probs = token_logit_probs[:, prompt_len-1:]
        
        # 求和得到总 logit_prob
        total_logit_probs = response_logit_probs.sum(dim=-1)
        
        return total_logit_probs
    
    @torch.no_grad()
    def compute_ref_logit_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """计算参考模型的 logit 概率"""
        
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        outputs = self.ref_model(full_ids)
        logitits = outputs.logitits
        
        shift_logitits = logitits[:, :-1, :].contiguous()
        shift_labels = full_ids[:, 1:].contiguous()
        
        logit_probs = F.logit_softmax(shift_logitits, dim=-1)
        token_logit_probs = torch.gather(
            logit_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        prompt_len = prompt_ids.size(1)
        response_logit_probs = token_logit_probs[:, prompt_len-1:]
        
        total_logit_probs = response_logit_probs.sum(dim=-1)
        
        return total_logit_probs
    
    def ppo_update(
        self,
        experiences: List[GRPOExperience],
        num_updates: int = 4,
    ) -> Dict:
        """
        执行多轮 PPO 更新
        
        Args:
            experiences: 经验列表
            num_updates: 更新次数
        """
        all_stats = {
            'policy_loss': [],
            'kl_div': [],
            'entropy': [],
            'ratio_mean': [],
        }
        
        for _ in range(num_updates):
            # 打乱经验
            import random
            random.shuffle(experiences)
            
            # 分批更新
            for batch_exp in self._batch_experiences(experiences, self.config.ppo_batch_size):
                # 准备批次数据
                prompt_ids = torch.cat([exp.prompt_ids for exp in batch_exp], dim=0).to(self.config.device)
                response_ids = torch.cat([exp.response_ids for exp in batch_exp], dim=0).to(self.config.device)
                
                old_logit_probs = torch.tensor(
                    [exp.old_logit_prob for exp in batch_exp],
                    device=self.config.device
                )
                
                advantages = torch.tensor(
                    [exp.advantage for exp in batch_exp],
                    device=self.config.device
                )
                
                # 计算当前策略的 logit_probs
                current_logit_probs = self.compute_logit_probs(prompt_ids, response_ids)
                
                # 计算参考模型的 logit_probs
                ref_logit_probs = self.compute_ref_logit_probs(prompt_ids, response_ids)
                
                # 计算损失
                stats = self._compute_ppo_loss(
                    current_logit_probs,
                    old_logit_probs,
                    advantages,
                    ref_logit_probs,
                )
                
                # 反向传播
                loss = stats['loss']
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 优化器步进
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 记录统计
                for key in all_stats:
                    if key in stats:
                        all_stats[key].append(stats[key])
        
        # 平均统计
        averaged_stats = {
            key: np.mean(values) for key, values in all_stats.items()
        }
        
        return averaged_stats
    
    def _batch_experiences(self, experiences: List[GRPOExperience], batch_size: int):
        """将经验分批"""
        for i in range(0, len(experiences), batch_size):
            yield experiences[i:i + batch_size]
    
    def _compute_ppo_loss(
        self,
        logit_probs: torch.Tensor,
        old_logit_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logit_probs: torch.Tensor,
    ) -> Dict:
        """计算 PPO 损失"""
        
        # 比率
        ratio = torch.exp(logit_probs - old_logit_probs)
        
        # PPO 裁剪
        clip_range = self.config.clip_range
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 散度惩罚
        kl_div = logit_probs - ref_logit_probs
        kl_penalty = self.config.kl_coef * kl_div.mean()
        
        # 总损失
        total_loss = policy_loss + kl_penalty
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.mean().item(),
            'ratio_mean': ratio.mean().item(),
        }
    
    def train(self):
        """训练主循环"""
        
        print("="*60)
        print("开始 GRPO 训练")
        print("="*60)
        print(f"轮数: {self.config.num_epochs}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"每个提示的样本数: {self.config.num_samples_per_prompt}")
        print(f"PPO 更新次数: {self.config.num_ppo_updates}")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(pbar):
                prompts = batch['prompt']
                
                # 采样轨迹 
                experiences = self.sample_trajectories(
                    prompts,
                    num_samples_per_prompt=self.config.num_samples_per_prompt,
                )
                
                # PPO 更新
                stats = self.ppo_update(
                    experiences,
                    num_updates=self.config.num_ppo_updates,
                )
                
                # 更新学习率
                self.scheduler.step()
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'reward': f"{np.mean([e.reward for e in experiences]):.3f}",
                    'kl': f"{stats.get('kl_div', 0):.4f}",
                    'loss': f"{stats.get('policy_loss', 0):.4f}",
                })
                
                # 日志
                if self.global_step % self.config.logitging_steps == 0:
                    self._logit_stats(experiences, stats)
                
                # 保存
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            print(f"\nEpoch {epoch+1} 完成")
        
        print("\n训练完成！")
    
    def _logit_stats(self, experiences: List[GRPOExperience], stats: Dict):
        """记录详细统计"""
        
        rewards = [e.reward for e in experiences]
        advantages = [e.advantage for e in experiences]
        
        print(f"\n[Step {self.global_step}]")
        print(f"  奖励: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"  优势: {np.mean(advantages):.4f} ± {np.std(advantages):.4f}")
        print(f"  策略损失: {stats.get('policy_loss', 0):.4f}")
        print(f"  KL散度: {stats.get('kl_div', 0):.4f}")
        print(f"  比率: {stats.get('ratio_mean', 0):.4f}")
    
    def save_checkpoint(self):
        """保存检查点"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        save_path = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n保存检查点到 {save_path}")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }, f"{save_path}/trainer_state.pt")
        
        self.model.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


# 配置类
@dataclass  
class GRPOConfigV2:
    """GRPO V2 配置"""
    output_dir: str = './grpo_v2_output'
    num_epochs: int = 3
    batch_size: int = 2
    ppo_batch_size: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    
    # GRPO 参数
    num_samples_per_prompt: int = 4
    num_ppo_updates: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # 奖励和优势
    kl_coef: float = 0.1
    clip_range: float = 0.2
    advantage_normalization: bool = True
    
    # 优化器
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # 生成
    max_gen_length: int = 200
    max_prompt_length: int = 256
    
    # 日志
    logitging_steps: int = 10
    save_steps: int = 500
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16: bool = False

class PromptDataset(Dataset):
    """
    GRPO 提示数据集
    
    只包含 prompts,模型会为每个 prompt 生成多个回答
    """
    
    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = []
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                for line in f:
                    item = json.loads(line)
                    self.prompts.append(item['prompt'])
            else:
                data = json.load(f)
                self.prompts = [item['prompt'] for item in data]
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}

def main():
    """主函数"""
    
    config = GRPOConfigV2(
        output_dir='./grpo_v2_output',
        num_epochs=2,
        batch_size=2,
        ppo_batch_size=4,
        num_samples_per_prompt=4,
        num_ppo_updates=3,
    )
    
    # 模型
    model_config = TransformerLMConfig(
        d_model=256,
        n_head=4,
        vocab_size=50257,
        max_seq_len=1024,
        d_ff=1024,
        theta=10000.0,
        n_layer=4,
    )
    
    model = HFTransformerLM(model_config)
    ref_model = HFTransformerLM(model_config)
    ref_model.load_state_dict(model.state_dict())
    
    # 数据
    tokenizer = create_tokenizer()
    
    train_dataset = PromptDataset('train_preferences.jsonl', tokenizer)
    eval_dataset = PromptDataset('eval_preferences.jsonl', tokenizer)
    
    # 奖励模型
    reward_model = AdvancedRewardModel(reward_type='combined')
    
    # 训练
    trainer = GRPOTrainerV2(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
    )
    
    trainer.train()
    trainer.save_checkpoint()


if __name__ == '__main__':
    main()