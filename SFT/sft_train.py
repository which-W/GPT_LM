import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch
os.environ["VLLM_USE_V1"] = "0"  # 强制使用 V0 引擎，保留 model_executor 路径
# 【显存优化】减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils.sft_util import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations,
    compute_entropy
)
from SFT.drgrpo_grader import r1_zero_reward_fn


# ==========================================
# 辅助函数
# ==========================================

def build_r1_response(answer_str: str) -> str:
    """
    将 GSM8K 原始 answer 字段转成 r1 格式的训练 response。
    原始格式：  "Natalia sold 48/2 = 24 clips in May.\n...#### 72"
    目标格式：  "<think>\n推理过程\n</think> <answer>72</answer>"
    这样模型才能学到正确的输出格式，评估时 format_reward 才不会全为 0。
    """
    if "####" in answer_str:
        reasoning, gold = answer_str.split("####", 1)
        return f"<think>\n{reasoning.strip()}\n</think> <answer>{gold.strip()}</answer>"
    else:
        return f"<think>\n\n</think> <answer>{answer_str.strip()}</answer>"


def init_vllm(model_id, device, seed, gpu_memory_utilization):
    """初始化 vLLM 实例"""
    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            max_model_len=2048,  # 【显存优化】GSM8K 用 2048 完全够
        )


def load_policy_into_vllm_instance(policy, llm):
    """同步权重到 vLLM（兼容 vllm 0.8.5 V1 引擎）"""
    state_dict = policy.state_dict()
    
    # vllm 0.8.5 V1 引擎的访问路径
    llm_model = (
        llm.llm_engine
           .model_executor
           .driver_worker  
           .worker
           .model_runner
           .model
    )
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] Policy weights synced to vLLM.")

def get_batch(tokenized_data, batch_size, device):
    """从预处理好的数据中随机采样一个 Batch（Infinite Dataloader）"""
    total_len = len(tokenized_data["input_ids"])
    batch_indices = random.sample(range(total_len), batch_size)
    return {
        "input_ids":     tokenized_data["input_ids"][batch_indices].to(device),
        "labels":        tokenized_data["labels"][batch_indices].to(device),
        "response_mask": tokenized_data["response_mask"][batch_indices].to(device),
    }


# ==========================================
# 核心训练逻辑
# ==========================================

def run_sft_experiment(args):

    # 【修复】gradient_accumulation_steps 默认 None/True 时自动推导
    if not args.gradient_accumulation_steps:
        args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    grad_accum_steps = args.gradient_accumulation_steps
    print(f"[Config] grad_accum_steps = {grad_accum_steps}")

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # 加载 Prompt 模版（验证集 & 训练集共用）
    with open(args.prompt_path, "r") as f:
        r1_template = f.read().strip()

    # ── 模型与分词器 ──────────────────────────────────────────────
    print(f"Initializing Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    policy.gradient_checkpointing_enable()  # 【显存优化】用计算换显存

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    # 【修复3】Warmup + Cosine LR schedule，抑制早期 loss 剧烈抖动
    warmup_steps   = max(1, int(args.max_steps * 0.05))
    scheduler_warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_cos  = CosineAnnealingLR(optimizer, T_max=args.max_steps - warmup_steps, eta_min=args.lr * 0.1)
    scheduler      = SequentialLR(optimizer, schedulers=[scheduler_warm, scheduler_cos], milestones=[warmup_steps])

    # 【修复2】Entropy 正则系数，防止 response entropy 快速坍缩
    entropy_coeff = args.entropy_coeff

    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)

    # ── 训练数据加载与预处理 ──────────────────────────────────────
    print(f"Loading training data from {args.train_data_path}...")
    raw_train_data = []
    with open(args.train_data_path, "r") as f:
        for line in f:
            raw_train_data.append(json.loads(line))

    if args.filter_correct:
        raw_train_data = [item for item in raw_train_data if item.get('is_correct', True)]
        print(f"Filtered data size: {len(raw_train_data)}")

    if args.dataset_size and args.dataset_size < len(raw_train_data):
        raw_train_data = random.sample(raw_train_data, args.dataset_size)
        print(f"Sampled subset size: {args.dataset_size}")

    # 【修复】从 question/answer 字段动态构造 prompt 和 r1 格式 response
    # 原代码读 item['prompt'] / item['response']，但 GSM8K 只有 question / answer
    train_prompts = [
        r1_template.replace("{question}", item['question'])
        for item in raw_train_data
    ]
    train_responses = [
        build_r1_response(item['answer'])
        for item in raw_train_data
    ]

    print("Pre-tokenizing entire training dataset...")
    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=train_prompts,
        output_strs=train_responses,
        tokenizer=tokenizer,
        max_length=args.max_train_len,  # 【显存优化】截断超长样本
    )
    print(f"Tokenization complete. Total samples: {len(tokenized_train_data['input_ids'])}")

    # ── 验证集 ────────────────────────────────────────────────────
    print(f"Loading validation data from {args.val_data_path}...")
    val_prompts, val_ground_truths = [], []
    with open(args.val_data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_eval_samples:
                break
            item = json.loads(line)
            raw_a = item['answer']
            gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            val_prompts.append(r1_template.replace("{question}", item['question']))
            val_ground_truths.append(gold)

    eval_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # ── 训练主循环 ────────────────────────────────────────────────
    progress_bar = tqdm(range(args.max_steps), desc="SFT Steps")

    print("\n[Step 0] Starting Evaluation...")
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_inst)
    metrics = log_generations(
        vllm_model=vllm_inst, sampling_params=eval_sampling_params,
        prompts=val_prompts, ground_truths=val_ground_truths,
        reward_fn=r1_zero_reward_fn, step=0, log_prefix="eval"
    )
    print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
    policy.train()

    for step in range(args.max_steps):

        accumulated_loss        = 0.0
        accumulated_entropy     = 0.0
        accumulated_res_entropy = 0.0

        # ── 梯度累积循环 ──
        for _ in range(grad_accum_steps):
            batch  = get_batch(tokenized_train_data, args.micro_batch_size, args.device)
            logits = policy(batch["input_ids"]).logits  # (B, L, V)

            # log-probs（显存高效写法，避免保留整个 log_softmax 矩阵）
            lse           = torch.logsumexp(logits, dim=-1)
            target_logits = torch.gather(logits, -1, batch["labels"].unsqueeze(-1)).squeeze(-1)
            log_probs     = target_logits - lse

            # 熵监控（no_grad，不占梯度图）
            with torch.no_grad():
                token_entropy      = compute_entropy(logits)
                valid_token_mask   = (batch["labels"] != tokenizer.pad_token_id)
                current_res_mask   = batch["response_mask"].bool() & valid_token_mask
                avg_res_entropy    = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                avg_global_entropy = token_entropy[valid_token_mask].mean().item()

            # 【修复1】用实际 response token 数归一化，消除序列长度差异导致的 loss 量级剧烈抖动
            num_response_tokens = batch["response_mask"].sum().float().clamp(min=1.0)

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=num_response_tokens.item()
            )

            # 【修复2】Entropy 正则：在 response 位置鼓励保留多样性，防止 entropy 坍缩
            # token_entropy 已在 no_grad 块中计算，需要重新计算可微版本
            if entropy_coeff > 0 and current_res_mask.any():
                res_entropy = compute_entropy(logits.detach())[current_res_mask].mean()
                entropy_loss = -entropy_coeff * res_entropy / grad_accum_steps
                entropy_loss.backward()

            accumulated_loss        += loss.item() * grad_accum_steps
            accumulated_entropy     += avg_global_entropy
            accumulated_res_entropy += avg_res_entropy

            # 【显存优化】backward 后立即释放中间张量，防止多个 micro-batch 同时堆积
            del logits, lse, target_logits, log_probs, token_entropy
            torch.cuda.empty_cache()

        # ── 优化器更新 ──
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # 【修复3】更新 LR schedule
        progress_bar.update(1)

        current_lr = scheduler.get_last_lr()[0]
        wandb.log({
            "train/loss":             accumulated_loss / grad_accum_steps,
            "train/global_entropy":   accumulated_entropy / grad_accum_steps,
            "train/response_entropy": accumulated_res_entropy / grad_accum_steps,
            "train/lr":               current_lr,
            "train_step":             step + 1,
        })

        # ── 定期评估 ──
        if (step + 1) % args.eval_every_steps == 0:
            print(f"\n[Step {step + 1}] Starting Evaluation...")
            policy.eval()
            load_policy_into_vllm_instance(policy, vllm_inst)
            metrics = log_generations(
                vllm_model=vllm_inst, sampling_params=eval_sampling_params,
                prompts=val_prompts, ground_truths=val_ground_truths,
                reward_fn=r1_zero_reward_fn, step=step + 1, log_prefix="eval"
            )
            print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
            policy.train()

    # ── 保存 ──
    print("Training finished. Saving model...")
    save_name  = f"sft_steps{args.max_steps}_subset{args.dataset_size}_filtered{args.filter_correct}"
    output_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Step-based Training")

    # 路径
    parser.add_argument("--model_id",        type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k-train.jsonl")
    parser.add_argument("--val_data_path",   type=str, default="data/gsm8k-val.jsonl")
    parser.add_argument("--prompt_path",     type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output_dir",      type=str, default="result/checkpoints")

    # 训练参数
    parser.add_argument("--lr",               type=float, default=5e-6,
                        help="峰值学习率（原 2e-5 对 1.5B 模型偏大，改为 5e-6）")
    parser.add_argument("--entropy_coeff",    type=float, default=0.02,
                        help="Entropy 正则系数，防止 response entropy 坍缩；设 0 关闭")
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--micro_batch_size", type=int,   default=1)
    parser.add_argument("--max_steps",        type=int,   default=200)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--max_tokens",       type=int,   default=512,  help="vLLM 生成最大 token 数")
    parser.add_argument("--max_train_len",    type=int,   default=1024, help="训练序列最大截断长度")
    # 【修复】default 改为 None，运行时自动从 batch_size/micro_batch_size 推导
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="不填则自动推导为 batch_size // micro_batch_size")

    # 实验设置
    parser.add_argument("--dataset_size",   type=int,  default=None)
    parser.add_argument("--filter_correct", action="store_true")

    # 硬件与评估
    parser.add_argument("--device",           type=str,   default="cuda:0")
    parser.add_argument("--vllm_device",      type=str,   default="cuda:1")
    parser.add_argument("--vllm_gpu_util",    type=float, default=0.45)
    parser.add_argument("--eval_every_steps", type=int,   default=20)
    parser.add_argument("--max_eval_samples", type=int,   default=100)

    # WandB
    parser.add_argument("--wandb_project",  type=str, default="sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_sft_experiment(args)