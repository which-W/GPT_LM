import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.post_training import SamplingParams, init_evaluator, load_policy_into_vllm_instance
import pandas as pd
# 【显存优化】减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# --- 导入自定义工具函数 ---
from utils.sft_util import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations,
    compute_entropy
)
from EI.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# ==========================================
# 辅助函数
# ==========================================

def get_batch(tokenized_data, batch_size, device):
    """从专家数据中采样 Batch，支持重复采样以保证 Step 逻辑。"""
    total_len = len(tokenized_data["input_ids"])
    if total_len == 0:
        raise ValueError("本轮没有可用于训练的专家样本")
    # 使用 np.random.choice 确保即使数据极少也能凑够一个 Batch
    batch_indices = np.random.choice(total_len, batch_size, replace=True)

    return {
        "input_ids": tokenized_data["input_ids"][batch_indices].to(device),
        "labels": tokenized_data["labels"][batch_indices].to(device),
        "response_mask": tokenized_data["response_mask"][batch_indices].to(device),
        "attention_mask": tokenized_data["attention_mask"][batch_indices].to(device),
        "label_mask": tokenized_data["label_mask"][batch_indices].to(device)
    }

def load_math12k_dataset(path, prompt_template=None):
    df = pd.read_parquet(path)
    processed_items = []
    for _, row in df.iterrows():
        q_text = row['problem']
        gold_answer = row['answer']

        if gold_answer:
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items

def load_gsm8k_dataset(path, prompt_template=None):

    processed_items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            item = json.loads(line)
            q_text = item['question']
            full_sol = item['answer']
            gold_answer = full_sol.split("####")[-1].strip() if "####" in full_sol else full_sol.strip()
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items


# ==========================================
# Expert Iteration 核心训练逻辑
# ==========================================

def run_expert_iteration(args):
    if args.micro_batch_size < 1 or args.batch_size < args.micro_batch_size or args.batch_size % args.micro_batch_size:
        raise ValueError("batch_size 必须能整除正数 micro_batch_size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 1. 基础配置
    grad_accum_steps = args.batch_size // args.micro_batch_size

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    if 'r1' in args.prompt_path.lower():
        print("使用 R1 评估模版 (零奖励函数)")
        reward_fn = r1_zero_reward_fn
    elif 'question_only' in args.prompt_path.lower():
        print("使用 Question-Only 评估模版")
        reward_fn = question_only_reward_fn
    else:
        raise ValueError("无法识别的评估模版，请确保 prompt_path 中包含 'r1' 或 'question_only' 以选择对应的奖励函数。")


    # 定义 WandB 坐标轴
    wandb.define_metric("global_step")
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")


    # 2. 模型与分词器初始化
    print(f"Initializing Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation
    ).to(args.device)
    policy.gradient_checkpointing_enable()

    optimizer = AdamW(policy.parameters(), lr=args.lr)
    amp_ctx = torch.amp.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=torch.device(args.device).type == "cuda")

    print(f"评估后端：{args.eval_backend}")
    vllm_inst = init_evaluator(policy, tokenizer, args)


    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()
    # 3. 数据池与验证集准备
    print("Loading data pools...")
    if str(args.train_data_path).lower().endswith('.parquet'):
        question_pool = load_math12k_dataset(args.train_data_path, prompt_template)
        val_samples = load_math12k_dataset(args.val_data_path, prompt_template)[:args.max_eval_samples]
    else:
        question_pool = load_gsm8k_dataset(args.train_data_path, prompt_template)
        val_samples = load_gsm8k_dataset(args.val_data_path, prompt_template)[:args.max_eval_samples]

    if not question_pool or not val_samples or min(args.rollouts, args.ei_batch_size, args.epochs_per_ei, args.eval_every_steps) < 1:
        raise ValueError("数据池、验证集及采样配置不能为空或为零")
    eval_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["</answer>"], include_stop_str_in_output=True)
    rollout_params = SamplingParams(n=args.rollouts, temperature=1.0, max_tokens=args.max_tokens,
                                    min_tokens=min(4, args.max_tokens), stop=["</answer>"], include_stop_str_in_output=True)

    # ----------------------------------------------------------
    # 4. Step 0 初始评估 (Baseline)
    # ----------------------------------------------------------
    print(f"\n[Step 0] 执行训练前初始评估...")
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_inst)
    metrics = log_generations(vllm_inst, eval_params,
                             [s['prompt'] for s in val_samples],
                             [s['gold'] for s in val_samples],
                             reward_fn, 0, "eval")
    print(f"Initial Accuracy: {metrics.get('eval/accuracy', 0):.2%}")

    # ----------------------------------------------------------
    # 5. Expert Iteration 主循环
    # ----------------------------------------------------------
    global_optim_step = 0

    for ei_step in range(args.n_ei_steps):
        print(f"\n{'='*20} 开始 EI 第 {ei_step + 1} 代演化 {'='*20}")

        # --- A. 采样阶段 (Rollout) ---
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)

        # 从池子中随机选题目进行“考试”
        batch_db = random.sample(question_pool, min(args.ei_batch_size, len(question_pool)))
        print(f">> 正在对 {len(batch_db)} 个问题进行采样 (G={args.rollouts})...")
        outputs = vllm_inst.generate([q['prompt'] for q in batch_db], rollout_params)

        # --- B. 过滤阶段 (Verify) ---
        expert_raw_data = []
        success_question_num = 0
        for i, output in enumerate(outputs):
            current_gold = batch_db[i]['gold']
            success_flag = 0
            for candidate in output.outputs:

                if reward_fn(candidate.text, current_gold)['reward'] == 1.0:
                    success_flag = 1
                    expert_raw_data.append({"prompt": batch_db[i]['prompt'], "response": candidate.text})
            success_question_num += success_flag

        success_rate = len(expert_raw_data) / (len(batch_db) * args.rollouts)

        print(f">> 采样成功率: {success_rate:.2%} | 获得专家样本: {len(expert_raw_data)} | 问题正确数量与比例：{success_question_num}/ {success_question_num/len(batch_db)}")
        wandb.log({"ei/success_rate": success_rate,
                    "ei/success_question_rate": success_question_num/len(batch_db),
                     "ei/collected_count": len(expert_raw_data),
                      "ei_step": ei_step + 1}, step=global_optim_step)


        # --- C. 动态训练阶段 (Training) ---
        # 核心逻辑：基于数据量动态计算训练步数
        train_steps = (len(expert_raw_data) * args.epochs_per_ei) // args.batch_size
        train_steps = max(1, train_steps)
        if not expert_raw_data:
            print("本轮没有正确候选答案，跳过训练并进入下一轮采样。")
            continue

        print(f">> 正在对新数据进行预分词...")
        tokenized_expert_data = tokenize_prompt_and_output(
            [ex['prompt'] for ex in expert_raw_data], [ex['response'] for ex in expert_raw_data], tokenizer,
            max_length=args.max_train_len,
        )

        print(f">> 启动 SFT 训练: 执行 {train_steps} 步更新 (等效 {args.epochs_per_ei} Epochs)...")
        policy.train()
        pbar = tqdm(range(train_steps), desc=f"EI-{ei_step+1} Training")

        for _ in pbar:
            acc_loss, acc_glob_ent, acc_res_ent = 0, 0, 0

            for _ in range(grad_accum_steps):
                batch = get_batch(tokenized_expert_data, args.micro_batch_size, args.device)

                with amp_ctx:
                    logits = policy(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits

                    # 显存优化 Log-Prob
                    lse = torch.logsumexp(logits.float(), dim=-1)
                    target_logits = torch.gather(logits, -1, batch["labels"].unsqueeze(-1)).squeeze(-1)
                    log_probs = target_logits - lse

                    # 熵计算
                    with torch.no_grad():
                        ent = compute_entropy(logits)
                        v_mask = batch["label_mask"].bool()
                        res_mask_bool = batch["response_mask"].bool() & v_mask
                        avg_res_ent = ent[res_mask_bool].mean().item() if res_mask_bool.any() else 0.0
                        avg_glob_ent = ent[v_mask].mean().item()

                    # 执行微批次训练步 (内含 backward)
                    loss, _ = sft_microbatch_train_step(
                        policy_log_probs=log_probs,
                        response_mask=batch["response_mask"],
                        gradient_accumulation_steps=grad_accum_steps,
                        normalize_constant=None
                    )
                    acc_loss += loss.item() * grad_accum_steps
                    acc_glob_ent += avg_glob_ent
                    acc_res_ent += avg_res_ent

            # 参数更新
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_optim_step += 1

            # 日志记录
            if global_optim_step % 5 == 0:
                wandb.log({
                    "train/loss": acc_loss / grad_accum_steps,
                    "train/global_entropy": acc_glob_ent / grad_accum_steps,
                    "train/response_entropy": acc_res_ent / grad_accum_steps,
                    "global_step": global_optim_step
                })

            if global_optim_step % args.eval_every_steps == 0:
                print(f"\n[Step {global_optim_step}] 训练中途评估...")
                policy.eval()
                # 必须同步权重，否则 vLLM 还在用本轮训练开始前的旧参数
                load_policy_into_vllm_instance(policy, vllm_inst)

                metrics = log_generations(
                    vllm_inst, eval_params,
                    [s['prompt'] for s in val_samples],
                    [s['gold'] for s in val_samples],
                    reward_fn,
                    global_optim_step,
                    "eval"
                )
                policy.train() # 切回训练模式

        # --- D. 迭代后评估 ---
        print(f">> 正在进行本轮迭代的验证...")
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        metrics = log_generations(vllm_inst, eval_params,
                                 [s['prompt'] for s in val_samples],
                                 [s['gold'] for s in val_samples],
                                 reward_fn, global_optim_step, "eval")
        print(f"Accuracy: {metrics.get('eval/accuracy', 0):.2%}")

        # 每轮迭代保存一次 Checkpoint
        save_path = os.path.join(args.output_dir, f"ei_iter{ei_step+1}_step{global_optim_step}")
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # 清理显存碎片
        torch.cuda.empty_cache()

    print("\n🎉 Expert Iteration 任务全部完成！")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Expert Iteration Dynamic Step Training")

    # 路径配置
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k-train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k-val.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/ei_checkpoints")

    # 基础训练参数
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_train_len", type=int, default=1024, help="专家样本的最大训练序列长度")

    # EI 动态参数
    parser.add_argument("--n_ei_steps", type=int, default=5, help="外层迭代轮数")
    parser.add_argument("--ei_batch_size", type=int, default=512, help="每一轮采样的题目数 Db")
    parser.add_argument("--rollouts", type=int, default=8, help="每一题生成的候选路径数 G")
    parser.add_argument("--epochs_per_ei", type=int, default=1, help="每一轮对新专家数据训练的次数")

    # 硬件与监控
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.2)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="cs336-ei-dynamic")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=4)

    parser.add_argument("--eval_backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--attn_implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    args = parser.parse_args()
    run_expert_iteration(args)
