import torch
from typing import List, Dict, Callable, Optional
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import wandb
from vllm import LLM, SamplingParams


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = 1024,  # 【新增】截断参数，防止超长样本把整批 pad 到天际导致 OOM
) -> Dict[str, torch.Tensor]:
    """
    对 prompt 和 response 进行分词、拼接，并生成 response_mask。

    max_length: 超过此长度的序列从右侧截断（优先保留 prompt 头部）。
                设为 None 则不截断（不推荐，容易 OOM）。
    """
    all_input_ids      = []
    all_response_masks = []
    all_lengths        = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)

        combined_ids = p_ids + o_ids
        mask         = [0] * len(p_ids) + [1] * len(o_ids)

        # 截断到 max_length
        if max_length is not None and len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
            mask         = mask[:max_length]

        all_input_ids.append(combined_ids)
        all_response_masks.append(mask)
        all_lengths.append(len(combined_ids))

    max_len    = max(all_lengths)
    batch_size = len(prompt_strs)
    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padded_masks     = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, m) in enumerate(zip(all_input_ids, all_response_masks)):
        length = len(ids)
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_masks[i, :length]     = torch.tensor(m)

    # Shift：input 取前 N-1，labels / mask 取后 N-1
    final_input_ids     = padded_input_ids[:, :-1]
    final_labels        = padded_input_ids[:, 1:].clone()
    final_response_mask = padded_masks[:, 1:]

    return {
        "input_ids":     final_input_ids,
        "labels":        final_labels,
        "response_mask": final_response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个位置的 next-token 预测熵。
    H = logsumexp(z) - sum(p_i * z_i)

    Args:
        logits: (batch_size, seq_len, vocab_size)
    Returns:
        entropy: (batch_size, seq_len)
    """
    lse        = torch.logsumexp(logits, dim=-1)
    probs      = F.softmax(logits, dim=-1)
    exp_logits = torch.sum(probs * logits, dim=-1)
    return lse - exp_logits


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """获取每个 token 的条件对数概率，可选返回熵。"""
    outputs       = model(input_ids)
    logits        = outputs.logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs     = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    results = {"log_probs": log_probs}
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)
    return results


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """掩码求和后除以归一化常数。"""
    masked_tensor = tensor * mask
    total_sum = torch.sum(masked_tensor) if dim is None else torch.sum(masked_tensor, dim=dim)
    return total_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """单次 micro-batch SFT 更新（含反向传播）。"""
    batch_size    = policy_log_probs.shape[0]
    nll_per_token = -policy_log_probs

    total_masked_loss    = masked_normalize(nll_per_token, response_mask, normalize_constant, dim=None)
    microbatch_loss_mean = total_masked_loss / batch_size
    scaled_loss          = microbatch_loss_mean / gradient_accumulation_steps

    scaled_loss.backward()
    return scaled_loss, {"loss": microbatch_loss_mean.detach()}


def log_generations(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    log_prefix: str = "eval",
) -> Dict[str, float]:
    """生成回答并记录评估指标到 wandb。"""
    outputs = vllm_model.generate(prompts, sampling_params)

    table_data = []
    all_lengths, correct_lengths, incorrect_lengths = [], [], []
    total_reward = total_format_reward = total_answer_reward = 0.0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer    = ground_truths[i]

        scores = reward_fn(generated_text, gold_answer)
        r  = scores.get("reward", 0.0)
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)

        resp_len = len(generated_text)
        all_lengths.append(resp_len)
        (correct_lengths if r > 0.5 else incorrect_lengths).append(resp_len)

        total_reward        += r
        total_format_reward += fr
        total_answer_reward += ar

        if i < 100:
            table_data.append([step, prompts[i], generated_text, gold_answer, r, fr, ar])

    n = len(prompts)
    metrics = {
        f"{log_prefix}/accuracy":             total_reward / n,
        f"{log_prefix}/format_score":         total_format_reward / n,
        f"{log_prefix}/answer_score":         total_answer_reward / n,
        f"{log_prefix}/avg_length":           np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct":   np.mean(correct_lengths)   if correct_lengths   else 0,
        f"{log_prefix}/avg_length_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    if wandb.run is not None:
        columns = ["step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log({f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)}, step=step)
        wandb.log(metrics, step=step)

    print(f"Step {step}: Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}")
    return metrics