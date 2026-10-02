"""统一模型加载与采样，保持 checkpoint、词表和生成参数一致。"""
import torch

from checkpoint_use import get_checkpoint_config, load_tokenizer, mask_tokenizer_logits
from transformer import TransformerLM


MODEL_KEYS = ('d_model', 'n_head', 'vocab_size', 'max_seq_len', 'd_ff', 'theta',
              'n_layer', 'use_rms_norm', 'norm_model', 'ffn_type')


def load_dense_model(path, tokenizer_path, device, config_overrides=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = get_checkpoint_config(checkpoint, config_overrides)
    tokenizer = load_tokenizer(checkpoint, tokenizer_path, config['vocab_size'])
    kind = config.get("model_type", "dense")
    base_keys = MODEL_KEYS[:7]
    options = {key: config[key] for key in base_keys}
    if "mhc" in kind:
        from mhc.transformer_mhc import TransformerLM as Model
        options.update({key: config[key] for key in ("n", "use_rms_norm", "norm_model", "ffn_type") if key in config})
        model = Model(**options, device=device)
    elif "moe" in kind or kind == "engram":
        if kind == "engram":
            from engram.engram_moe_transformer import EngramMoETransformerLM as Model
        elif "hybrid" in kind:
            from moe.moe_transformer import HybridMoETransformerLM as Model
        else:
            from moe.moe_transformer import MoETransformerLM as Model
        options.update({key: config[key] for key in ("n_experts", "top_k", "use_moe_aux_loss", "moe_aux_loss_weight",
                       "use_rms_norm", "moe_layer_indices", "engram_layer_indices", "engram_max_ngram",
                       "engram_n_heads", "engram_embed_dim", "engram_table_sizes") if key in config})
        target = torch.device(device)
        model = Model(**options, device_ids=[] if target.type == "cpu" else [target.index or 0], main_device=target.index or 0)
    else:
        model = TransformerLM(**{key: config[key] for key in MODEL_KEYS}, device=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    model.eval()
    return model, tokenizer, config


def sampling_probs(logits, temperature=1.0, top_k=None, top_p=1.0,
                   token_counts=None, repetition_penalty=1.0, tokenizer=None):
    if temperature < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
        raise ValueError('温度必须非负，top_p 须在 (0,1] 内，重复惩罚必须为正数')
    scores = logits.float().clone()
    if tokenizer is not None:
        scores = mask_tokenizer_logits(scores, tokenizer)
    for token_id, count in (token_counts or {}).items():
        penalty = repetition_penalty ** count
        score = scores[..., token_id]
        scores[..., token_id] = torch.where(score < 0, score * penalty, score / penalty)
    if temperature == 0:
        return torch.nn.functional.one_hot(scores.argmax(dim=-1), scores.shape[-1]).float()
    scores /= temperature
    if top_k is not None and top_k != 0:
        if top_k < 1:
            raise ValueError('top_k 必须为正数')
        cutoff = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, float('-inf'))
    if top_p < 1:
        sorted_scores, indices = scores.sort(dim=-1, descending=True)
        remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter(-1, indices, remove), float('-inf'))
    return scores.softmax(dim=-1)
