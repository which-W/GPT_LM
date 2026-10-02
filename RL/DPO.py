"""将自定义语言模型接入 Hugging Face，并使用真实偏好数据进行 DPO。"""
import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformer import TransformerLM
from checkpoint_use import get_checkpoint_config, load_tokenizer
from utils.generation import MODEL_KEYS, sampling_probs


class TransformerLMConfig(PretrainedConfig):
    """保存模型结构、词表及终止符配置。"""
    model_type = "transformer_lm"

    def __init__(self, d_model=512, n_head=8, vocab_size=4642, max_seq_len=1024,
                 d_ff=2048, theta=10000.0, n_layer=6, use_rms_norm=True,
                 norm_model="pre", ffn_type="swiglu", **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.d_model, self.n_head, self.vocab_size = d_model, n_head, vocab_size
        self.max_seq_len, self.d_ff, self.theta, self.n_layer = max_seq_len, d_ff, theta, n_layer
        self.use_rms_norm, self.norm_model, self.ffn_type = use_rms_norm, norm_model, ffn_type


class HFTransformerLM(PreTrainedModel):
    """训练时支持填充掩码，生成时每次仅向缓存加入新令牌。"""
    config_class = TransformerLMConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerLM(**{key: getattr(config, key) for key in MODEL_KEYS})

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.model.ln_output

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        loss = None
        if labels is not None:
            labels = labels.clone()
            if attention_mask is not None:
                labels.masked_fill_(~attention_mask.bool(), -100)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                                     labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_length=None, max_new_tokens=None,
                 do_sample=False, temperature=1.0, top_k=None, top_p=1.0,
                 eos_token_id=None, pad_token_id=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("生成输入必须是非空的二维令牌张量")
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.clone()
        if mask.shape != input_ids.shape or not mask.bool().any(-1).all():
            raise ValueError("每个提示词至少需要一个有效令牌")
        budget = max_new_tokens if max_new_tokens is not None else (max_length or 50) - input_ids.shape[1]
        if budget < 0:
            raise ValueError("生成长度不能小于提示词长度")
        budget = min(budget, self.config.max_seq_len - input_ids.shape[1])
        if budget <= 0:
            return input_ids.clone()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = ([eos] if isinstance(eos, int) else eos) or []
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id
        pad = pad if pad is not None else (eos_ids[0] if eos_ids else 0)
        generated = input_ids.clone()
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.model.clear_cache()
        try:
            logits = self.model(input_ids, use_cache=True, attention_mask=mask)
            positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
            last = positions.masked_fill(~mask.bool(), -1).max(-1).values
            scores = logits[torch.arange(input_ids.size(0), device=input_ids.device), last]
            for step in range(budget):
                if hasattr(self.config, "valid_token_ids"):
                    valid = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
                    valid[self.config.valid_token_ids] = True
                    scores = scores.masked_fill(~valid, float("-inf"))
                probs = sampling_probs(scores, temperature=temperature if do_sample else 0,
                                       top_k=top_k if top_k and top_k > 0 else None, top_p=top_p)
                next_ids = torch.multinomial(probs, 1)
                next_ids[finished] = pad
                generated = torch.cat([generated, next_ids], dim=1)
                mask = torch.cat([mask, (~finished).long().unsqueeze(-1)], dim=1)
                if eos_ids:
                    finished |= torch.isin(next_ids.squeeze(-1), torch.tensor(eos_ids, device=input_ids.device))
                if finished.all() or step + 1 == budget:
                    break
                scores = self.model(next_ids, use_cache=True, attention_mask=mask)[:, -1]
            return generated
        finally:
            self.model.clear_cache()


class PreferenceDataset(Dataset):
    """读取包含 prompt、chosen、rejected 三个文本字段的偏好数据。"""
    def __init__(self, data_path, tokenizer=None):
        self.tokenizer = tokenizer
        with open(data_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()] if str(data_path).endswith(".jsonl") else json.load(f)
        if not self.data or any(any(not isinstance(item.get(k), str) for k in ("prompt", "chosen", "rejected")) for item in self.data):
            raise ValueError("偏好数据不能为空，且必须包含 prompt、chosen、rejected 文本字段")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {key: self.data[index][key] for key in ("prompt", "chosen", "rejected")}


def create_tokenizer(path="tokenizer.json"):
    """使用项目本地分词器，避免另一个词表中的令牌越界。"""
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    return tokenizer


def load_policy(checkpoint_path=None, tokenizer_path="tokenizer.json", **model_options):
    """优先使用检查点内嵌分词器及结构；无检查点时创建小型模型。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    if checkpoint is not None:
        config = get_checkpoint_config(checkpoint)
        if config.get("model_type", "dense") != "dense":
            raise ValueError("DPO/GRPO 的当前包装器仅支持基础 Transformer 检查点")
        tokenizer_backend = load_tokenizer(checkpoint, tokenizer_path, config["vocab_size"])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
        options = {key: config[key] for key in MODEL_KEYS}
    else:
        tokenizer = create_tokenizer(tokenizer_path)
        options = dict(d_model=256, n_head=4, vocab_size=len(tokenizer), max_seq_len=1024, d_ff=1024, n_layer=4)
        options.update(model_options)
    hf_config = TransformerLMConfig(**options, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    hf_config.valid_token_ids = sorted(tokenizer.get_vocab().values())
    model = HFTransformerLM(hf_config)
    if checkpoint is not None:
        model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, tokenizer


def main():
    from datasets import Dataset as HFDataset
    from trl import DPOTrainer, DPOConfig
    parser = argparse.ArgumentParser(description="使用本地偏好数据进行 DPO")
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--val_data_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--tokenizer_path", default="tokenizer.json")
    parser.add_argument("--output_dir", default="dpo_output")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    train = HFDataset.from_list(PreferenceDataset(args.train_data_path).data)
    validation = HFDataset.from_list(PreferenceDataset(args.val_data_path).data) if args.val_data_path else None
    model, tokenizer = load_policy(args.checkpoint_path, args.tokenizer_path)
    ref_model = copy.deepcopy(model).eval().requires_grad_(False)
    options = dict(output_dir=args.output_dir, num_train_epochs=3, max_steps=args.max_steps,
                   per_device_train_batch_size=4, per_device_eval_batch_size=4,
                   gradient_accumulation_steps=4, learning_rate=5e-6, beta=0.1,
                   max_length=min(512, model.config.max_seq_len),
                   max_prompt_length=min(256, model.config.max_seq_len // 2),
                   bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                   gradient_checkpointing=False, remove_unused_columns=False, report_to="none")
    fields = inspect.signature(DPOConfig).parameters
    options["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "steps" if validation else "no"
    training_args = DPOConfig(**options)
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=training_args, train_dataset=train,
                         eval_dataset=validation, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
