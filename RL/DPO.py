"""
DPO (Direct Preference Optimization) 训练脚本
使用 TRL 库对自定义 Transformer 模型进行偏好学习
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig
from trl import DPOTrainer, DPOConfig
from typing import Optional
import json
from transformer import TransformerLM


class TransformerLMConfig(PretrainedConfig):
    """配置类，用于 HuggingFace 兼容"""
    model_type = "transformer_lm"
    
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        d_ff: int = 2048,
        theta: float = 10000.0,
        n_layer: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.theta = theta
        self.n_layer = n_layer


class HFTransformerLM(PreTrainedModel):
    """
    包装TransformerLM 使其兼容 HuggingFace Transformers
    """
    config_class = TransformerLMConfig
    
    def __init__(self, config: TransformerLMConfig):
        super().__init__(config)
        
        # 初始化你的模型
        self.model = TransformerLM(
            d_model=config.d_model,
            n_head=config.n_head,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_ff=config.d_ff,
            theta=config.theta,
            n_layer=config.n_layer,
            device=self.device,
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        HuggingFace 标准的 forward 接口
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (可选)
            labels: [batch, seq_len] (用于计算loss)
        """
        # 获取 logits
        logits = self.model(input_ids, use_cache=False)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 返回 HuggingFace 标准输出
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    def generate(self, input_ids, max_length=50, **kwargs):
        """简单的贪婪生成"""
        self.model.clear_cache()
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # Prefill或Generate
            logits = self.model(generated, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成了结束符 (假设 EOS token_id = 50256)
            if (next_token == 50256).all():
                break
                
        self.model.clear_cache()
        return generated


class PreferenceDataset(Dataset):
    """
    DPO 偏好数据集
    
    数据格式:
    {
        "prompt": "用户问题",
        "chosen": "更好的回答",
        "rejected": "较差的回答"
    }
    """
    
    def __init__(self, data_path: str, tokenizer):
        """
        Args:
            data_path: JSON 或 JSONL 文件路径
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
        self.data = []
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                for line in f:
                    self.data.append(json.loads(line))
            else:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 格式化为完整文本
        # prompt + chosen/rejected
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
        }


def create_tokenizer():
    """
    创建一个简单的分词器
    你可以替换为实际的 BPE/SentencePiece tokenizer
    """
    from transformers import AutoTokenizer
    
    # 使用 GPT-2 tokenizer 作为示例
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def main():
    # 配置 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型配置
    model_config = TransformerLMConfig(
        d_model=512,
        n_head=8,
        vocab_size=30000,  
        max_seq_len=1024,
        d_ff=2048,
        theta=10000.0,
        n_layer=6,
    )
    
    # 初始化模型 
    model = HFTransformerLM(model_config).to(device)
    
    # 可选：加载预训练的检查点
    # model.load_state_dict(torch.load('pretrained_model.pt'))
    
    # 创建参考模型（frozen copy）
    ref_model = HFTransformerLM(model_config).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 加载数据
    tokenizer = create_tokenizer()
    
    train_dataset = PreferenceDataset(
        data_path='train_preferences.jsonl',
        tokenizer=tokenizer
    )
    
    eval_dataset = PreferenceDataset(
        data_path='eval_preferences.jsonl',
        tokenizer=tokenizer
    )
    
    # DPO 训练配置
    training_args = DPOConfig(
        output_dir='./dpo_output',
        
        # 基础训练参数
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        
        # 学习率
        learning_rate=5e-6,
        lr_scheduler_type='cosine',
        warmup_steps=100,
        
        # DPO 特定参数
        beta=0.1,  # DPO 温度参数，控制偏好强度
        loss_type='sigmoid',  # 'sigmoid' 或 'hinge' 或 'ipo'
        
        # 优化器
        optim='adamw_torch',
        max_grad_norm=1.0,
        
        # 评估和保存
        evaluation_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
        
        # 日志
        logging_steps=10,
        report_to='tensorboard',
        
        # 其他
        bf16=True,  # 使用 bfloat16 混合精度
        remove_unused_columns=False,
        
        # 序列长度
        max_length=512,
        max_prompt_length=256,
    )
    
    # 初始化 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        
        # 可选：自定义数据整理函数
        # data_collator=custom_collator,
    )
    
    # 开始训练
    print("开始 DPO 训练...")
    dpo_trainer.train()
    
    # 保存模型 
    dpo_trainer.save_model('./dpo_final_model')
    tokenizer.save_pretrained('./dpo_final_model')
    
    print("训练完成！模型已保存到 ./dpo_final_model")


if __name__ == '__main__':
    main()