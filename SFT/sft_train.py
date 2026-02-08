import torch
import wandb
import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerLM
from adamw import AdamW
from shedule import CosineAnnealingWarmupScheduler
from cross_entropy import Cross_entropy
from clip_gradient_noem import Clip_gradient_noem
from checpoint_use import save_checkpoint, load_checkpoint
from tokenizers import Tokenizer

class SFTDataset(Dataset):
    """
    SFT数据集类
    
    数据格式要求 (JSONL):
    {
        "instruction": "用户指令",
        "input": "可选的输入上下文",
        "output": "期望的输出"
    }
    
    或者简化格式:
    {
        "prompt": "完整的提示词",
        "response": "期望的回答"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_encode_fn,
        max_seq_len: int = 512,
        prompt_template: str = "default"
    ):
        """
        Args:
            data_path: JSONL格式的数据文件路径
            tokenizer_encode_fn: 分词器编码函数
            max_seq_len: 最大序列长度
            prompt_template: 提示词模板类型
        """
        self.data = []
        self.tokenizer_encode = tokenizer_encode_fn
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"加载了 {len(self.data)} 条SFT数据")
    
    def format_prompt(self, item: Dict) -> Tuple[str, str]:
        """
        格式化提示词
        
        Returns:
            (prompt, response) 元组
        """
        if "prompt" in item and "response" in item:
            # 简化格式
            return item["prompt"], item["response"]
        
        # 标准格式: instruction + input + output
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if self.prompt_template == "default":
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        elif self.prompt_template == "alpaca":
            if input_text:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        elif self.prompt_template == "chat":
            if input_text:
                prompt = f"User: {instruction}\n{input_text}\n\nAssistant: "
            else:
                prompt = f"User: {instruction}\n\nAssistant: "
        
        else:
            raise ValueError(f"未知的模板类型: {self.prompt_template}")
        
        return prompt, output
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回: (input_ids, labels, attention_mask)
        """
        item = self.data[idx]
        prompt, response = self.format_prompt(item)
        
        # 组合完整文本
        full_text = prompt + response
        
        # 编码
        prompt_ids = self.tokenizer_encode(prompt)
        full_ids = self.tokenizer_encode(full_text)
        
        # 截断到最大长度
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
        
        # 创建labels: 只计算response部分的损失
        # prompt部分的label设为-100 (忽略)
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        
        # 如果截断了,调整labels长度
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]
        
        # Padding
        input_ids = full_ids + [0] * (self.max_seq_len - len(full_ids))
        labels = labels + [-100] * (self.max_seq_len - len(labels))
        attention_mask = [1] * len(full_ids) + [0] * (self.max_seq_len - len(full_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def collate_fn(batch):
    """数据批次整理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


def compute_sft_loss(logits, labels):
    """
    计算SFT损失 (只在非-100的位置计算)
    
    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
    """
    # 展平
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # 只计算非-100位置的损失
    vocab_size = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )
    
    return loss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SFT Training for Transformer Language Model')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE的theta参数')
    
    # SFT训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=5e-5, help='最大学习率 (SFT通常较低)')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='预热步数比例')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 实验参数
    parser.add_argument("--no_rms_norm", action="store_true", help="移除RMSNorm")
    parser.add_argument("--norm_rope", type=str, default="pre", choices=["pre", "post"], 
                       help="Normalization位置")
    parser.add_argument("--no_rope", action="store_true", help="禁用RoPE")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"],
                       help="前馈层类型")
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True, 
                       help="训练集路径 (JSONL格式)")
    parser.add_argument("--valid_data_path", type=str, default=None, 
                       help="验证集路径 (JSONL格式)")
    parser.add_argument("--prompt_template", type=str, default="default",
                       choices=["default", "alpaca", "chat"],
                       help="提示词模板")
    
    # 分词器参数
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="分词器路径 (用于加载编码/解码函数)")
    
    # 检查点参数
    parser.add_argument('--pretrain_checkpoint', type=str, required=True,
                       help='预训练模型检查点路径')
    parser.add_argument('--checkpoint_dir', type=str, default='sft_checkpoints',
                       help='SFT检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='保存检查点的间隔步数')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从SFT检查点恢复训练')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=50, help='打印日志的间隔步数')
    parser.add_argument('--eval_interval', type=int, default=200, help='评估的间隔步数')
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='transformer-sft',
                       help='wandb项目名')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='wandb运行名称')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    
    # 设备参数
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader工作进程数')
    
    return parser.parse_args()



def evaluate(model, dataloader, device, max_eval_batches=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_eval_batches and i >= max_eval_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids, use_cache=False)
            loss = compute_sft_loss(logits, labels)
            
            # 统计
            non_pad_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return avg_loss


def train(args):
    """SFT主训练函数"""
    
    # 设置数据类型
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # 加载分词器
    print("加载分词器...")
    encode_fn = Tokenizer.encode
    
    # 加载数据集
    print("加载SFT数据集...")
    train_dataset = SFTDataset(
        data_path=args.train_data_path,
        tokenizer_encode_fn=encode_fn,
        max_seq_len=args.max_seq_len,
        prompt_template=args.prompt_template
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = None
    if args.valid_data_path:
        print("加载验证数据集...")
        val_dataset = SFTDataset(
            data_path=args.valid_data_path,
            tokenizer_encode_fn=encode_fn,
            max_seq_len=args.max_seq_len,
            prompt_template=args.prompt_template
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # 处理实验参数
    actual_rope_theta = None if args.no_rope else args.theta
    use_rms_norm = not args.no_rms_norm
    
    # 初始化模型
    print("初始化模型...")
    model = TransformerLM(
        d_model=args.d_model,
        n_head=args.n_head,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_ff=args.d_ff,
        theta=actual_rope_theta,
        n_layer=args.n_layer,
        device=args.device,
        dtype=dtype,
        use_rms_norm=use_rms_norm,
        norm_model=args.norm_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    
    # 加载预训练权重
    print(f"从预训练检查点加载: {args.pretrain_checkpoint}")
    checkpoint = torch.load(args.pretrain_checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("预训练权重加载成功!")
    
    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    
    # 计算总步数
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    
    # 初始化学习率调度器
    scheduler = CosineAnnealingWarmupScheduler(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # 从SFT检查点恢复
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        print(f"从SFT检查点恢复: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        start_epoch = start_step // len(train_dataloader)
        print(f"从epoch {start_epoch}, step {start_step} 恢复训练")
    
    # 开始训练
    print("开始SFT训练...")
    model.train()
    global_step = start_step
    running_loss = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 获取当前学习率
            current_lr = scheduler.get_lr_cosine_shedule(global_step)
            
            # 更新优化器学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # 数据移到设备
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # 前向传播
            logits = model(input_ids, use_cache=False)
            loss = compute_sft_loss(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            Clip_gradient_noem(model.parameters(), args.max_grad_norm)
            
            # 优化器步进
            optimizer.step()
            
            # 累积损失
            running_loss += loss.item()
            global_step += 1
            
            # 日志记录
            if global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                print(f"Epoch [{epoch+1}/{args.num_epochs}] "
                      f"Step [{global_step}/{total_steps}] | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e}")
                
                if args.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch + 1,
                        'train/step': global_step
                    })
                
                running_loss = 0.0
            
            # 评估
            if val_dataloader and global_step % args.eval_interval == 0:
                print("开始评估...")
                val_loss = evaluate(model, val_dataloader, args.device, max_eval_batches=50)
                print(f"Validation Loss: {val_loss:.4f}")
                
                if args.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'train/step': global_step
                    })
            
            # 保存检查点
            if global_step % args.save_interval == 0:
                checkpoint_path = checkpoint_dir / f'sft_checkpoint_step_{global_step}.pt'
                save_checkpoint(model, optimizer, global_step, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = checkpoint_dir / 'sft_checkpoint_final.pt'
    save_checkpoint(model, optimizer, global_step, final_checkpoint_path)
    print(f"\n最终SFT模型已保存: {final_checkpoint_path}")
    
    # 结束wandb运行
    if args.use_wandb:
        wandb.finish()
    
    print("\nSFT训练完成!")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
