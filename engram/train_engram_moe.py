"""
Engram + MoE Transformer 训练脚本
演示如何训练结合条件记忆和混合专家的模型
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict
import os
import argparse
from utils.token_data import TokenWindowDataset, TOKEN_DTYPES
from checkpoint_use import model_config
from engram.engram_moe_transformer import EngramMoETransformerLM, FlexibleEngramMoELM


class SimpleTextDataset(Dataset):
    """简单的文本数据集用于演示"""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机序列
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return tokens


def train_engram_moe_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "./checkpoints",
    log_interval: int = 10,
    max_steps: int = 0,
    eval_steps: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """按轮次或总步数训练；限制验证批次数，并保存最终状态。"""
    from utils.precision import TrainingPrecision
    if min(n_epochs, log_interval) < 1 or min(max_steps, eval_steps) < 0:
        raise ValueError("轮数和日志间隔必须为正，步数限制不能为负")
    if not len(train_loader):
        raise ValueError("训练数据不足一个批次")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = TrainingPrecision(device, dtype)
    engram_params, other_params = [], []
    for name, param in model.named_parameters():
        (engram_params if 'engram' in name else other_params).append(param)
    # 记忆模块使用五倍学习率，并关闭权重衰减。
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 0.1},
        {'params': engram_params, 'lr': learning_rate * 5, 'weight_decay': 0.0},
    ])
    budget = max_steps or n_epochs * len(train_loader)
    epochs = (budget + len(train_loader) - 1) // len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
    os.makedirs(checkpoint_dir, exist_ok=True)
    global_step, best_val_loss = 0, float('inf')
    print(f"设备：{device}，训练目标：{budget} 步，精度：{dtype}")

    def store(filename, epoch, val_loss=None):
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), iteration=global_step, epoch=epoch,
                     config=model_config(model), tokenizer_json=getattr(model, 'tokenizer_json', None))
        if val_loss is not None:
            state['val_loss'] = val_loss
        torch.save(state, os.path.join(checkpoint_dir, filename))

    for epoch in range(epochs):
        model.train()
        epoch_loss, batches = 0.0, 0
        for tokens in train_loader:
            tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            with precision.context():
                logits = model(tokens[:, :-1])
                lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
                total_loss = lm_loss + model.get_aux_loss()
            precision.backward(total_loss)
            precision.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            precision.step(optimizer)
            scheduler.step()
            global_step += 1
            batches += 1
            epoch_loss += total_loss.item()
            if global_step % log_interval == 0:
                print(f"训练步 {global_step}/{budget}，平均损失 {epoch_loss / batches:.4f}")
            if global_step >= budget:
                break
        print(f"轮次 {epoch + 1}，平均损失 {epoch_loss / batches:.4f}")
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device, eval_steps, precision)
            print(f"验证损失：{val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                store('best_model.pt', epoch, val_loss)
        if (epoch + 1) % 5 == 0:
            store(f'checkpoint_epoch_{epoch + 1}.pt', epoch)
        if global_step >= budget:
            break
    store('checkpoint_final.pt', epoch)
    print(f"训练完成，实际更新 {global_step} 步")
    return model


def evaluate_model(model, val_loader, device, max_steps=0, precision=None) -> float:
    """按实际处理批次计算验证均值，可限制验证步数。"""
    from contextlib import nullcontext
    model.eval()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)
            with precision.context() if precision else nullcontext():
                logits = model(tokens[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            total_loss += loss.item()
            batches += 1
            if max_steps and batches >= max_steps:
                break
    if not batches:
        raise ValueError("验证数据不足一个批次")
    return total_loss / batches


def main():
    """默认训练真实 token 数据；随机数据仅在显式演示模式下使用。"""
    parser = argparse.ArgumentParser(description='训练 Engram + MoE 语言模型')
    parser.add_argument('--train_data_path', default='data/TinyStories-train.bin')
    parser.add_argument('--valid_data_path', default='data/TinyStories-valid.bin')
    parser.add_argument('--data_dtype', choices=TOKEN_DTYPES, default='int64')
    parser.add_argument('--tokenizer_path', default='tokenizer_tinystories.json')
    parser.add_argument('--demo_random', action='store_true')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=0, help='总更新步数；零表示按轮数训练')
    parser.add_argument('--eval_steps', type=int, default=0, help='验证批次数；零表示完整验证')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--engram_layers', type=int, nargs='+', default=[1])
    parser.add_argument('--engram_embed_dim', type=int, default=64)
    parser.add_argument('--engram_table_size', type=int, default=8192)
    parser.add_argument('--device_ids', type=int, nargs='*', default=None)
    parser.add_argument('--checkpoint_dir', default='checkpoints_engram')
    args = parser.parse_args()
    if args.demo_random:
        train_dataset = SimpleTextDataset(128, args.seq_len + 1, args.vocab_size)
        val_dataset = SimpleTextDataset(32, args.seq_len + 1, args.vocab_size)
    else:
        train_dataset = TokenWindowDataset(args.train_data_path, args.seq_len, args.data_dtype, args.vocab_size, args.tokenizer_path)
        val_dataset = TokenWindowDataset(args.valid_data_path, args.seq_len, args.data_dtype, args.vocab_size, args.tokenizer_path)
    model = EngramMoETransformerLM(
        d_model=args.d_model, n_head=args.n_head, vocab_size=args.vocab_size,
        max_seq_len=args.seq_len, d_ff=args.d_ff, theta=10000, n_layer=args.n_layer,
        engram_layer_indices=args.engram_layers, engram_n_heads=4,
        engram_embed_dim=args.engram_embed_dim,
        engram_table_sizes={2: args.engram_table_size, 3: args.engram_table_size},
        n_experts=args.n_experts, top_k=args.top_k, device_ids=args.device_ids,
    )
    from pathlib import Path
    model.tokenizer_json = Path(args.tokenizer_path).read_text(encoding='utf-8')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    train_engram_moe_model(model, train_loader, val_loader, n_epochs=args.n_epochs,
                           learning_rate=args.learning_rate, device=model.main_device,
                           checkpoint_dir=args.checkpoint_dir, max_steps=args.max_steps,
                           eval_steps=args.eval_steps, log_interval=args.log_interval,
                           dtype=getattr(torch, args.dtype))


if __name__ == '__main__':
    main()
