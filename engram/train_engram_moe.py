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
from checkpoint_use import save_checkpoint
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
    log_interval: int = 10
):
    """
    训练 Engram + MoE 模型
    
    参数：
        model: Engram+MoE模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        checkpoint_dir: 检查点保存目录
        log_interval: 日志打印间隔
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器
    # 论文中: Engram参数使用Adam,学习率放大5倍,无权重衰减
    # 其他参数使用Muon优化器(这里简化使用AdamW)

    engram_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'engram' in name:
            engram_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': learning_rate, 'weight_decay': 0.1},
        {'params': engram_params, 'lr': learning_rate * 5, 'weight_decay': 0.0}  # 论文配置
    ])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader)
    )

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    global_step = 0
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"开始训练 Engram + MoE 模型")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"批大小: {train_loader.batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"{'='*60}\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_aux_loss = 0.0

        for batch_idx, tokens in enumerate(train_loader):
            tokens = tokens.to(device)

            # 前向传播
            # 输入: [batch_size, seq_len]
            # 输出: [batch_size, seq_len, vocab_size]
            logits = model(tokens[:, :-1])  # 预测下一个token

            # 计算语言模型损失
            targets = tokens[:, 1:]  # 目标是下一个token
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='mean'
            )

            # 获取MoE辅助损失
            aux_loss = model.get_aux_loss()

            # 总损失 = 语言模型损失 + MoE辅助损失
            total_loss = lm_loss + aux_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 累计损失
            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_aux_loss += aux_loss.item()
            global_step += 1

            # 打印日志
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_lm_loss = epoch_lm_loss / (batch_idx + 1)
                avg_aux_loss = epoch_aux_loss / (batch_idx + 1)

                print(f"Epoch [{epoch+1}/{n_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(LM: {avg_lm_loss:.4f}, Aux: {avg_aux_loss:.6f}) "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Epoch结束
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_lm_loss = epoch_lm_loss / len(train_loader)
        avg_epoch_aux_loss = epoch_aux_loss / len(train_loader)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} 完成")
        print(f"平均损失: {avg_epoch_loss:.4f}")
        print(f"  - LM损失: {avg_epoch_lm_loss:.4f}")
        print(f"  - Aux损失: {avg_epoch_aux_loss:.6f}")

        # 验证
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device)
            print(f"验证损失: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': dict(model.config),
                    'tokenizer_json': getattr(model, 'tokenizer_json', None),
                }, checkpoint_path)
                print(f"✓ 保存最佳模型到 {checkpoint_path}")

        print(f"{'='*60}\n")

        # 保存定期检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': dict(model.config),
                'tokenizer_json': getattr(model, 'tokenizer_json', None),
            }, checkpoint_path)
            print(f"✓ 保存检查点到 {checkpoint_path}\n")

    print("训练完成!")
    return model


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='mean'
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    """默认训练真实 token 数据；随机数据仅在显式演示模式下使用。"""
    parser = argparse.ArgumentParser(description='训练 Engram + MoE 语言模型')
    parser.add_argument('--train_data_path', default='data/TinyStories-train.bin')
    parser.add_argument('--valid_data_path', default='data/TinyStories-valid.bin')
    parser.add_argument('--data_dtype', choices=TOKEN_DTYPES, default='int64')
    parser.add_argument('--tokenizer_path', default='tokenizer.json')
    parser.add_argument('--demo_random', action='store_true')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=1)
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
                           checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()
