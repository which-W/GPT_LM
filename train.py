import torch
import wandb
import argparse
import os
import numpy as np
from pathlib import Path
from transformer import TransformerLM
from adamw import AdamW
from shedule import CosineAnnealingWarmupScheduler
from cross_entropy import Cross_entropy
from get_batch import get_batch
from clip_gradient_noem import Clip_gradient_noem
from checpoint_use import save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE的theta参数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='最大学习率')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--total_steps', type=int, default=5000, help='总训练步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    #实验参数
    #移除RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true",help="让RMSNORM移除")
    #pre-norm or pre-norm
    parser.add_argument("--norm_rope", type=str, default="pre", choices=["pre","post"], help="Normalization Placement")
    #移除RoPE
    parser.add_argument("--no_rope",action="store_true",help="禁用Rope")
    #SwiGLU or SiLU
    parser.add_argument("--ffn_type",type=str,default="swiglu",choices=["swiglu","silu"],help="feed-forward层选择")
    
    
    # 数据参数
    parser.add_argument("--train_data_path",type=str,required=True,help="训练集")
    parser.add_argument("--valid_data_path",type=str,required=True,help="测试集")
    
    
    # 检查点参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存检查点的间隔步数')
    parser.add_argument('--resume_from', type=str, default=None, help='从检查点恢复训练')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100, help='打印日志的间隔步数')
    parser.add_argument('--eval_interval', type=int, default=500, help='评估的间隔步数')
    parser.add_argument('--eval_steps', type=int, default=100, help='评估步数')
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='transformer-lm', help='wandb项目名')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='训练设备')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    
    return parser.parse_args()


def train(args):
    """主训练函数"""
    
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
    
    # 加载数据(使用memmap)
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.train_data_path}")
    
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    print(f"训练数据: {args.train_data_path}, 数据量: {len(train_data):,} tokens")
    
    val_data = None
    if args.valid_data_path:
        if not os.path.exists(args.valid_data_path):
            raise FileNotFoundError(f"验证数据文件不存在: {args.valid_data_path}")
        val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')
        print(f"验证数据: {args.valid_data_path}, 数据量: {len(val_data):,} tokens")
    
    #处理消融实验逻辑
    actual_rope_theta = None if args.no_rope else 10000.0
    #use_rms_norm逻辑取反
    use_rms_norm = not args.no_rms_norm
    
    # 初始化模型
    model = TransformerLM(
        d_model=args.d_model,
        n_head=args.n_head,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_ff=args.d_ff,
        theta=args.theta,
        n_layer=args.n_layer,
        device=args.device,
        dtype=dtype,
        use_rms_norm=use_rms_norm,
        norm_model=args.norm_rope,
        ffn_type=args.ffn_type,
        
    ).to(args.device)
    
    # 设置学习器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    
    # 初始化学习率调度器
    scheduler = CosineAnnealingWarmupScheduler(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps
    )
    
    # 从检查点恢复
    start_step = 0
    if args.resume_from:
        print(f"从检查点恢复: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"从步数 {start_step} 恢复训练")
    
    # 开始训练
    model.train()
    running_loss = 0.0
    
    for step in range(start_step, args.total_steps):
        # 获取当前学习率
        current_lr = scheduler.get_lr_cosine_shedule(step)
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 获取批次数据
        x, y = get_batch(train_data, args.batch_size, args.max_seq_len, args.device)
        
        # 前向传播
        logits = model(x,)
        loss = Cross_entropy(logits, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        Clip_gradient_noem(model.parameters(), args.max_grad_norm)
        
        # 优化器步进
        optimizer.step()
        
        # 累积损失
        running_loss += loss.item()
        
        # 日志记录
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            print(f"Step [{step+1}/{args.total_steps}] | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/learning_rate': current_lr,
                    'train/step': step + 1
                })
            
            running_loss = 0.0
        
        # 评估
        if val_data is not None and (step + 1) % args.eval_interval == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.max_seq_len, args.device)
                    logits_val = model(x_val)
                    loss_val = Cross_entropy(logits_val, y_val)
                    val_losses.append(loss_val.item())
            
            val_loss = np.mean(val_losses)
            model.train()
            
            print(f"Step [{step+1}/{args.total_steps}] | Validation Loss: {val_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'train/step': step + 1
                })
        
        # 保存检查点
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step+1}.pt'
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'
    save_checkpoint(model, optimizer, args.total_steps, final_checkpoint_path)
    print(f"最终模型已保存: {final_checkpoint_path}")
    
    # 结束wandb运行
    if args.use_wandb:
        wandb.finish()
    
    print("训练完成!")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()