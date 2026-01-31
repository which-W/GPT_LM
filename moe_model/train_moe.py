import torch
import wandb
import argparse
import os
import numpy as np
from pathlib import Path
from moe_model.moe_transformer import MoETransformerLM, HybridMoETransformerLM
from adamw import AdamW
from shedule import CosineAnnealingWarmupScheduler
from cross_entropy import Cross_entropy
from get_batch import get_batch
from clip_gradient_noem import Clip_gradient_noem
from checpoint_use import save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train MoE Transformer Language Model')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE的theta参数')
    
    # MoE特定参数
    parser.add_argument('--use_moe', action='store_true', help='使用MoE模型')
    parser.add_argument('--n_experts', type=int, default=8, help='专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='每个token激活的专家数')
    parser.add_argument('--use_moe_aux_loss', action='store_true', default=True, help='使用MoE负载均衡损失')
    parser.add_argument('--moe_aux_loss_weight', type=float, default=0.01, help='MoE辅助损失权重')
    
    # 混合MoE参数
    parser.add_argument('--use_hybrid_moe', action='store_true', help='使用混合MoE模型')
    parser.add_argument('--moe_layer_indices', type=str, default=None, 
                       help='使用MoE的层索引,用逗号分隔,例如"0,2,4,6"。如果为None则所有层都用MoE')
    
    # 多GPU参数
    parser.add_argument('--device_ids', type=str, default=None,
                       help='GPU设备列表,用逗号分隔,例如"0,1,2,3"')
    parser.add_argument('--main_device', type=int, default=0, help='主GPU设备ID')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='最大学习率')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--total_steps', type=int, default=10000, help='总训练步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 实验参数
    parser.add_argument("--no_rms_norm", action="store_true", help="移除RMSNorm")
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True, help="训练集路径")
    parser.add_argument("--valid_data_path", type=str, required=True, help="验证集路径")
    
    # 检查点参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe', help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存检查点的间隔步数')
    parser.add_argument('--resume_from', type=str, default=None, help='从检查点恢复训练')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100, help='打印日志的间隔步数')
    parser.add_argument('--eval_interval', type=int, default=500, help='评估的间隔步数')
    parser.add_argument('--eval_steps', type=int, default=100, help='评估步数')
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='moe-transformer-lm', help='wandb项目名')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='训练设备')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    
    return parser.parse_args()


def create_model(args, dtype):
    """根据参数创建模型"""
    
    # 处理device_ids
    device_ids = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(',')]
        print(f"使用多GPU: {device_ids}")
    
    use_rms_norm = not args.no_rms_norm
    
    # 如果使用混合MoE
    if args.use_hybrid_moe:
        # 处理moe_layer_indices
        moe_layer_indices = None
        if args.moe_layer_indices:
            moe_layer_indices = [int(x) for x in args.moe_layer_indices.split(',')]
            print(f"MoE层索引: {moe_layer_indices}")
        
        model = HybridMoETransformerLM(
            d_model=args.d_model,
            n_head=args.n_head,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            d_ff=args.d_ff,
            theta=args.theta,
            n_layer=args.n_layer,
            moe_layer_indices=moe_layer_indices,
            n_experts=args.n_experts,
            top_k=args.top_k,
            use_moe_aux_loss=args.use_moe_aux_loss,
            moe_aux_loss_weight=args.moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=args.main_device,
            dtype=dtype,
            use_rms_norm=use_rms_norm,
        )
        model.print_architecture()
        
    # 如果使用标准MoE
    elif args.use_moe:
        model = MoETransformerLM(
            d_model=args.d_model,
            n_head=args.n_head,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            d_ff=args.d_ff,
            theta=args.theta,
            n_layer=args.n_layer,
            n_experts=args.n_experts,
            top_k=args.top_k,
            use_moe_aux_loss=args.use_moe_aux_loss,
            moe_aux_loss_weight=args.moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=args.main_device,
            dtype=dtype,
            use_rms_norm=use_rms_norm,
        )
        print(f"MoE模型: {args.n_layer}层, 每层{args.n_experts}个专家, top-{args.top_k}")
    
    else:
        raise ValueError("请使用 --use_moe 或 --use_hybrid_moe 指定MoE模型类型")
    
    return model


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
    
    # 初始化模型
    model = create_model(args, dtype)
    
    # 打印模型参数量
    n_params = model.get_num_params(non_embedding=False)
    n_params_no_embed = model.get_num_params(non_embedding=True)
    print(f"模型总参数量: {n_params:,}")
    print(f"模型参数量(不含embedding): {n_params_no_embed:,}")
    
    # 设置优化器
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
    running_aux_loss = 0.0
    
    for step in range(start_step, args.total_steps):
        # 获取当前学习率
        current_lr = scheduler.get_lr_cosine_shedule(step)
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 获取批次数据
        x, y = get_batch(train_data, args.batch_size, args.max_seq_len, args.device)
        
        # 前向传播
        logits = model(x)
        
        # 计算主损失
        lm_loss = Cross_entropy(logits, y)
        
        # 获取MoE辅助损失
        aux_loss = model.get_aux_loss()
        
        # 总损失 = 语言模型损失 + MoE辅助损失
        total_loss = lm_loss + aux_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        Clip_gradient_noem(model.parameters(), args.max_grad_norm)
        
        # 优化器步进
        optimizer.step()
        
        # 累积损失
        running_loss += lm_loss.item()
        running_aux_loss += aux_loss.item()
        
        # 日志记录
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            avg_aux_loss = running_aux_loss / args.log_interval
            avg_total_loss = avg_loss + avg_aux_loss
            
            print(f"Step [{step+1}/{args.total_steps}] | "
                  f"LM Loss: {avg_loss:.4f} | "
                  f"Aux Loss: {avg_aux_loss:.6f} | "
                  f"Total Loss: {avg_total_loss:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    'train/lm_loss': avg_loss,
                    'train/aux_loss': avg_aux_loss,
                    'train/total_loss': avg_total_loss,
                    'train/learning_rate': current_lr,
                    'train/step': step + 1
                })
            
            running_loss = 0.0
            running_aux_loss = 0.0
        
        # 评估
        if val_data is not None and (step + 1) % args.eval_interval == 0:
            model.eval()
            val_lm_losses = []
            val_aux_losses = []
            
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.max_seq_len, args.device)
                    logits_val = model(x_val)
                    lm_loss_val = Cross_entropy(logits_val, y_val)
                    aux_loss_val = model.get_aux_loss()
                    
                    val_lm_losses.append(lm_loss_val.item())
                    val_aux_losses.append(aux_loss_val.item())
            
            val_lm_loss = np.mean(val_lm_losses)
            val_aux_loss = np.mean(val_aux_losses)
            val_total_loss = val_lm_loss + val_aux_loss
            
            model.train()
            
            print(f"Step [{step+1}/{args.total_steps}] | "
                  f"Val LM Loss: {val_lm_loss:.4f} | "
                  f"Val Aux Loss: {val_aux_loss:.6f} | "
                  f"Val Total Loss: {val_total_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    'val/lm_loss': val_lm_loss,
                    'val/aux_loss': val_aux_loss,
                    'val/total_loss': val_total_loss,
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
    
    # 打印配置信息
    print("=" * 80)
    print("MoE Transformer 训练配置")
    print("=" * 80)
    print(f"模型类型: {'混合MoE' if args.use_hybrid_moe else 'MoE'}")
    print(f"模型维度: {args.d_model}")
    print(f"层数: {args.n_layer}")
    print(f"专家数: {args.n_experts}")
    print(f"Top-K: {args.top_k}")
    print(f"辅助损失权重: {args.moe_aux_loss_weight}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大学习率: {args.max_lr}")
    print(f"总训练步数: {args.total_steps}")
    print("=" * 80)
    
    train(args)


if __name__ == "__main__":
    main()