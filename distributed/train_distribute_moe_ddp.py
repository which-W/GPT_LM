"""
MoE分布式训练脚本
结合DDP数据并行和专家并行
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import argparse
import os
import numpy as np
from pathlib import Path
from moe.moe_transformer import MoETransformerLM, HybridMoETransformerLM
from adamw import AdamW
from schedule import CosineAnnealingWarmupScheduler
from cross_entropy import Cross_entropy
from get_batch import get_batch
from clip_gradient_noem import Clip_gradient_noem
from checkpoint_use import save_checkpoint, load_checkpoint


def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # 初始化进程组
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}/{world_size}] Initialized distributed training")


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train MoE Transformer with DDP')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE的theta参数')
    
    # MoE特定参数
    parser.add_argument('--n_experts', type=int, default=8, help='每层的专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='每个token激活的专家数')
    parser.add_argument('--use_moe_aux_loss', action='store_true', default=True,
                       help='是否使用MoE负载均衡损失')
    parser.add_argument('--moe_aux_loss_weight', type=float, default=0.01,
                       help='MoE辅助损失权重')
    parser.add_argument('--use_hybrid_moe', action='store_true',
                       help='使用混合MoE模型(部分层使用MoE)')
    parser.add_argument('--moe_layer_indices', type=int, nargs='+', default=None,
                       help='使用MoE的层索引,例如: 2 5 8 11')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='每个GPU的批次大小')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='最大学习率')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--total_steps', type=int, default=5000, help='总训练步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='梯度累积步数')
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True, help="训练集")
    parser.add_argument("--valid_data_path", type=str, required=True, help="测试集")
    
    # 检查点参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe', 
                       help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存检查点的间隔步数')
    parser.add_argument('--resume_from', type=str, default=None, help='从检查点恢复训练')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100, help='打印日志的间隔步数')
    parser.add_argument('--eval_interval', type=int, default=500, help='评估的间隔步数')
    parser.add_argument('--eval_steps', type=int, default=100, help='评估步数')
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='moe-transformer-lm', 
                       help='wandb项目名')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    
    # 设备参数
    parser.add_argument('--dtype', type=str, default='float32', 
                       choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    
    # 分布式参数
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    parser.add_argument('--world_size', type=int, default=None, 
                       help='总GPU数量(默认使用所有可用GPU)')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                       help='分布式后端')
    parser.add_argument('--local_rank', type=int, default=-1, 
                       help='本地rank(由torch.distributed.launch设置)')
    
    # 专家并行配置
    parser.add_argument('--expert_parallel_size', type=int, default=None,
                       help='专家并行组大小(默认等于world_size,即所有GPU都用于专家并行)')
    
    return parser.parse_args()


def get_distributed_batch(data, batch_size, seq_len, device, rank, world_size):
    """
    为分布式训练获取批次数据
    确保不同rank获取不同的数据
    """
    # 每个rank使用不同的随机种子
    np.random.seed(rank + int(torch.randint(0, 1000000, (1,)).item()))
    
    indices = np.random.randint(0, len(data) - seq_len, size=(batch_size,))
    
    x_list = []
    y_list = []
    
    for idx in indices:
        chunk = torch.from_numpy(data[idx:idx + seq_len + 1].astype(np.int64))
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])
    
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    
    return x, y


def train_worker(rank, world_size, args):
    """每个进程的训练worker函数"""
    
    # 设置分布式环境
    if args.distributed:
        setup_distributed(rank, world_size, backend=args.backend)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_main_process = (rank == 0) or not args.distributed
    
    # 设置数据类型
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # 创建检查点目录(仅主进程)
    if is_main_process:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"MoE分布式训练配置")
        print(f"{'='*80}")
        print(f"模型: {'Hybrid MoE' if args.use_hybrid_moe else 'Full MoE'}")
        print(f"专家数: {args.n_experts}, Top-K: {args.top_k}")
        print(f"层数: {args.n_layer}, 模型维度: {args.d_model}")
        print(f"总GPU数: {world_size}")
        print(f"每GPU批次: {args.batch_size}, 梯度累积: {args.gradient_accumulation_steps}")
        print(f"有效批次大小: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"数据类型: {args.dtype}")
        print(f"{'='*80}\n")
    
    # 初始化wandb(仅主进程)
    if args.use_wandb and is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # 加载数据
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.train_data_path}")
    
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    if is_main_process:
        print(f"训练数据: {args.train_data_path}, 数据量: {len(train_data):,} tokens")
    
    val_data = None
    if args.valid_data_path:
        if not os.path.exists(args.valid_data_path):
            raise FileNotFoundError(f"验证数据文件不存在: {args.valid_data_path}")
        val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')
        if is_main_process:
            print(f"验证数据: {args.valid_data_path}, 数据量: {len(val_data):,} tokens")
    
    # 确定专家并行的GPU列表
    # 在DDP中,每个rank都有完整的模型副本,但MoE专家分布在不同GPU上
    if args.expert_parallel_size is None:
        args.expert_parallel_size = world_size
    
    device_ids = list(range(world_size))
    
    if is_main_process:
        print(f"\n专家并行配置:")
        print(f"  专家并行GPU数: {args.expert_parallel_size}")
        print(f"  使用的GPU列表: {device_ids}")
    
    # 初始化模型
    if args.use_hybrid_moe:
        model = HybridMoETransformerLM(
            d_model=args.d_model,
            n_head=args.n_head,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            d_ff=args.d_ff,
            theta=args.theta,
            n_layer=args.n_layer,
            moe_layer_indices=args.moe_layer_indices,
            n_experts=args.n_experts,
            top_k=args.top_k,
            use_moe_aux_loss=args.use_moe_aux_loss,
            moe_aux_loss_weight=args.moe_aux_loss_weight,
            device_ids=device_ids,
            main_device=rank,
            dtype=dtype,
        )
    else:
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
            main_device=rank,
            dtype=dtype,
        )
    
    if is_main_process:
        n_params = model.get_num_params(non_embedding=True)
        print(f"\n模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # 包装为DDP模型
    # 注意: MoE模型中的专家已经分布在多个GPU上
    # DDP主要用于同步梯度和参数更新
    if args.distributed:
        # 对于MoE模型,我们不使用标准的DDP包装
        # 因为专家已经分布在多个设备上,DDP会导致冲突
        # 相反,我们手动管理梯度同步
        if is_main_process:
            print(f"\n使用手动梯度同步进行MoE分布式训练")
        # model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
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
    if args.resume_from and is_main_process:
        print(f"从检查点恢复: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"从步数 {start_step} 恢复训练")
    
    # 同步起始步数
    if args.distributed:
        start_step_tensor = torch.tensor([start_step], device=device)
        dist.broadcast(start_step_tensor, src=0)
        start_step = int(start_step_tensor.item())
        dist.barrier()
    
    # 开始训练
    model.train()
    running_loss = 0.0
    running_aux_loss = 0.0
    
    if is_main_process:
        print(f"\n开始训练...")
        print(f"总步数: {args.total_steps}, 起始步数: {start_step}")
    
    for step in range(start_step, args.total_steps):
        # 获取当前学习率
        current_lr = scheduler.get_lr_cosine_shedule(step)
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 梯度累积循环
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_aux_loss = 0.0
        
        for micro_step in range(args.gradient_accumulation_steps):
            # 获取批次数据
            if args.distributed:
                x, y = get_distributed_batch(
                    train_data, args.batch_size, args.max_seq_len, device, rank, world_size
                )
            else:
                x, y = get_batch(train_data, args.batch_size, args.max_seq_len, device)
            
            # 前向传播
            logits = model(x)
            
            # 计算主损失
            lm_loss = Cross_entropy(logits, y)
            
            # 获取MoE辅助损失
            aux_loss = model.get_aux_loss()
            
            # 总损失 = 语言模型损失 + MoE负载均衡损失
            total_loss = lm_loss + aux_loss
            
            # 缩放损失(用于梯度累积)
            total_loss = total_loss / args.gradient_accumulation_steps
            
            # 反向传播
            total_loss.backward()
            
            accum_loss += lm_loss.item()
            accum_aux_loss += aux_loss.item()
        
        # 手动同步梯度(针对MoE模型)
        if args.distributed:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        # 梯度裁剪
        Clip_gradient_noem(model.parameters(), args.max_grad_norm)
        
        # 优化器步进
        optimizer.step()
        
        # 累积损失
        step_loss = accum_loss / args.gradient_accumulation_steps
        step_aux_loss = accum_aux_loss / args.gradient_accumulation_steps
        running_loss += step_loss
        running_aux_loss += step_aux_loss
        
        # 日志记录(仅主进程)
        if is_main_process and (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            avg_aux_loss = running_aux_loss / args.log_interval
            
            # 同步损失
            if args.distributed:
                loss_tensor = torch.tensor([avg_loss, avg_aux_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss, avg_aux_loss = loss_tensor.tolist()
            
            print(f"Step [{step+1}/{args.total_steps}] | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Aux Loss: {avg_aux_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/aux_loss': avg_aux_loss,
                    'train/total_loss': avg_loss + avg_aux_loss,
                    'train/learning_rate': current_lr,
                    'train/step': step + 1
                })
            
            running_loss = 0.0
            running_aux_loss = 0.0
        
        # 评估
        if val_data is not None and (step + 1) % args.eval_interval == 0:
            model.eval()
            val_losses = []
            val_aux_losses = []
            
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    if args.distributed:
                        x_val, y_val = get_distributed_batch(
                            val_data, args.batch_size, args.max_seq_len, device, rank, world_size
                        )
                    else:
                        x_val, y_val = get_batch(val_data, args.batch_size, args.max_seq_len, device)
                    
                    logits_val = model(x_val)
                    loss_val = Cross_entropy(logits_val, y_val)
                    aux_loss_val = model.get_aux_loss()
                    
                    val_losses.append(loss_val.item())
                    val_aux_losses.append(aux_loss_val.item())
            
            val_loss = np.mean(val_losses)
            val_aux_loss = np.mean(val_aux_losses)
            
            # 同步验证损失
            if args.distributed:
                val_loss_tensor = torch.tensor([val_loss, val_aux_loss], device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss, val_aux_loss = val_loss_tensor.tolist()
            
            model.train()
            
            if is_main_process:
                print(f"Step [{step+1}/{args.total_steps}] | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Aux Loss: {val_aux_loss:.6f}")
                
                if args.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/aux_loss': val_aux_loss,
                        'val/total_loss': val_loss + val_aux_loss,
                        'train/step': step + 1
                    })
        
        # 保存检查点(仅主进程)
        if is_main_process and (step + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step+1}.pt'
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        
        # 同步所有进程
        if args.distributed:
            dist.barrier()
    
    # 保存最终模型(仅主进程)
    if is_main_process:
        final_checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'
        save_checkpoint(model, optimizer, args.total_steps, final_checkpoint_path)
        print(f"最终模型已保存: {final_checkpoint_path}")
    
    # 结束wandb运行(仅主进程)
    if args.use_wandb and is_main_process:
        wandb.finish()
    
    if is_main_process:
        print("\n训练完成!")
    
    # 清理分布式环境
    if args.distributed:
        cleanup_distributed()


def main():
    args = parse_args()
    
    # 确定world_size
    if args.distributed:
        if args.world_size is None:
            args.world_size = torch.cuda.device_count()
        
        if args.world_size < 1:
            raise ValueError("分布式训练需要至少1个GPU")
        
        print(f"启动MoE分布式训练,使用 {args.world_size} 个GPU")
        
        # 使用torch.multiprocessing启动多进程
        mp.spawn(
            train_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # 单GPU训练
        print("启动单设备训练")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()