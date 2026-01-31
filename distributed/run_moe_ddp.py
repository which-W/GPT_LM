#!/usr/bin/env python3
"""
MoE分布式训练便捷启动工具
简化torchrun命令的使用
"""

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='MoE分布式训练启动工具')
    
    # 分布式配置
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='使用的GPU数量(默认使用所有可用GPU)')
    parser.add_argument('--master_port', type=int, default=29500,
                       help='主节点端口')
    
    # 快速配置预设
    parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large'],
                       help='使用预设配置: small(4GPU), medium(4GPU大模型), large(8GPU)')
    
    # 数据路径
    parser.add_argument('--train_data', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True,
                       help='验证数据路径')
    
    # MoE配置
    parser.add_argument('--n_experts', type=int, default=8,
                       help='专家数量')
    parser.add_argument('--top_k', type=int, default=2,
                       help='Top-K专家数')
    parser.add_argument('--hybrid', action='store_true',
                       help='使用混合MoE模型')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=4,
                       help='每GPU批次大小')
    parser.add_argument('--grad_accum', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--total_steps', type=int, default=50000,
                       help='总训练步数')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    
    # WandB
    parser.add_argument('--wandb', action='store_true',
                       help='启用WandB日志')
    parser.add_argument('--wandb_project', type=str, default='moe-transformer',
                       help='WandB项目名')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='WandB运行名称')
    
    # 其他参数将传递给训练脚本
    args, unknown = parser.parse_known_args()
    
    # 检测可用GPU数量
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            args.num_gpus = available_gpus
        elif args.num_gpus > available_gpus:
            print(f"请求{args.num_gpus}个GPU,但只有{available_gpus}个可用")
            sys.exit(1)
    except ImportError:
        print("无法导入torch,无法检测GPU数量")
        if args.num_gpus is None:
            args.num_gpus = 1
    
    # 应用预设配置
    preset_configs = {
        'small': {
            'd_model': 512,
            'n_head': 8,
            'n_layer': 12,
            'd_ff': 2048,
            'n_experts': 8,
            'batch_size': 4,
            'grad_accum': 4,
        },
        'medium': {
            'd_model': 768,
            'n_head': 12,
            'n_layer': 16,
            'd_ff': 3072,
            'n_experts': 16,
            'batch_size': 2,
            'grad_accum': 8,
        },
        'large': {
            'd_model': 1024,
            'n_head': 16,
            'n_layer': 24,
            'd_ff': 4096,
            'n_experts': 32,
            'batch_size': 2,
            'grad_accum': 16,
        }
    }
    
    # 构建torchrun命令
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.num_gpus}',
        f'--master_port={args.master_port}',
        'train_moe_distributed.py',
        '--distributed',
        f'--train_data_path={args.train_data}',
        f'--valid_data_path={args.val_data}',
        f'--dtype={args.dtype}',
        f'--total_steps={args.total_steps}',
    ]
    
    # 应用预设或用户参数
    if args.preset:
        config = preset_configs[args.preset]
        print(f"\n使用预设配置: {args.preset}")
        print(f"  模型维度: {config['d_model']}")
        print(f"  层数: {config['n_layer']}")
        print(f"  专家数: {config['n_experts']}")
        print(f"  批次大小: {config['batch_size']} × {config['grad_accum']}")
        
        for key, value in config.items():
            cmd.append(f'--{key}={value}')
    else:
        # 使用用户指定的参数
        cmd.extend([
            f'--n_experts={args.n_experts}',
            f'--top_k={args.top_k}',
            f'--batch_size={args.batch_size}',
            f'--gradient_accumulation_steps={args.grad_accum}',
        ])
    
    # 混合MoE
    if args.hybrid:
        cmd.append('--use_hybrid_moe')
        if args.preset == 'large':
            cmd.append('--moe_layer_indices=4 8 12 16 20')
        else:
            cmd.append('--moe_layer_indices=2 5 8 11')
    
    # WandB
    if args.wandb:
        cmd.append('--use_wandb')
        cmd.append(f'--wandb_project={args.wandb_project}')
        if args.wandb_name:
            cmd.append(f'--wandb_run_name={args.wandb_name}')
        else:
            name = f"{args.num_gpus}gpu-{args.n_experts}experts"
            if args.preset:
                name = f"{args.preset}-{name}"
            cmd.append(f'--wandb_run_name={name}')
    
    # 添加额外参数
    cmd.extend(unknown)
    
    # 打印命令
    print("\n" + "="*80)
    print("执行命令:")
    print("="*80)
    print(' \\\n  '.join(cmd))
    print("="*80 + "\n")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败,退出码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)


if __name__ == '__main__':
    main()