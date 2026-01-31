#!/usr/bin/env python3
"""
分布式训练启动工具
简化torchrun命令的使用
"""

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='简化的分布式训练启动工具')
    
    # 分布式配置
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='使用的GPU数量(默认使用所有可用GPU)')
    parser.add_argument('--master_port', type=int, default=29500,
                       help='主节点端口')
    
    # 多节点配置
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='节点数量')
    parser.add_argument('--node_rank', type=int, default=0,
                       help='当前节点rank')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='主节点地址')
    
    # 训练脚本
    parser.add_argument('--script', type=str, default='train_distributed.py',
                       help='训练脚本路径')
    
    # 其他参数将传递给训练脚本
    args, unknown = parser.parse_known_args()
    
    # 检测可用GPU数量
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            args.num_gpus = available_gpus
        elif args.num_gpus > available_gpus:
            print(f"警告: 请求{args.num_gpus}个GPU,但只有{available_gpus}个可用")
            sys.exit(1)
    except ImportError:
        print("警告: 无法导入torch,无法检测GPU数量")
        if args.num_gpus is None:
            args.num_gpus = 1
    
    # 构建torchrun命令
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.num_gpus}',
        f'--master_port={args.master_port}',
    ]
    
    # 多节点配置
    if args.num_nodes > 1:
        cmd.extend([
            f'--nnodes={args.num_nodes}',
            f'--node_rank={args.node_rank}',
            f'--master_addr={args.master_addr}',
        ])
    
    # 添加训练脚本和参数
    cmd.append(args.script)
    cmd.append('--distributed')
    cmd.extend(unknown)
    
    # 打印命令
    print("执行命令:")
    print(' '.join(cmd))
    print()
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练失败,退出码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)


if __name__ == '__main__':
    main()
