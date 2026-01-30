# 分布式训练使用指南

## 概述

`train_distributed.py` 支持PyTorch的分布式数据并行(DDP)训练,可以在单机多卡或多机多卡环境下加速训练。

## 快速开始

### 1. 单GPU训练(基线)

```bash
python train_distributed.py \
    --train_data_path /data/train.bin \
    --valid_data_path /data/val.bin \
    --batch_size 8 \
    --total_steps 10000
```

### 2. 单机多GPU训练(推荐)

**方法A: 使用torchrun (PyTorch 1.10+)**

```bash
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --train_data_path /data/train.bin \
    --valid_data_path /data/val.bin \
    --batch_size 8 \
    --total_steps 10000
```

**方法B: 使用torch.distributed.launch**

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_distributed.py \
    --distributed \
    --train_data_path /data/train.bin \
    --valid_data_path /data/val.bin \
    --batch_size 8
```

**方法C: 直接运行(使用mp.spawn)**

```bash
python train_distributed.py \
    --distributed \
    --world_size 4 \
    --train_data_path /path/to/train.bin \
    --valid_data_path /path/to/val.bin \
    --batch_size 8
```

### 3. 多机多GPU训练

在每个节点上运行以下命令:

**主节点(Node 0, IP: 192.168.1.1):**

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \ #根据你具体的机器进行修改
    --master_port=29500 \
    set_distributed.py \
    --distributed \
    --train_data_path /path/to/train.bin \
    --valid_data_path /path/to/val.bin
```

**从节点(Node 1):**

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \ #根据你具体的机器进行修改
    --master_port=29500 \
    set_distributed.py \
    --distributed \
    --train_data_path /path/to/train.bin \
    --valid_data_path /path/to/val.bin
```

## 重要参数说明

### 分布式相关

- `--distributed`: 启用分布式训练
- `--world_size`: GPU总数(仅用于mp.spawn方法)
- `--backend`: 分布式后端,默认'nccl'(GPU),可选'gloo'(CPU)
- `--local_rank`: 本地rank,由启动器自动设置

### 性能优化

- `--batch_size`: **每个GPU的批次大小**
  - 总批次大小 = batch_size × world_size × gradient_accumulation_steps
  - 例: 4个GPU,batch_size=8,accumulation=2 → 总批次=64

- `--gradient_accumulation_steps`: 梯度累积步数
  - 用于模拟更大的批次大小而不增加显存
  - 默认1(不累积)

- `--dtype`: 数据类型
  - `float32`: 最稳定,显存占用最大
  - `float16`: 速度快,但可能不稳定
  - `bfloat16`: 平衡速度和稳定性(推荐,需A100或H100)

## 最佳实践

### 1. 批次大小设置

```bash
# 单GPU: batch_size=8
python train_distributed.py --batch_size 8

# 4个GPU: 保持总批次大小相同
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --batch_size 2  # 总批次=2×4=8
```

### 2. 使用梯度累积

```bash
# 模拟更大批次但不增加显存
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --batch_size 4 \
    --gradient_accumulation_steps 4
    # 有效批次大小 = 4 × 4 × 4 = 64
```

### 3. 混合精度训练

```bash
# 使用bfloat16加速训练(需要Ampere架构或更新)
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --dtype bfloat16 \
    --batch_size 16  # 可以使用更大的批次
```

### 4. 从检查点恢复

```bash
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --resume_from checkpoints/checkpoint_step_5000.pt \
    --train_data_path /path/to/train.bin \
    --valid_data_path /path/to/val.bin
```

### 5. 环境变量配置

```bash
# 设置NCCL参数优化性能
export NCCL_DEBUG=INFO  # 调试NCCL通信
export NCCL_IB_DISABLE=0  # 启用InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 多机训练
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500

torchrun --nproc_per_node=8 train_distributed.py --distributed ...
```

## 故障排查

### 问题1: "Address already in use"

```bash
# 更改master_port
torchrun --master_port=29501 ...
```

### 问题2: NCCL错误

```bash
# 使用gloo后端(较慢但更稳定)
python train_distributed.py --distributed --backend gloo ...
```

### 问题3: 多机训练连接失败

```bash
# 检查防火墙设置
sudo ufw allow 29500/tcp

# 检查网络连通性
ping 192.168.1.1

# 设置正确的网络接口
export NCCL_SOCKET_IFNAME=eth0
```

### 问题4: 显存溢出

```bash
# 减小批次大小
--batch_size 4

# 或使用梯度累积
--batch_size 2 --gradient_accumulation_steps 2

# 或使用混合精度
--dtype bfloat16
```

## 监控和日志

### 使用WandB

```bash
torchrun --nproc_per_node=4 train_distributed.py \
    --distributed \
    --use_wandb \
    --wandb_project "my-transformer" \
    --wandb_run_name "4gpu-bf16-experiment"
```

只有主进程(rank 0)会上传日志到WandB,避免重复记录。

### 查看训练进度

```bash
# 查看日志
tail -f train.log

# 使用nvidia-smi监控GPU
watch -n 1 nvidia-smi
```

## 代码修改说明

相比原始`train.py`,主要修改包括:

1. **添加分布式初始化**
   ```python
   dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
   ```

2. **使用DDP包装模型**
   ```python
   model = DDP(model, device_ids=[rank])
   ```

3. **分布式数据加载**
   ```python
   get_distributed_batch()  # 确保不同rank获取不同数据
   ```

4. **同步操作**
   ```python
   dist.all_reduce(loss_tensor)  # 同步损失
   dist.barrier()  # 同步所有进程
   ```

5. **主进程控制**
   - 只有rank 0保存检查点
   - 只有rank 0记录WandB日志
   - 只有rank 0打印详细日志

## 性能优化建议

1. **使用bfloat16**: 在支持的硬件上(A100, H100)使用bfloat16可提升速度约2倍
2. **调整批次大小**: 尽可能使用大批次以充分利用GPU
3. **启用TensorFloat32**: `torch.backends.cuda.matmul.allow_tf32 = True`
4. **使用InfiniBand**: 多机训练时使用高速网络
5. **数据预加载**: 使用SSD存储训练数据



