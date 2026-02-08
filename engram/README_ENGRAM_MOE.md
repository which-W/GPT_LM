# Engram + MoE Transformer 架构

基于 DeepSeek Engram 论文实现的条件记忆 + 混合专家Transformer架构。

## 📚 论文背景

**论文**: [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)

**核心思想**:
- 语言建模包含两种不同的任务:
  - **动态推理**: 需要深度计算(逻辑组合、多步推理、代码生成等)
  - **静态模式召回**: 固定知识(实体名称、常见短语、公式等)
  
- 传统Transformer缺乏原生的知识查找机制,必须通过计算模拟检索
- Engram引入**条件记忆**作为与MoE**条件计算**互补的稀疏性维度

## 🏗️ 架构设计

### 1. Engram 条件记忆模块

#### 核心组件:

```
输入 token序列 → 词汇压缩 → N-gram提取 → 多头哈希 → 嵌入检索
                                                    ↓
输出 ← 短卷积 ← 门控调制 ← 上下文感知门控 ← Key/Value投影
```

#### 特性:

- **词汇压缩 (Tokenizer Compression)**
  - 将语义等价的token映射到规范形式
  - 例: "Apple" 和 " apple" → 相同ID
  - 论文中实现了23%的词汇量减少

- **多头哈希N-gram嵌入**
  - 支持2-gram和3-gram
  - 每个N-gram级别使用多个哈希函数减少碰撞
  - 使用质数大小的哈希表
  - O(1)查找复杂度

- **上下文感知门控**
  - Query: 当前隐藏状态(已聚合全局上下文)
  - Key/Value: 检索的记忆
  - 通过缩放点积计算门控权重
  - 过滤不相关或冲突的记忆

- **短深度卷积**
  - 因果卷积扩展感受野
  - SwiGLU激活 + 残差连接

### 2. MoE 混合专家

- 使用现有的专家并行MoE实现
- 每个专家分配到不同GPU
- Top-K路由机制
- 负载均衡损失

### 3. 完整Transformer Block

```
输入
  ↓
[可选] Engram层 (仅特定层)
  ↓
RMSNorm → 多头注意力 → 残差连接
  ↓
RMSNorm → MoE FFN → 残差连接
  ↓
输出
```

## 📊 最优容量分配

论文通过实验发现了**U型缩放定律**:

- **纯MoE (ρ=100%)**: 次优,浪费深度重构静态模式
- **纯Engram (ρ=0%)**: 次优,失去动态推理能力
- **最优分配**: ρ ≈ 75-80% (MoE), 20-25% (Engram)

### DeepSeek Engram-27B 配置:

```
总参数: 26.7B
MoE: 55个routed experts + 2个shared experts (top-6)
Engram: 5.7B参数
层数: 30层
Engram位置: 第2层和第15层
```

## 🎯 Engram 放置策略

论文发现**早期层放置效果最好**:

1. **第2层最优**: 
   - 经过一轮attention,已有足够上下文用于门控
   - 仍然足够早,可以卸载静态模式重构

2. **多层配置**:
   - 论文使用第2层和第15层
   - 平衡早期干预和后期精确门控

3. **系统考虑**:
   - 早期放置允许预取overlap通信
   - 充分利用内存层次结构

## 🔧 使用方法

### 1. 基本使用

```python
from engram_moe_transformer import EngramMoETransformerLM

model = EngramMoETransformerLM(
    d_model=512,
    n_head=8,
    vocab_size=10000,
    max_seq_len=256,
    d_ff=2048,
    theta=10000,
    n_layer=6,
    # Engram配置
    engram_layer_indices=[1, 4],  # 第1层和第4层使用Engram
    engram_max_ngram=3,
    engram_n_heads=8,
    engram_embed_dim=256,
    # MoE配置
    n_experts=8,
    top_k=2,
    device_ids=[0, 1, 2, 3],
    main_device=0
)

# 前向传播
token_ids = torch.randint(0, 10000, (2, 32))
logits = model(token_ids)

# 获取MoE辅助损失
aux_loss = model.get_aux_loss()
total_loss = lm_loss + aux_loss
```

### 2. 灵活配置

```python
from engram_moe_transformer import FlexibleEngramMoELM

# 早期层策略
model = FlexibleEngramMoELM(
    d_model=512,
    n_head=8,
    vocab_size=10000,
    max_seq_len=256,
    d_ff=2048,
    theta=10000,
    n_layer=12,
    engram_placement_strategy='early',  # 自动放在早期
    n_engram_layers=2,
    n_experts=16,
    top_k=2
)

# 分布式策略
model = FlexibleEngramMoELM(
    engram_placement_strategy='distributed',  # 均匀分布
    n_engram_layers=3
)

# 自定义策略
model = FlexibleEngramMoELM(
    engram_placement_strategy='custom',
    custom_engram_layers=[2, 7, 15]  # 自定义位置
)
```

### 3. 训练

```python
from train_engram_moe import train_engram_moe_model

# 训练模型
trained_model = train_engram_moe_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=10,
    learning_rate=3e-4,
    checkpoint_dir="./checkpoints"
)
```

**优化器配置** (遵循论文):
- Engram参数: Adam, 学习率×5, 无权重衰减
- 其他参数: AdamW/Muon, 标准学习率, 权重衰减0.1

## 📁 文件结构

```
.
├── engram_module.py                    # Engram条件记忆核心实现
├── engram_moe_transformer_block.py     # Engram+MoE Transformer Block
├── engram_moe_transformer.py           # 完整语言模型
├── train_engram_moe.py                 # 训练脚本
├── README_ENGRAM_MOE.md                # 本文档
├── moe_model/
│   ├── moe_experts.py                  # 专家并行MoE
│   ├── moe_layer.py                    # MoE层
│   ├── moe_router.py                   # MoE路由器
│   └── ...
          
```

## 🔬 实现细节

### 1. N-gram哈希

```python
# 多项式哈希
hash = (c0*id0 + c1*id1 + ... + cn*idn) % table_size

# 多头设计减少碰撞
n_heads = 8
每个头使用不同的哈希系数
```

### 2. 门控机制

```python
query = RMSNorm(hidden_state)
key = RMSNorm(W_K @ memory)
value = W_V @ memory

gate_score = (query · key) / √d_model
gate_weight = sigmoid(gate_score)
gated_output = gate_weight * value
```

### 3. 参数共享策略

- 单一Value投影矩阵 (所有分支共享)
- 多个Key投影矩阵 (每个分支独立)
- 融合为单个FP8矩阵乘法,最大化GPU利用率

## ⚡ 系统效率

### 1. 确定性检索

- 与MoE动态路由不同,Engram使用静态哈希ID
- 可以在forward前预计算索引
- 支持运行时预取

### 2. 通信-计算重叠

```
Layer 1 (Compute) → Layer 2 (Engram, PCIe Fetch) → Layer 3 (Compute)
                      ↑
                  早期层计算掩盖通信延迟
```

### 3. 内存层次

利用Zipfian分布:
- 高频N-gram: GPU HBM缓存
- 中频N-gram: Host DRAM
- 低频N-gram: NVMe SSD

论文实验: 100B参数表完全offload到CPU内存,吞吐量损失<3%

## 🎓 关键洞察

### 1. 有效深度增加

通过LogitLens和CKA分析:
- Engram的浅层 ≈ MoE的深层
- 早期层更快达到"预测就绪"状态
- 释放深度用于复杂推理

### 2. 注意力容量释放

- 将局部依赖委托给查找
- 注意力专注于全局上下文
- 长上下文性能大幅提升

### 3. 功能二分性

通过ablation实验:
- 事实知识严重依赖Engram (保留率29-44%)
- 阅读理解主要依赖骨干网络 (保留率81-93%)

## 📊 缩放定律

### 1. 分配定律 (U型曲线)

```
验证损失
    ↑
    |     ╱╲
    |    ╱  ╲
    |   ╱    ╲___
    |  ╱          ╲___
    | ╱                ╲___
    |╱______________________╲
    0%  20%  40%  60%  80%  100%
         MoE分配比例 (ρ)
    
最优点: ρ ≈ 75-80%
```

### 2. 无限内存缩放

```
验证损失 ∝ log(记忆槽数量)

线性缩放: 更大的记忆 → 更好的性能
无需增加计算量
```

## 🚀 未来方向

1. **更大规模验证**: 扩展到100B+参数
2. **自适应N-gram**: 动态调整N-gram大小
3. **领域特定记忆**: 针对特定任务的记忆配置
4. **可编辑性**: 支持记忆的增量更新
5. **多模态扩展**: 图像、视频的条件记忆

## 📝 引用

```bibtex
@article{cheng2025engram,
  title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author={Cheng, Xin and Zeng, Wangding and Dai, Damai and others},
  journal={arXiv preprint arXiv:2601.07372},
  year={2025}
}
```

## 📄 许可

本实现仅供学习和研究使用。

## 🙏 致谢

感谢DeepSeek团队的开创性工作,为LLM架构设计开辟了新的方向。

---

**注**: 这是基于论文的教学实现。生产环境使用需要进一步优化和测试。
