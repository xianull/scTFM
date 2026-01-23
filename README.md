- 数据路径： `/fast/data/scTFM`
- 模型路径：`/fast/projects/scTFM`

## 自编码器（SetSCAE）

支持四种微环境建模策略的单细胞自编码器框架。

```plaintext
  SetSCAE (统一接口)
      ├── GraphSetAE       (GAT + 链路预测)
      ├── ContrastiveSetAE (对比学习)
      ├── MaskedSetAE      (掩码预测)
      └── StackSetAE       (条件嵌入) ← 新增
```

### 方案对比

|              | GraphSetAE              | ContrastiveSetAE    | MaskedSetAE                                  | StackSetAE                          |
|--------------|-------------------------|---------------------|----------------------------------------------|-------------------------------------|
| 编码器       | input_proj + GAT layers | cell_encoder (scVI) | input_proj + Transformer                     | cell_encoder + 条件注入              |
| 编码方式     | 图注意力聚合邻居信息     | 独立编码每个细胞     | 掩码中心 + 上下文编码                         | 邻居聚合 → 条件注入                  |
| 特有模块     | gat_layers              | projection_head     | mask_token, context_encoder, prediction_head | context attention, injection layer  |
| 特有损失     | 链路预测 (BCE)          | InfoNCE 对比损失     | 掩码基因预测 (MSE)                            | 无（仅重建损失）                     |
| 复杂度       | 高                      | 中                  | 高                                           | 低                                  |

### Loss

**所有策略共享**：NB 重建损失 (所有细胞)
```python
recon_loss = -log_nb_positive(target, mu_all, theta_all).mean()
```

**策略特定损失**：

| 策略          | 特定损失    | 公式                             |
|-------------|---------|--------------------------------|
| Graph       | 链路预测    | BCE(adj_pred, adj)             |
| Contrastive | InfoNCE | CrossEntropy(pos_sim, neg_sim) |
| Masked      | 掩码预测    | MSE(pred_genes[mask], x[mask]) |
| Stack       | 无       | 仅使用重建损失                    |

### StackSetAE 详解

基于 [Stack](https://www.nature.com/articles/s41592-024-02361-5) 论文的条件嵌入方法，核心思想是通过简单的条件注入机制将微环境信息融入细胞表征。

**工作流程**：
```
x_set (batch, set_size, n_genes)
    │
    ▼
┌─────────────────────────────────┐
│  cell_encoder (共享)             │  编码所有细胞
│  h_all = encoder(x_set)         │  → (batch, set_size, n_hidden)
└─────────────────────────────────┘
    │
    ├──► h_center = h_all[:, 0, :]     中心细胞
    │
    └──► h_neighbors = h_all[:, 1:, :] 邻居细胞
              │
              ▼
        ┌─────────────────────┐
        │  Context Aggregation │  聚合邻居信息
        │  (mean/attention/max)│
        └─────────────────────┘
              │
              ▼
          context (batch, n_hidden)
              │
              ▼
        ┌─────────────────────┐
        │  Condition Injection │  注入条件
        │  (concat/add/film)   │
        └─────────────────────┘
              │
              ▼
          z_center (batch, n_latent)  条件化的潜在表征
```

**配置选项**：

| 参数                  | 选项                        | 说明                                |
|----------------------|----------------------------|-------------------------------------|
| `context_aggregation`| `mean`, `attention`, `max` | 邻居信息聚合方式                     |
| `condition_injection`| `concat`, `add`, `film`    | 条件信息注入方式                     |
| `context_weight`     | float (默认 1.0)            | 上下文权重                           |
| `n_context_heads`    | int (默认 4)                | attention 聚合的头数                 |

**条件注入方式**：
- `concat`: `z = Linear([h_center; context])` - 拼接后投影
- `add`: `z = Linear(h_center + proj(context))` - 残差相加
- `film`: `z = Linear(γ(context) * h_center + β(context))` - Feature-wise Linear Modulation

**优势**：
1. 架构简单，计算效率高
2. 无额外损失函数，训练稳定
3. 灵活的配置组合（3×3=9种）
4. 与 scVI encoder 完全兼容

**使用示例**：
```python
from src.models.components.ae.set_scae import SetSCAE

model = SetSCAE(
    strategy='stack',
    n_input=28231,
    n_hidden=256,
    n_latent=64,
    n_layers=2,
    set_size=16,
    context_aggregation='attention',
    condition_injection='concat',
    context_weight=1.0,
)

outputs = model(x_set)  # x_set: (batch, set_size, n_genes)
z_center = outputs['z_center']  # (batch, n_latent) - 条件化表征
z_all = outputs['z_all']        # (batch, set_size, n_latent)
```

### 训练命令

```bash
# 同时跑四种策略 (multirun)
python src/train.py experiment=stage1_setscae --multirun

# 单独跑某种策略
python src/train.py experiment=stage1_setscae hydra.mode=RUN model.net.strategy=stack
python src/train.py experiment=stage1_setscae hydra.mode=RUN model.net.strategy=contrastive
```

### 更新日志

#### 2026-01-14
- /fast/projects/scTFM/models/ae/2026-01-16_04-13-53
  - z_all 包含所有细胞的 latent (batch, set_size, n_latent)
  - 但 decode_cell 只解码中心细胞
  - 重建损失只计算中心细胞

- 之前：每个 batch 只训练 batch_size 个细胞（中心细胞）
- 现在：每个 batch 训练 batch_size × set_size 个细胞

应该是对于每一个细胞，都要被选择作为一次中心细胞，然后进行微环境建模学习
- 同一个 tissue 内随机采样就是微环境
- 每个 forward pass 中，轮流让 bag 中的每个细胞作为中心，计算所有细胞的重建损失。
    - 折中方案，如果每个细胞再随机采的话太慢了

#### 2026-01-18
- 修复 DDP 未使用参数问题
  - 重构 BaseSetSCAE，移除基类中的 cell_encoder
  - 各子类自行定义编码器，避免参数冗余
- 新增 StackSetAE 策略（基于 Stack 论文的条件嵌入方法）

## RTF
### 2026-01-13
一致性不再使用`split_label`，训练的时候划分。
直接把数据集的细胞按链匹配
- `/fast/projects/scTFM/models/rtf/2026-01-15_10-07-57`


### 2026-01-19
实现了普通RTF的细胞天数筛选