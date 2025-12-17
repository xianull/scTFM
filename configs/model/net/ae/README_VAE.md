# VAE 配置指南

## 📁 配置文件概览

### 基础配置
- **`vae.yaml`** - 标准 VAE 配置
- **`vae_beta.yaml`** - β-VAE (增强解耦性)
- **`vae_lightweight.yaml`** - 轻量级 VAE (快速实验)

### 完整模型配置（包含训练参数）
- **`ae_stage1_vae.yaml`** - 标准 VAE 完整配置
- **`ae_stage1_beta_vae.yaml`** - β-VAE 完整配置

---

## 🔧 关键超参数

### 1. **KL 散度权重 (`kld_weight`)** - 最重要！

控制 KL 散度在总损失中的比重：
```yaml
model:
  kld_weight: 0.00001  # 配置在 model config 中
```

**推荐值**：
- **标准 VAE**: `1e-5` ~ `1e-4`
  - 平衡重构质量和潜在空间规整性
  - 适合大多数场景
  
- **β-VAE (解耦表示)**: `1e-3` ~ `1e-2`
  - 更强的 KL 约束，牺牲重构换取解耦性
  - 适合需要可解释潜在因子的场景
  
- **轻量 KL (防止后验坍塌)**: `1e-6` ~ `1e-5`
  - 几乎不约束 KL，重构优先
  - 适合初期训练或高维数据

**调参策略**：
```python
# 从小到大逐步增加
kld_weight: [1e-6, 1e-5, 1e-4, 1e-3]

# 监控指标
train/kl_loss      # 应该在 10-100 之间（过低说明后验坍塌）
train/recon_loss   # 重构损失（不应显著变差）
val/loss           # 验证损失（最终目标）
```

---

### 2. **潜在空间维度 (`latent_dim`)**

```yaml
model:
  net:
    latent_dim: 64  # 配置在 net config 中
```

**推荐值**：
- **小规模数据**: 32 ~ 64
- **中等规模**: 64 ~ 128
- **大规模/复杂数据**: 128 ~ 512

**权衡**：
- ✅ 更大：表达能力强，但计算量大
- ✅ 更小：快速训练，但可能欠拟合

---

### 3. **网络架构 (`hidden_dims`)**

```yaml
model:
  net:
    hidden_dims: [2048, 1024, 512]  # Encoder 层
    # Decoder 会自动镜像为 [512, 1024, 2048]
```

**推荐配置**：
```yaml
# 轻量级（快速实验）
hidden_dims: [1024, 512]

# 标准（平衡性能）
hidden_dims: [2048, 1024, 512]

# 深层（高表达能力）
hidden_dims: [4096, 2048, 1024, 512]
```

---

### 4. **激活函数 (`activation`)**

```yaml
model:
  net:
    activation: "GELU"  # LeakyReLU, GELU, SiLU, ReLU, SwiGLU
```

**性能对比**：
- **GELU**: 平滑，训练稳定，推荐首选 ✅
- **SiLU**: 类似 GELU，性能略好
- **LeakyReLU**: 计算快，传统选择
- **SwiGLU**: 最强表达能力，但需要更多参数 ⚠️

**注意**：SwiGLU 会自动使用 LayerNorm 而非 BatchNorm！

---

### 5. **正则化 (`dropout_rate`, `use_batch_norm`)**

```yaml
model:
  net:
    dropout_rate: 0.1     # Dropout 比例
    use_batch_norm: True  # 是否使用 Normalization
```

**推荐值**：
- **Dropout**: 0.05 ~ 0.15（过高会损害性能）
- **BatchNorm**: 通常建议启用（SwiGLU 除外）

---

## 📊 完整配置示例

### 示例1：标准 VAE（推荐起点）

```yaml
# configs/model/net/ae/vae.yaml
_target_: src.models.components.ae.vae.VariationalAE
input_dim: 28231
hidden_dims: [2048, 1024, 512]
latent_dim: 64
dropout_rate: 0.1
use_batch_norm: True
activation: "GELU"
```

```yaml
# configs/model/ae_stage1_vae.yaml
model:
  _target_: src.models.ae_module.AELitModule
  kld_weight: 0.00001  # 关键：KL 权重
  
  optimizer:
    lr: 0.0001
    weight_decay: 1e-5
```

**运行**：
```bash
python train.py experiment=stage1_ae model=ae_stage1_vae
```

---

### 示例2：β-VAE（增强解耦性）

```yaml
# configs/model/net/ae/vae_beta.yaml
hidden_dims: [4096, 2048, 1024, 512]  # 更深
latent_dim: 128                        # 更大
dropout_rate: 0.15
```

```yaml
# configs/model/ae_stage1_beta_vae.yaml
model:
  kld_weight: 0.001  # 10-100x 标准 VAE
```

**运行**：
```bash
python train.py experiment=stage1_ae model=ae_stage1_beta_vae
```

---

### 示例3：轻量级 VAE（快速实验）

```yaml
# configs/model/net/ae/vae_lightweight.yaml
hidden_dims: [1024, 512]  # 更浅
latent_dim: 32             # 更小
dropout_rate: 0.05
activation: "LeakyReLU"    # 更快
```

---

## 🔬 超参数 Sweep

### Sweep KL 权重（最重要）

```yaml
# configs/experiment/stage1_vae_sweep.yaml
hydra:
  sweeper:
    params:
      model.kld_weight: 1e-6,1e-5,1e-4,1e-3
      model.net.latent_dim: 32,64,128,256
```

**运行**：
```bash
python train.py experiment=stage1_vae_sweep
```

**总运行数**：4 (KL) × 4 (latent_dim) = 16 次

---

### 对比不同变体

```bash
# 命令行 Sweep
python train.py experiment=stage1_ae \
  model=ae_stage1_vae,ae_stage1_beta_vae \
  model.net.latent_dim=64,128 \
  -m
```

---

## 📈 监控指标

### 训练时关注：

1. **`train/recon_loss`** - 重构损失
   - 应该平滑下降
   - 最终值：0.01 ~ 0.1（取决于数据）

2. **`train/kl_loss`** - KL 散度
   - 应该在 10 ~ 100 之间
   - ⚠️ **过低（< 1）**：后验坍塌，需要增加 `kld_weight`
   - ⚠️ **过高（> 500）**：过度约束，降低 `kld_weight`

3. **`train/loss`** - 总损失
   - = recon_loss + kld_weight * kl_loss

4. **`val/loss`** - 验证损失
   - 最终优化目标

### W&B 可视化：

```python
# 在 W&B 中创建自定义图表
x: train/kl_loss
y: train/recon_loss
color: model.kld_weight
```

---

## ⚠️ 常见问题

### 1. **后验坍塌 (Posterior Collapse)**

**症状**：
- `train/kl_loss` 接近 0
- VAE 退化为普通 AE

**解决方案**：
```yaml
# 方法1：增加 KL 权重
model.kld_weight: 0.0001  # 从 1e-5 增加到 1e-4

# 方法2：KL 退火 (需要修改代码)
# 从 0 逐步增加到目标值

# 方法3：增大潜在空间
model.net.latent_dim: 128  # 从 64 增加到 128
```

---

### 2. **重构质量差**

**症状**：
- `train/recon_loss` 很高
- 重构图像模糊

**解决方案**：
```yaml
# 方法1：降低 KL 权重
model.kld_weight: 1e-6  # 从 1e-4 降低到 1e-6

# 方法2：增加网络容量
model.net.hidden_dims: [4096, 2048, 1024, 512]

# 方法3：降低 dropout
model.net.dropout_rate: 0.05
```

---

### 3. **训练不稳定 / NaN**

**解决方案**：
```yaml
# 已修复 SwiGLU！但如果还有问题：

# 方法1：降低学习率
model.optimizer.lr: 5e-5

# 方法2：增大梯度裁剪
trainer.gradient_clip_val: 1.0

# 方法3：切换激活函数
model.net.activation: "GELU"  # 不要用 SwiGLU
```

---

## 🎯 最佳实践

1. **从标准配置开始**
   ```bash
   python train.py experiment=stage1_ae model=ae_stage1_vae
   ```

2. **Sweep KL 权重**
   ```bash
   python train.py experiment=stage1_vae_sweep
   ```

3. **选择最佳 `kld_weight`**
   - 在 W&B 中对比 `val/loss`
   - 检查 `train/kl_loss` 是否在合理范围（10-100）

4. **Fine-tune 其他参数**
   - 调整 `latent_dim`
   - 调整网络深度/宽度
   - 调整学习率

5. **Benchmark 评估**
   ```bash
   python bench/benchmark_ae.py --dir logs/vae_kl_sweep/multiruns/
   ```

---

## 📚 参考文献

- **标准 VAE**: Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- **β-VAE**: Higgins et al. (2017) - β-VAE: Learning Basic Visual Concepts
- **SwiGLU**: Shazeer (2020) - GLU Variants Improve Transformer

---

## 🔗 相关配置

- **Vanilla AE**: `configs/model/net/ae/vanilla.yaml`
- **RAE (L2正则)**: `configs/model/net/ae/rae.yaml`
- **SAE (L1稀疏)**: `configs/model/net/ae/sae.yaml`

