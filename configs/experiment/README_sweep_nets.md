# AE 模型 Sweep 配置说明

## 方案对比

### 方案A：直接 Sweep 配置组（推荐，简洁）

**配置文件**：`stage1_ae_sweep_nets.yaml`

**原理**：直接在 sweeper 中切换 `model/net/ae` 配置组

**使用方法**：
```bash
python train.py experiment=stage1_ae_sweep_nets
```

**优点**：
- ✅ 配置简洁，只需一个实验文件
- ✅ 容易添加新的 net 类型
- ✅ 可以灵活组合其他参数

**Sweep 组合**：
- `model/net/ae`: vanilla, vae (2种)
- `model.net.latent_dim`: 64, 128, 512 (3种)
- **总共运行数**：2 × 3 = 6 次

---

### 方案B：Sweep 模型配置（更稳妥）

**配置文件**：`stage1_ae_sweep_models.yaml`

**原理**：创建独立的模型配置文件，在 sweeper 中切换整个 model 配置

**新增文件**：
- `configs/model/ae_stage1_vanilla.yaml`
- `configs/model/ae_stage1_vae.yaml`

**使用方法**：
```bash
python train.py experiment=stage1_ae_sweep_models
```

**优点**：
- ✅ 每个模型类型可以有独立的优化器/scheduler 配置
- ✅ 更明确的配置结构
- ✅ 兼容性更好（适用于老版本 Hydra）

**Sweep 组合**：
- `model`: ae_stage1_vanilla, ae_stage1_vae (2种)
- `model.net.latent_dim`: 64, 128, 512 (3种)
- **总共运行数**：2 × 3 = 6 次

---

## 命令行覆盖（灵活方式）

如果你想临时测试，不修改配置文件，可以直接在命令行指定：

```bash
# 单次运行：vanilla + latent_dim=64
python train.py experiment=stage1_ae model/net/ae=vanilla model.net.latent_dim=64

# 单次运行：vae + latent_dim=128
python train.py experiment=stage1_ae model/net/ae=vae model.net.latent_dim=128

# Sweep：vanilla 和 vae
python train.py experiment=stage1_ae model/net/ae=vanilla,vae -m

# 完整 Sweep：测试多种组合
python train.py experiment=stage1_ae \
  model/net/ae=vanilla,vae \
  model.net.latent_dim=64,128,512 \
  model.optimizer.lr=1e-4,5e-4 \
  -m
```

---

## 扩展：添加更多 Net 类型

### 1. 添加新的 net 配置（如 rae.yaml 已存在）

只需在 sweeper 参数中添加：

```yaml
# stage1_ae_sweep_nets.yaml
hydra:
  sweeper:
    params:
      model/net/ae: vanilla,vae,rae,sae  # 添加 rae 和 sae
```

### 2. 创建对应的模型配置（方案B）

```bash
# 创建 configs/model/ae_stage1_rae.yaml
# 创建 configs/model/ae_stage1_sae.yaml
```

然后在 sweeper 中添加：
```yaml
hydra:
  sweeper:
    params:
      model: ae_stage1_vanilla,ae_stage1_vae,ae_stage1_rae,ae_stage1_sae
```

---

## 监控和查看结果

### W&B 分组

两种方案都会在 W&B 中创建清晰的分组：
- **方案A**：根据 `group: ae_nets_comparison` 分组
- **方案B**：根据 `group: ae_models_sweep` 分组

### 查看最佳模型

运行完成后，使用 benchmark 脚本评估：

```bash
# 评估所有 runs
python bench/benchmark_ae.py --dir logs/ae_stage1_sweep_nets/multiruns/

# 查看特定 run
python bench/benchmark_ae.py --dir logs/ae_stage1_sweep_nets/multiruns/2024-12-16_10-30-00/
```

---

## 推荐工作流

1. **快速测试**（命令行）：
   ```bash
   python train.py experiment=stage1_ae model/net/ae=vanilla,vae -m
   ```

2. **正式实验**（使用配置文件）：
   ```bash
   python train.py experiment=stage1_ae_sweep_nets
   ```

3. **大规模 Sweep**（组合多个参数）：
   编辑 `stage1_ae_sweep_nets.yaml`，添加更多参数后运行

4. **评估和对比**：
   ```bash
   python bench/benchmark_ae.py --dir logs/ae_stage1_sweep_nets/multiruns/
   ```

