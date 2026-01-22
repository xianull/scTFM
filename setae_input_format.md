# SetAE 模型输入格式总结

## 数据流

### 1. 数据集输出 (`SomaNeighborhoodDataset`)

从 `neighborhood_datamodule.py` 和 `neighborhood_dataset.py` 可以看到：

```python
# Dataset 返回的 batch 格式
{
    'x': torch.Tensor,           # shape: (batch_size, bag_size, n_genes)
    'counts': torch.Tensor,      # shape: (batch_size, bag_size, n_genes)
    'library_size': torch.Tensor, # shape: (batch_size, bag_size)
    'mask': torch.Tensor,        # shape: (batch_size, n_genes) - 可选，用于 MaskedSetAE
}
```

**维度说明**：
- `batch_size`: 每个 batch 中的 **bag 数量**（微环境数量）
- `bag_size`: 每个 bag 中的 **细胞数量**（set_size，默认 16）
- `n_genes`: 基因特征维度（默认 28231）

### 2. 三个 SetAE 模型的输入

所有三个模型都接受相同的输入格式：

#### GraphSetAE
```python
def forward(
    self,
    x_set: torch.Tensor,        # (batch_size, bag_size, n_genes)
    library_size: Optional[torch.Tensor] = None,  # (batch_size, bag_size)
    adj: Optional[torch.Tensor] = None,  # (batch_size, bag_size, bag_size) - 邻接矩阵
    center_idx: int = 0,
    **kwargs
) -> Dict[str, torch.Tensor]:
```

#### ContrastiveSetAE
```python
def forward(
    self,
    x_set: torch.Tensor,        # (batch_size, bag_size, n_genes)
    library_size: Optional[torch.Tensor] = None,  # (batch_size, bag_size)
    center_idx: int = 0,
    **kwargs
) -> Dict[str, torch.Tensor]:
```

#### MaskedSetAE
```python
def forward(
    self,
    x_set: torch.Tensor,        # (batch_size, bag_size, n_genes)
    library_size: Optional[torch.Tensor] = None,  # (batch_size, bag_size)
    center_idx: int = 0,
    gene_mask: Optional[torch.Tensor] = None,  # (batch_size, n_genes) - 预定义掩码
    hvg_indices: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
```

## 输入格式总结

| 参数 | 形状 | 说明 |
|------|------|------|
| `x_set` | `(batch_size, bag_size, n_genes)` | **主要输入**：细胞集合（中心细胞 + 邻居细胞） |
| `library_size` | `(batch_size, bag_size)` | 每个细胞的 library size（可选，会自动计算） |
| `adj` | `(batch_size, bag_size, bag_size)` | 邻接矩阵（仅 GraphSetAE 需要） |
| `gene_mask` | `(batch_size, n_genes)` | 基因掩码（仅 MaskedSetAE 需要） |
| `center_idx` | `int` | 中心细胞在 bag 中的索引（默认 0） |

## 关键点

1. **统一输入格式**：三个模型都使用 `(batch_size, bag_size, n_genes)` 格式
   - `batch_size` = 每个 batch 的微环境（bag）数量
   - `bag_size` = 每个微环境中的细胞数量（set_size，默认 16）
   - `n_genes` = 基因特征维度

2. **微环境概念**：
   - 每个 bag 包含来自同一个 tissue 的 `bag_size` 个细胞
   - 这些细胞构成一个微环境
   - 第一个细胞（`center_idx=0`）通常是中心细胞

3. **数据预处理**：
   - `x`: log1p normalized 的表达数据
   - `counts`: expm1(x) = normalized counts（用于计算 NB loss）
   - `library_size`: 每个细胞的总 counts

4. **模型特定输入**：
   - **GraphSetAE**: 需要 `adj` 邻接矩阵（如果提供）
   - **MaskedSetAE**: 可以使用 `gene_mask` 或自动生成掩码
   - **ContrastiveSetAE**: 不需要额外输入

## 示例

```python
# 假设配置
batch_size = 64      # 每个 batch 64 个微环境
bag_size = 16        # 每个微环境 16 个细胞
n_genes = 28231      # 基因数量

# 输入形状
x_set.shape          # (64, 16, 28231)
library_size.shape   # (64, 16)
adj.shape            # (64, 16, 16) - 如果使用 GraphSetAE
```
