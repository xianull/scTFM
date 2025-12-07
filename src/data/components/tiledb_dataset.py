import torch
from torch.utils.data import Dataset
import tiledb
import numpy as np
from scipy.sparse import coo_matrix

class TileDBCollator:
    def __init__(self, tiledb_path: str, n_genes: int, ctx_cfg: dict = None):
        self.tiledb_path = tiledb_path
        self.n_genes = n_genes
        # Default config optimized for concurrent reading without file locks
        self.ctx_cfg = ctx_cfg if ctx_cfg else {
            "sm.compute_concurrency_level": "1",
            "sm.io_concurrency_level": "1",
            "vfs.file.enable_filelocks": "false",
        }
        self._ctx = None
        self._array = None

    def _get_array(self):
        if self._ctx is None:
            self._ctx = tiledb.Ctx(self.ctx_cfg)
        if self._array is None:
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
        return self._array

    def __call__(self, batch_indices):
        # batch_indices 是数据集中的整数索引列表
        if not batch_indices:
            return torch.empty(0, self.n_genes), torch.empty(0)
            
        # 确保索引是列表或数组
        indices = batch_indices
        
        A = self._get_array()
        
        # 使用 multi_index 进行批量读取 - 这是优化的关键
        # 它在一个 C++ 调用中获取所请求细胞的所有数据
        try:
            results = A.multi_index[indices, :]
        except Exception as e:
            # 如果出现问题（例如 context/handle 无效），尝试重新打开
            # 这是一个基本的恢复机制
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
            results = self._array.multi_index[indices, :]

        # 识别维度和属性
        # 基于用户之前的代码，我们假设通用处理或特定名称。
        # 为了安全起见，我们会动态获取维度名称，但如果需要，回退到 "gene_index"。
        dom = A.schema.domain
        dim0_name = dom.dim(0).name # 细胞维度
        dim1_name = dom.dim(1).name # 基因维度
        attr_name = A.schema.attr(0).name # 数据属性
        
        cell_coords = results[dim0_name]
        gene_coords = results[dim1_name]
        data_vals = results[attr_name]
        
        # 我们需要将全局细胞索引映射回批次行索引 (0..batch_size-1)
        # 创建一个从 global_index 到 batch_row_index 的映射
        # 注意：传递给此函数的 'indices' 与批次的顺序相匹配
        idx_map = {global_idx: i for i, global_idx in enumerate(indices)}
        
        # 向量化映射可能会更快，但这样更稳健
        row_indices = np.array([idx_map[c] for c in cell_coords])
        
        # 构建稀疏矩阵然后转为密集矩阵
        # shape: (batch_size, n_genes)
        mat = coo_matrix(
            (data_vals, (row_indices, gene_coords)), 
            shape=(len(indices), self.n_genes)
        )
        
        # 转换为密集的 torch tensor
        # float32 是标准格式
        batch_x = torch.from_numpy(mat.toarray()).float()
        
        return batch_x, torch.tensor(indices)

    def __getstate__(self):
        """
        在 fork/spawn 时丢弃不可 pickle 的 TileDB 句柄。
        """
        state = self.__dict__.copy()
        state["_ctx"] = None
        state["_array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 句柄将被延迟重新创建

class TileDBDataset(Dataset):
    def __init__(self, tiledb_path: str, indices: np.ndarray, n_genes: int):
        super().__init__()
        self.tiledb_path = tiledb_path
        self.indices = indices
        self.n_genes = n_genes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 仅返回全局索引。
        # 繁重的工作由 Collator 完成。
        return self.indices[idx]
