import torch
from torch.utils.data import IterableDataset, DataLoader
import tiledbsoma
import numpy as np
import os
import random
import gc
from typing import Optional, Dict, List
import scipy.sparse

class FlowSomaDataset(IterableDataset):
    """
    专门用于 Flow Matching 的数据集。
    支持两种模式：
    1. Latent Space: 读取提取好的 Dense Latent Code (latent_key="X_latent")
    2. Raw Space: 读取原始 Sparse Gene Expression (latent_key="ms/RNA/X/data")，并转为 Dense。
    """
    def __init__(
        self, 
        root_dir: str, 
        split_label: int = 0, 
        io_chunk_size: int = 4096, 
        batch_size: int = 256,
        latent_key: str = "X_latent",
        condition_keys: Optional[Dict[str, str]] = None
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.latent_key = latent_key
        
        # 默认条件键名映射
        self.condition_keys = condition_keys or {
            "time": "unified_time",
            "tissue": "tissue_code",
            "celltype": "celltype_code"
        }
        
        if not os.path.exists(root_dir):
             raise ValueError(f"❌ 路径不存在: {root_dir}")

        self.sub_uris = sorted([
            os.path.join(root_dir, d) 
            for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        print(f"🌍 [FlowDataset] 扫描 Shards: {len(self.sub_uris)} 个 | Layer: {latent_key}")

    def _get_context(self):
        return tiledbsoma.SOMATileDBContext()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_uris = self.sub_uris.copy()
        else:
            # 简单的分片逻辑
            import math
            per_worker = int(math.ceil(len(self.sub_uris) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.sub_uris))
            my_uris = self.sub_uris[start:end].copy()

        random.shuffle(my_uris)
        ctx = self._get_context()
        
        # 提取关键列名
        time_col = self.condition_keys.get("time", "unified_time")
        tissue_col = self.condition_keys.get("tissue", "tissue_code")
        celltype_col = self.condition_keys.get("celltype", "celltype_code")

        for uri in my_uris:
            try:
                with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                    # 1. 读取 Metadata
                    try:
                        obs_df = exp.obs.read(
                            value_filter=f"split_label == {self.split_label}",
                            column_names=["soma_joinid", "next_cell_idx", "is_valid_transition", 
                                          time_col, tissue_col, celltype_col]
                        ).concat().to_pandas()
                    except Exception as e:
                        # 容错：可能某些字段不存在
                        print(f"⚠️ Error reading obs from {uri}: {e}")
                        continue
                    
                    if len(obs_df) == 0: continue
                    
                    valid_df = obs_df[obs_df['is_valid_transition'] == 1]
                    if len(valid_df) == 0: continue
                    
                    # 2. 动态定位 Data Array
                    x_arr = None
                    if self.latent_key in exp:
                        x_arr = exp[self.latent_key]
                    elif '/' in self.latent_key:
                        # 解析路径: ms/RNA/X/data
                        parts = self.latent_key.split('/')
                        curr = exp
                        for p in parts:
                            curr = curr[p]
                        x_arr = curr
                    
                    if x_arr is None:
                         print(f"⚠️ Could not find array {self.latent_key} in {uri}")
                         continue

                    # 3. 读取数据 (处理 Sparse vs Dense)
                    # 如果是 Raw Data (Sparse)，必须转为 Dense
                    is_sparse = x_arr.soma_type == "SOMASparseNDArray"
                    
                    if is_sparse:
                        # Sparse 读取 -> COO -> Dense
                        # 注意：如果 Shard 很大且 input_dim 很大 (20k)，这里内存会暴涨
                        # Shard 建议控制在 2000-4000 细胞
                        table = x_arr.read().tables().concat()
                        if len(table) == 0: continue
                        
                        # 获取维度 (Shards 应该知道全局 shape 吗？或者只用 max index)
                        # 为了安全，这里读取到的 shape 只是当前 shard 的最大值
                        # 但 DiT 需要固定的 input_dim。
                        # 我们假设 x_arr.shape[1] 是正确的 feature 数量
                        n_vars = x_arr.shape[1] 
                        n_cells_shard = obs_df.index.max() + 1 # 假设 soma_joinid 是局部的 0..N-1
                        
                        # 构建 CSR
                        rows = table['soma_dim_0'].to_numpy()
                        cols = table['soma_dim_1'].to_numpy()
                        data = table['soma_data'].to_numpy()
                        
                        # 确保 rows 对应到 0..N-1
                        # 这里的 obs_df.index 就是 soma_joinid
                        # 如果是 Sparse，我们先构建大矩阵，再切片
                        full_matrix_sparse = scipy.sparse.csr_matrix(
                            (data, (rows, cols)), 
                            shape=(n_cells_shard, n_vars),
                            dtype=np.float32
                        )
                        full_latents = full_matrix_sparse.toarray()
                        
                    else:
                        # Dense 读取 (Latent)
                        full_latents = x_arr.read().to_numpy()

                    full_latents_t = torch.from_numpy(full_latents)
                    
                    # 4. 构建 Batch
                    indices = valid_df.index.values # Source Indices
                    np.random.shuffle(indices)
                    
                    num_samples = len(indices)
                    num_batches = (num_samples + self.batch_size - 1) // self.batch_size
                    
                    for b in range(num_batches):
                        batch_idx = indices[b*self.batch_size : (b+1)*self.batch_size]
                        
                        if len(batch_idx) <= 1: continue 
                        
                        # 获取数据
                        x_curr = full_latents_t[batch_idx]
                        
                        next_indices = obs_df.loc[batch_idx, 'next_cell_idx'].values.astype(int)
                        x_next = full_latents_t[next_indices]
                        
                        times = torch.tensor(obs_df.loc[batch_idx, time_col].values, dtype=torch.float32)
                        next_times = torch.tensor(obs_df.loc[next_indices, time_col].values, dtype=torch.float32)
                        dt = next_times - times
                        
                        tissues = torch.tensor(obs_df.loc[batch_idx, tissue_col].values, dtype=torch.long)
                        celltypes = torch.tensor(obs_df.loc[batch_idx, celltype_col].values, dtype=torch.long)
                        
                        yield {
                            'x_curr': x_curr,
                            'x_next': x_next,
                            'cond_meta': {
                                'time': times,
                                'dt': dt,
                                'tissue': tissues,
                                'celltype': celltypes
                            }
                        }

            except Exception as e:
                print(f"⚠️ Error processing {os.path.basename(uri)}: {e}")
                continue
