import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import math
import os
import random
import gc
from typing import Optional, Dict, List

class SomaRTFDataset(IterableDataset):
    """
    用于 Rectified Flow 训练的 TileDB-SOMA 数据集。
    
    核心逻辑：
    1. 读取 TileDB shards，每个 shard 是一个独立的 Experiment
    2. 根据 obs 中的 'next_cell_id' 或 'prev_cell_id' 构建细胞对
    3. 返回格式: {x_curr, x_next, cond_meta}
    4. 支持 Latent 和 Raw 两种模式（通过 latent_key 控制）
    
    参数:
        root_dir: TileDB SOMA 根目录（包含多个 Experiment Shards）
        split_label: 数据集划分标签 (0=Train, 1=Val, 2=Test ID, 3=Test OOD)
        io_chunk_size: TileDB 读取块大小
        batch_size: 批次大小
        measurement_name: 测量名称 (默认 "RNA")
        latent_key: 潜在空间键名 (None=Raw模式, "X_latent"=Latent模式)
        direction: 细胞对方向 ('forward'=使用next_cell_id, 'backward'=使用prev_cell_id, 'both'=混合)
        preloaded_sub_uris: 预扫描的 shard URI 列表（可选，由 DataModule 提供）
        shard_assignment: 智能负载均衡方案（可选，由 DataModule 提供）
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split_label: int = 0, 
        io_chunk_size: int = 16384, 
        batch_size: int = 256, 
        measurement_name: str = "RNA",
        latent_key: Optional[str] = None,  # None=Raw, "X_latent"=Latent
        direction: str = "forward",  # "forward", "backward", "both"
        preloaded_sub_uris: Optional[List[str]] = None,
        shard_assignment: Optional[Dict[str, List[str]]] = None
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self.latent_key = latent_key  # 关键：控制读取 Latent 还是 Raw
        self.direction = direction
        self._n_vars = None  # 延迟加载
        self.shard_assignment = shard_assignment
        
        # [关键优化] 如果 DataModule 提供了预扫描的列表，直接使用
        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None  # 延迟加载（向后兼容）
        
        if not os.path.exists(root_dir):
            raise ValueError(f"❌ 路径不存在: {root_dir}")
    
    @property
    def sub_uris(self):
        """延迟加载 Shards 列表（如果 DataModule 没有预扫描）"""
        if self._sub_uris is None:
            self._sub_uris = sorted([
                os.path.join(self.root_dir, d) 
                for d in os.listdir(self.root_dir) 
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
            
            if len(self._sub_uris) == 0:
                raise ValueError(f"❌ 路径 {self.root_dir} 下没有发现子文件夹！")
        
        return self._sub_uris
    
    @property
    def n_vars(self):
        """延迟加载特征维度（只在第一次真正需要时才读取元数据）"""
        if self._n_vars is None:
            tmp_ctx = tiledbsoma.SOMATileDBContext()
            try:
                with tiledbsoma.Experiment.open(self.sub_uris[0], context=tmp_ctx) as exp:
                    if self.latent_key:
                        # Latent 模式：读取 obsm[latent_key] 的维度
                        if self.latent_key in exp.obs.keys():
                            # 如果存储在 obs 中
                            self._n_vars = 1  # Placeholder
                        else:
                            # 假设存储在 obsm 中
                            self._n_vars = 64  # 默认 latent 维度（可以后续从配置读取）
                    else:
                        # Raw 模式：读取基因数量
                        self._n_vars = exp.ms[self.measurement_name].var.count
            except Exception as e:
                # Fallback 到第二个 shard
                if len(self.sub_uris) > 1:
                    with tiledbsoma.Experiment.open(self.sub_uris[1], context=tmp_ctx) as exp:
                        if self.latent_key:
                            self._n_vars = 64
                        else:
                            self._n_vars = exp.ms[self.measurement_name].var.count
                else:
                    raise
        
        return self._n_vars
    
    def __iter__(self):
        # 1. 获取 DDP 和 Worker 信息
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers_per_gpu = 1
        else:
            worker_id = worker_info.id
            num_workers_per_gpu = worker_info.num_workers
        
        # 计算全局 worker ID 和总 worker 数
        global_worker_id = rank * num_workers_per_gpu + worker_id
        total_workers = world_size * num_workers_per_gpu
        
        # 2. 根据智能负载均衡方案获取当前 worker 的 shards
        if self.shard_assignment:
            my_worker_uris = self.shard_assignment.get(str(global_worker_id), [])
        else:
            # Fallback 到简单切片（向后兼容）
            all_shards = sorted(self.sub_uris)
            my_worker_uris = all_shards[global_worker_id::total_workers]
        
        if len(my_worker_uris) == 0:
            return
        
        random.shuffle(my_worker_uris)  # 在 worker 内部打乱 shards 顺序
        
        # 3. 遍历当前 worker 的所有 shards
        ctx = tiledbsoma.SOMATileDBContext()
        
        for uri in my_worker_uris:
            try:
                # 为每个 shard 单独创建 context，避免跨 shard 污染
                with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                    obs = exp.obs.read().concat().to_pandas()
                    
                    # 过滤当前 split_label
                    mask = obs['split_label'] == self.split_label
                    obs_filtered = obs[mask]
                    
                    if len(obs_filtered) == 0:
                        continue
                    
                    # 读取数据矩阵（Raw 或 Latent）
                    if self.latent_key:
                        # Latent 模式：从 obsm 读取
                        # 注意：TileDB-SOMA 的 obsm 结构可能不同，需要根据实际情况调整
                        # 这里假设 latent 存储在 obs 的某个列中
                        if self.latent_key in obs_filtered.columns:
                            X_data = np.stack(obs_filtered[self.latent_key].values)
                        else:
                            # 如果存储在 obsm 中，需要另外读取
                            # 这里简化处理，假设可以直接读取
                            raise ValueError(f"❌ Latent key '{self.latent_key}' not found in obs")
                    else:
                        # Raw 模式：从 ms[RNA].X 读取
                        X_data = exp.ms[self.measurement_name].X[self.measurement_name].read(
                            coords=(obs_filtered.index.tolist(), slice(None))
                        ).tables().concat().to_pandas().to_numpy()
                    
                    # 构建细胞对
                    pairs = self._build_cell_pairs(obs_filtered, X_data)
                    
                    if len(pairs) == 0:
                        continue
                    
                    # Batch 并返回
                    yield from self._batch_and_yield(pairs)
                    
            except Exception as e:
                print(f"⚠️  [Worker {global_worker_id}] 读取 Shard {uri} 失败: {e}")
                continue
            finally:
                gc.collect()
    
    def _build_cell_pairs(self, obs_filtered, X_data):
        """
        根据 obs 中的 next_cell_id/prev_cell_id 构建细胞对。
        
        返回格式: List[Dict] = [
            {
                'x_curr': np.array,
                'x_next': np.array,
                'time_curr': float,
                'time_next': float,
                'cell_type_curr': str,
                'cell_type_next': str,
                'tissue_curr': str,
                'tissue_next': str,
                ...
            }
        ]
        """
        pairs = []
        
        # 创建 obs_id 到 行索引 的映射
        obs_id_to_idx = {obs_id: i for i, obs_id in enumerate(obs_filtered.index)}
        
        for i, row in obs_filtered.iterrows():
            x_curr = X_data[obs_id_to_idx[i]]
            
            # 根据 direction 选择细胞对方向
            if self.direction == "forward" or self.direction == "both":
                next_cell_id = row.get('next_cell_id')
                if next_cell_id is not None and next_cell_id in obs_id_to_idx:
                    next_idx = obs_id_to_idx[next_cell_id]
                    x_next = X_data[next_idx]
                    next_row = obs_filtered.loc[next_cell_id]
                    
                    pairs.append({
                        'x_curr': x_curr,
                        'x_next': x_next,
                        'time_curr': row['time'],
                        'time_next': next_row['time'],
                        'delta_t': next_row['time'] - row['time'],
                        'cell_type_curr': row.get('cell_type', 'Unknown'),
                        'cell_type_next': next_row.get('cell_type', 'Unknown'),
                        'tissue_curr': row.get('tissue', 'Unknown'),
                        'tissue_next': next_row.get('tissue', 'Unknown'),
                    })
            
            if self.direction == "backward" or self.direction == "both":
                prev_cell_id = row.get('prev_cell_id')
                if prev_cell_id is not None and prev_cell_id in obs_id_to_idx:
                    prev_idx = obs_id_to_idx[prev_cell_id]
                    x_prev = X_data[prev_idx]
                    prev_row = obs_filtered.loc[prev_cell_id]
                    
                    # 注意：这里 curr 和 next 的顺序是反的
                    pairs.append({
                        'x_curr': x_prev,
                        'x_next': x_curr,
                        'time_curr': prev_row['time'],
                        'time_next': row['time'],
                        'delta_t': row['time'] - prev_row['time'],
                        'cell_type_curr': prev_row.get('cell_type', 'Unknown'),
                        'cell_type_next': row.get('cell_type', 'Unknown'),
                        'tissue_curr': prev_row.get('tissue', 'Unknown'),
                        'tissue_next': row.get('tissue', 'Unknown'),
                    })
        
        return pairs
    
    def _batch_and_yield(self, pairs):
        """
        将细胞对打包成 batch 并返回。
        """
        n_pairs = len(pairs)
        n_batches = math.ceil(n_pairs / self.batch_size)
        
        # 随机打乱
        random.shuffle(pairs)
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_pairs)
            batch_pairs = pairs[start:end]
            
            # Drop 单样本 batch（避免 BatchNorm 崩溃）
            if len(batch_pairs) <= 1:
                continue
            
            # 转换为 Tensor
            batch = {
                'x_curr': torch.tensor(
                    np.stack([p['x_curr'] for p in batch_pairs]), 
                    dtype=torch.float32
                ),
                'x_next': torch.tensor(
                    np.stack([p['x_next'] for p in batch_pairs]), 
                    dtype=torch.float32
                ),
                'cond_meta': {
                    'time_curr': torch.tensor(
                        [p['time_curr'] for p in batch_pairs], 
                        dtype=torch.float32
                    ),
                    'time_next': torch.tensor(
                        [p['time_next'] for p in batch_pairs], 
                        dtype=torch.float32
                    ),
                    'delta_t': torch.tensor(
                        [p['delta_t'] for p in batch_pairs], 
                        dtype=torch.float32
                    ),
                    # 可以根据需要添加更多条件信息
                    # 'cell_type_curr': [p['cell_type_curr'] for p in batch_pairs],
                    # 'tissue_curr': [p['tissue_curr'] for p in batch_pairs],
                }
            }
            
            yield batch

