import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import math
import os
import random
import gc

class SomaCollectionDataset(IterableDataset):
    def __init__(self, root_dir, split_label=0, io_chunk_size=16384, batch_size=256, measurement_name="RNA", preloaded_sub_uris=None, shard_assignment=None):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self._n_vars = None  # 延迟加载
        self.shard_assignment = shard_assignment  # 智能负载均衡方案
        
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
            # 兼容模式：如果没有预加载，则由 Worker 自己扫描
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
                    self._n_vars = exp.ms[self.measurement_name].var.count
            except Exception:
                if len(self.sub_uris) > 1:
                    with tiledbsoma.Experiment.open(self.sub_uris[1], context=tmp_ctx) as exp:
                        self._n_vars = exp.ms[self.measurement_name].var.count
                else:
                    raise
        
        return self._n_vars

    def _get_context(self):
        return tiledbsoma.SOMATileDBContext(tiledb_config={
            "py.init_buffer_bytes": 512 * 1024**2,
            "sm.memory_budget": 4 * 1024**3,
        })

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
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # 2. 计算全局 Worker ID（跨 GPU）
        global_worker_id = rank * num_workers + worker_id
        
        # 3. 选择分片策略
        if self.shard_assignment is not None:
            # 策略 A：使用智能负载均衡方案（推荐）
            assigned_shard_names = self.shard_assignment.get(str(global_worker_id), [])
            # 将 shard 名称转换为完整路径
            shard_name_to_uri = {os.path.basename(uri): uri for uri in self.sub_uris}
            my_worker_uris = [shard_name_to_uri[name] for name in assigned_shard_names if name in shard_name_to_uri]
        else:
            # 策略 B：简单轮询（向后兼容）
            total_workers = world_size * num_workers
            global_uris = sorted(self.sub_uris)
            my_worker_uris = global_uris[global_worker_id::total_workers]

        if len(my_worker_uris) == 0:
            return

        # 4. 打乱处理顺序（每个 epoch 都不一样）
        random.shuffle(my_worker_uris)
        
        ctx = self._get_context()
        
        # 大块内存池（复用）
        dense_buffer = np.zeros((self.io_chunk_size, self.n_vars), dtype=np.float32)

        try:
            for uri in my_worker_uris:
                try:
                    with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                        try:
                            query = exp.obs.read(
                                value_filter=f"split_label == {self.split_label}",
                                column_names=["soma_joinid"]
                            ).concat()
                            chunk_ids = query["soma_joinid"].to_numpy().copy()
                        except Exception:
                            continue 
                        
                        if len(chunk_ids) == 0: continue
                        np.random.shuffle(chunk_ids)
                        
                        x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")
                        
                        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                            for i in range(0, len(chunk_ids), self.io_chunk_size):
                                sub_ids = chunk_ids[i : i + self.io_chunk_size]
                                current_len = len(sub_ids)
                                read_ids = np.sort(sub_ids)
                                
                                data = X.read(coords=(read_ids, slice(None))).tables().concat()
                                
                                row_indices = data["soma_dim_0"].to_numpy()
                                col_indices = data["soma_dim_1"].to_numpy()
                                values = data["soma_data"].to_numpy()
                                
                                local_rows = np.searchsorted(read_ids, row_indices)
                                
                                # --- 🔥 修复点在这里 🔥 ---
                                # 必须先定义 active_buffer 是 dense_buffer 的一个切片
                                active_buffer = dense_buffer[:current_len]
                                
                                # 然后才能清零和赋值
                                active_buffer.fill(0)
                                active_buffer[local_rows, col_indices] = values
                                
                                perm = np.random.permutation(current_len)
                                num_batches = (current_len + self.batch_size - 1) // self.batch_size
                                
                                for b in range(num_batches):
                                    start_idx = b * self.batch_size
                                    end_idx = min(start_idx + self.batch_size, current_len)
                                    batch_perm_idx = perm[start_idx:end_idx]
                                    
                                    # [CRITICAL FIX] 检查最后一个 batch 是否太小
                                    # 如果太小 (比如 1)，BatchNorm 会崩溃
                                    if len(batch_perm_idx) <= 1:
                                        continue
                                    
                                    out_tensor = torch.from_numpy(active_buffer[batch_perm_idx].copy())
                                    out_labels = torch.zeros(len(out_tensor), dtype=torch.long)
                                    
                                    yield out_tensor, out_labels
                                    
                except Exception as e:
                    print(f"⚠️ Error processing {os.path.basename(uri)}: {e}")
                    continue
                    
        finally:
            del dense_buffer
            del ctx
            gc.collect()
