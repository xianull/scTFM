"""
scVI-style Dataset for Single-Cell Data (Fast Version)

使用 normalized counts (expm1) 代替 raw counts:
- x: log1p(normalized) - Encoder 输入
- counts: expm1(x) = normalized counts - NB Loss 目标
- library_size: counts.sum() - Decoder 缩放

这种方式只需要读取一个矩阵，与 ae_dataset.py 相同的 I/O 效率。

数学上:
- raw_counts ≈ normalized_counts * (library_size / target_sum)
- 使用 normalized counts 做 NB loss 是合理的（很多 scVI 变体这样做）
"""

import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import os
import random
import gc


class SomaSCVIDataset(IterableDataset):
    """
    scVI-style Dataset (Fast Version)

    只读取一个矩阵，使用 expm1 计算 normalized counts。

    Yields:
        dict with keys:
            - 'x': log1p(normalized) tensor, shape (batch, n_genes) - Encoder 输入
            - 'counts': normalized counts tensor, shape (batch, n_genes) - NB Loss 目标
            - 'library_size': library size tensor, shape (batch,) - Decoder 缩放
    """

    def __init__(
        self,
        root_dir: str,
        split_label: int = 0,
        io_chunk_size: int = 16384,
        batch_size: int = 256,
        measurement_name: str = "RNA",
        preloaded_sub_uris: list = None,
        shard_assignment: dict = None,
        use_counts_layer: bool = False,  # 保留参数（兼容性），但不使用
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self._n_vars = None
        self.shard_assignment = shard_assignment

        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None

        if not os.path.exists(root_dir):
            raise ValueError(f"Path does not exist: {root_dir}")

    @property
    def sub_uris(self):
        """延迟加载 Shards 列表"""
        if self._sub_uris is None:
            self._sub_uris = sorted([
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])

            if len(self._sub_uris) == 0:
                raise ValueError(f"No subdirectories found in {self.root_dir}")

        return self._sub_uris

    @property
    def n_vars(self):
        """延迟加载特征维度"""
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

        global_worker_id = rank * num_workers + worker_id

        # 2. 选择分片策略
        if self.shard_assignment is not None:
            assigned_shard_names = self.shard_assignment.get(str(global_worker_id), [])
            shard_name_to_uri = {os.path.basename(uri): uri for uri in self.sub_uris}
            my_worker_uris = [shard_name_to_uri[name] for name in assigned_shard_names if name in shard_name_to_uri]
        else:
            total_workers = world_size * num_workers
            global_uris = sorted(self.sub_uris)
            my_worker_uris = global_uris[global_worker_id::total_workers]

        if len(my_worker_uris) == 0:
            return

        random.shuffle(my_worker_uris)

        ctx = self._get_context()

        # 内存池
        dense_buffer = np.zeros((self.io_chunk_size, self.n_vars), dtype=np.float32)

        try:
            for uri in my_worker_uris:
                try:
                    with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                        # 读取 soma_joinid
                        try:
                            query = exp.obs.read(
                                value_filter=f"split_label == {self.split_label}",
                                column_names=["soma_joinid"]
                            ).concat()
                            chunk_ids = query["soma_joinid"].to_numpy().copy()
                        except Exception:
                            continue

                        if len(chunk_ids) == 0:
                            continue

                        np.random.shuffle(chunk_ids)

                        x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")

                        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                            for i in range(0, len(chunk_ids), self.io_chunk_size):
                                sub_ids = chunk_ids[i: i + self.io_chunk_size]
                                current_len = len(sub_ids)
                                read_ids = np.sort(sub_ids)

                                # 读取 X (log1p normalized)
                                data = X.read(coords=(read_ids, slice(None))).tables().concat()
                                row_indices = data["soma_dim_0"].to_numpy()
                                col_indices = data["soma_dim_1"].to_numpy()
                                values = data["soma_data"].to_numpy()
                                local_rows = np.searchsorted(read_ids, row_indices)

                                active_buffer = dense_buffer[:current_len]
                                active_buffer.fill(0)
                                active_buffer[local_rows, col_indices] = values

                                # Batch 划分
                                perm = np.random.permutation(current_len)
                                num_batches = (current_len + self.batch_size - 1) // self.batch_size

                                for b in range(num_batches):
                                    start_idx = b * self.batch_size
                                    end_idx = min(start_idx + self.batch_size, current_len)
                                    batch_idx = perm[start_idx:end_idx]

                                    if len(batch_idx) <= 1:
                                        continue

                                    # x: log1p(normalized)
                                    x_batch = active_buffer[batch_idx].copy()

                                    # counts: expm1(x) = normalized counts
                                    counts_batch = np.expm1(x_batch)

                                    # library_size: sum of normalized counts per cell
                                    library_batch = counts_batch.sum(axis=1)
                                    library_batch = np.maximum(library_batch, 1.0)

                                    yield {
                                        'x': torch.from_numpy(x_batch),
                                        'counts': torch.from_numpy(counts_batch),
                                        'library_size': torch.from_numpy(library_batch.astype(np.float32)),
                                    }

                except Exception as e:
                    print(f"Warning: Error processing {os.path.basename(uri)}: {e}")
                    continue

        finally:
            del dense_buffer
            del ctx
            gc.collect()
