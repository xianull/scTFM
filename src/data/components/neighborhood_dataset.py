"""
Neighborhood Dataset for Set-based Single-Cell Autoencoder

微环境采样策略：
- 一个 tiledb (tissue) = 一个微环境池
- 随机采样 bag_size 个细胞作为一个微环境 bag
- 支持多种训练方案：Graph AE, Contrastive AE, Masked AE

数据格式：
- 每个 batch 返回 (batch_size, bag_size, n_genes) 的张量
- bag 内的细胞来自同一个 tissue，构成微环境
"""

import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import os
import random
import gc
from typing import Optional, Dict, List


class SomaNeighborhoodDataset(IterableDataset):
    """
    微环境采样数据集

    每次迭代返回一个 batch 的 neighborhood bags:
    - 每个 bag 包含 bag_size 个来自同一 tissue 的细胞
    - 这些细胞构成一个微环境

    Args:
        root_dir: tiledb 数据根目录
        split_label: 数据集划分标签 (0=train, 1=val, 2=test)
        bag_size: 每个微环境 bag 中的细胞数量
        batch_size: 每个 batch 中的 bag 数量
        io_chunk_size: I/O 读取块大小
        measurement_name: SOMA measurement 名称
        mask_ratio: Masked AE 的掩码比例 (0 表示不使用)
        mask_strategy: 掩码策略 ("random" | "hvg")
    """

    def __init__(
        self,
        root_dir: str,
        split_label: int = 0,
        bag_size: int = 16,
        batch_size: int = 64,
        io_chunk_size: int = 16384,
        measurement_name: str = "RNA",
        preloaded_sub_uris: list = None,
        shard_assignment: dict = None,
        # Masked AE 参数
        mask_ratio: float = 0.0,
        mask_strategy: str = "random",
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.io_chunk_size = io_chunk_size
        self.measurement_name = measurement_name
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.shard_assignment = shard_assignment

        self._n_vars = None

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

    def _create_mask(self, n_genes: int) -> np.ndarray:
        """创建基因掩码"""
        if self.mask_ratio <= 0:
            return None

        mask = np.zeros(n_genes, dtype=np.float32)
        n_mask = int(n_genes * self.mask_ratio)

        if self.mask_strategy == "random":
            mask_indices = np.random.choice(n_genes, n_mask, replace=False)
        else:
            # hvg 策略：可以后续扩展，目前用随机
            mask_indices = np.random.choice(n_genes, n_mask, replace=False)

        mask[mask_indices] = 1.0
        return mask

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
                yield from self._process_tissue(uri, ctx, dense_buffer)
        finally:
            del dense_buffer
            del ctx
            gc.collect()

    def _process_tissue(self, uri: str, ctx, dense_buffer: np.ndarray):
        """处理单个 tissue，生成 neighborhood bags"""
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
                    return

                if len(chunk_ids) < self.bag_size:
                    # tissue 细胞数不足，跳过
                    return

                np.random.shuffle(chunk_ids)

                x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")

                with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                    for i in range(0, len(chunk_ids), self.io_chunk_size):
                        sub_ids = chunk_ids[i: i + self.io_chunk_size]
                        current_len = len(sub_ids)

                        if current_len < self.bag_size:
                            continue

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

                        # 生成 neighborhood bags
                        yield from self._generate_bags(active_buffer, current_len)

        except Exception as e:
            print(f"Warning: Error processing {os.path.basename(uri)}: {e}")
            return

    def _generate_bags(self, buffer: np.ndarray, n_cells: int):
        """从 buffer 中生成 neighborhood bags"""
        # 计算可以生成多少个 bag
        n_bags = n_cells // self.bag_size

        if n_bags == 0:
            return

        # 随机打乱
        perm = np.random.permutation(n_cells)

        # 收集 bags 直到达到 batch_size
        bag_buffer = []

        for b in range(n_bags):
            start_idx = b * self.bag_size
            end_idx = start_idx + self.bag_size
            bag_indices = perm[start_idx:end_idx]

            # x: log1p(normalized), shape (bag_size, n_genes)
            x_bag = buffer[bag_indices].copy()

            # counts: expm1(x) = normalized counts
            counts_bag = np.expm1(x_bag)

            # library_size: sum per cell
            library_bag = counts_bag.sum(axis=1)
            library_bag = np.maximum(library_bag, 1.0)

            bag_data = {
                'x': x_bag,
                'counts': counts_bag,
                'library_size': library_bag.astype(np.float32),
            }

            # 添加掩码（如果需要）
            if self.mask_ratio > 0:
                bag_data['mask'] = self._create_mask(self.n_vars)

            bag_buffer.append(bag_data)

            # 达到 batch_size，yield 一个 batch
            if len(bag_buffer) >= self.batch_size:
                yield self._collate_bags(bag_buffer)
                bag_buffer = []

        # 处理剩余的 bags
        if len(bag_buffer) > 0:
            yield self._collate_bags(bag_buffer)

    def _collate_bags(self, bags: List[Dict]) -> Dict[str, torch.Tensor]:
        """将多个 bag 合并成一个 batch"""
        batch_size = len(bags)

        # Stack all bags: (batch_size, bag_size, n_genes)
        x = np.stack([b['x'] for b in bags], axis=0)
        counts = np.stack([b['counts'] for b in bags], axis=0)
        library_size = np.stack([b['library_size'] for b in bags], axis=0)

        result = {
            'x': torch.from_numpy(x),
            'counts': torch.from_numpy(counts),
            'library_size': torch.from_numpy(library_size),
        }

        # 添加掩码
        if 'mask' in bags[0] and bags[0]['mask'] is not None:
            masks = np.stack([b['mask'] for b in bags], axis=0)
            result['mask'] = torch.from_numpy(masks)

        return result
