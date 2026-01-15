"""
时间链一致性数据集 (Consistency Dataset)

核心改进：
1. 支持细胞序列（而非仅细胞对）用于时间链一致性训练
2. 支持最大时间筛选（如 100 天内）
3. 随机采样子序列，支持不同时间跨度的预测

混合一致性 Loss：
- L_flow: 标准 Flow loss
- L_step: 逐步一致性（每步预测准确）
- L_e2e: 端到端一致性（直接预测 ≈ 链式预测）
"""

import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import pandas as pd
import os
import random
import gc
from typing import Optional, Dict, List

from src.data.components.rtf_dataset import (
    normalize_time_vec,
    normalize_delta_t_vec,
    get_stage_map,
)


class SomaConsistencyDataset(IterableDataset):
    """
    支持时间链一致性训练的数据集。

    核心特性：
    1. 构建细胞序列（链）而非仅细胞对
    2. 支持最大时间筛选（max_time_days）
    3. 随机采样子序列用于一致性训练

    输出格式（每个 batch）：
    - x_seq: [B, seq_len, D] 细胞序列
    - time_seq: [B, seq_len] 时间序列
    - seq_mask: [B, seq_len] 有效位置掩码
    - stage: [B] 发育阶段
    """

    def __init__(
        self,
        root_dir: str,
        io_chunk_size: int = 16384,
        batch_size: int = 64,
        measurement_name: str = "RNA",
        latent_key: Optional[str] = None,
        max_time_days: float = 100.0,
        min_seq_len: int = 2,
        max_seq_len: int = 5,
        preloaded_sub_uris: Optional[List[str]] = None,
        shard_assignment: Optional[Dict[str, List[str]]] = None,
        stage_info_path: Optional[str] = None,
        use_log_time: bool = True,
    ):
        """
        Args:
            root_dir: TileDB SOMA 根目录
            max_time_days: 最大时间筛选（天），只保留该时间内的细胞
            min_seq_len: 最小序列长度
            max_seq_len: 最大序列长度
        """
        self.root_dir = root_dir
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self.latent_key = latent_key
        self.max_time_days = max_time_days
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self._n_vars = None
        self.shard_assignment = shard_assignment
        self.use_log_time = use_log_time

        self.stage_map = get_stage_map(stage_info_path)

        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None

        if not os.path.exists(root_dir):
            raise ValueError(f"路径不存在: {root_dir}")

    @property
    def sub_uris(self):
        if self._sub_uris is None:
            self._sub_uris = sorted([
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
        return self._sub_uris

    @property
    def n_vars(self):
        if self._n_vars is None:
            tmp_ctx = tiledbsoma.SOMATileDBContext()
            with tiledbsoma.Experiment.open(self.sub_uris[0], context=tmp_ctx) as exp:
                self._n_vars = exp.ms[self.measurement_name].var.count
        return self._n_vars

    def _get_context(self):
        return tiledbsoma.SOMATileDBContext(tiledb_config={
            "py.init_buffer_bytes": 1024 * 1024**2,  # 1GB buffer
            "sm.memory_budget": 8 * 1024**3,  # 8GB memory budget
            "sm.compute_concurrency_level": 8,  # 并行计算线程
            "sm.io_concurrency_level": 8,  # 并行IO线程
        })

    def __iter__(self):
        # 获取 DDP 和 Worker 信息
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id

        # 分配 shards
        if self.shard_assignment is not None:
            assigned = self.shard_assignment.get(str(global_worker_id), [])
            name_to_uri = {os.path.basename(u): u for u in self.sub_uris}
            my_uris = [name_to_uri[n] for n in assigned if n in name_to_uri]
        else:
            total = world_size * num_workers
            my_uris = sorted(self.sub_uris)[global_worker_id::total]

        if not my_uris:
            return

        random.shuffle(my_uris)
        ctx = self._get_context()
        n_vars = self.n_vars
        buffer = np.zeros((self.io_chunk_size, n_vars), dtype=np.float32)

        try:
            for uri in my_uris:
                try:
                    yield from self._process_shard(uri, ctx, buffer, global_worker_id)
                except Exception as e:
                    print(f"[Worker {global_worker_id}] Shard {os.path.basename(uri)} 失败: {e}")
        finally:
            del buffer
            gc.collect()

    def _process_shard(self, uri: str, ctx, buffer: np.ndarray, worker_id: int):
        """处理单个 shard，构建细胞链"""
        shard_name = os.path.basename(uri)
        shard_stage_id = self.stage_map.get(shard_name, 0)

        with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
            # 读取必要的 obs 列
            required_cols = ["soma_joinid", "time", "next_cell_id"]
            try:
                available_cols = [f.name for f in exp.obs.schema]
            except Exception:
                return

            actual_cols = [c for c in required_cols if c in available_cols]
            if "soma_joinid" not in actual_cols:
                return

            # 找到 cell_id 列
            cell_id_col = None
            for col in ["obs_id", "new_index"]:
                if col in available_cols:
                    cell_id_col = col
                    actual_cols.append(col)
                    break
            if cell_id_col is None:
                return

            # 读取 obs 数据
            try:
                obs_table = exp.obs.read(column_names=actual_cols).concat()
                obs_df = obs_table.to_pandas()
            except Exception:
                return

            if len(obs_df) == 0:
                return

            # 构建索引映射
            cell_ids = obs_df[cell_id_col].astype(str).values
            soma_joinids = obs_df["soma_joinid"].values
            times = obs_df["time"].values.astype(np.float64)
            times = np.nan_to_num(times, nan=0.0)

            cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

            # 筛选：时间在 max_time_days 内
            valid_mask = times <= self.max_time_days
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                return

            # 构建细胞链
            chains = self._build_chains(
                obs_df, valid_indices, cell_to_idx, times,
                "next_cell_id" if "next_cell_id" in obs_df.columns else None
            )

            if not chains:
                return

            # 读取表达数据并生成 batch
            x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")
            yield from self._yield_chain_batches(
                chains, soma_joinids, times, x_uri, ctx, buffer, shard_stage_id
            )

    def _build_chains(
        self,
        obs_df: pd.DataFrame,
        valid_indices: np.ndarray,
        cell_to_idx: Dict[str, int],
        times: np.ndarray,
        next_col: Optional[str],
    ) -> List[List[int]]:
        """
        构建细胞链。

        从每个有效细胞出发，沿着 next_cell_id 构建链，
        直到链断开或超出时间范围。
        """
        if next_col is None:
            return []

        next_ids = obs_df[next_col].values
        visited = set()
        chains = []

        for start_idx in valid_indices:
            if start_idx in visited:
                continue

            # 构建从 start_idx 开始的链
            chain = [start_idx]
            visited.add(start_idx)
            curr_idx = start_idx

            while True:
                next_id = next_ids[curr_idx]
                if next_id is None or pd.isna(next_id):
                    break

                next_id_str = str(next_id)
                if next_id_str not in cell_to_idx:
                    break

                next_idx = cell_to_idx[next_id_str]

                # 检查时间是否在范围内
                if times[next_idx] > self.max_time_days:
                    break

                # 检查是否已访问（避免循环）
                if next_idx in visited:
                    break

                chain.append(next_idx)
                visited.add(next_idx)
                curr_idx = next_idx

            # 只保留足够长的链
            if len(chain) >= self.min_seq_len:
                chains.append(chain)

        return chains

    def _yield_chain_batches(
        self,
        chains: List[List[int]],
        soma_joinids: np.ndarray,
        times: np.ndarray,
        x_uri: str,
        ctx,
        buffer: np.ndarray,
        shard_stage_id: int,
    ):
        """从细胞链生成训练 batch"""
        # 收集所有需要的细胞索引
        all_indices = set()
        for chain in chains:
            all_indices.update(chain)
        all_indices = np.array(sorted(all_indices))

        if len(all_indices) == 0:
            return

        # 读取表达数据
        read_joinids = soma_joinids[all_indices]
        read_joinids_sorted = np.sort(read_joinids)

        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
            try:
                data = X.read(coords=(read_joinids_sorted, slice(None))).tables().concat()
            except Exception:
                return

            row_indices = data["soma_dim_0"].to_numpy()
            col_indices = data["soma_dim_1"].to_numpy()
            values = data["soma_data"].to_numpy()

            # 构建映射
            joinid_to_local = {jid: i for i, jid in enumerate(read_joinids_sorted)}
            n_cells = len(read_joinids_sorted)
            n_vars = buffer.shape[1]

            # 动态处理超过 buffer 大小的情况
            if n_cells <= buffer.shape[0]:
                # 正常情况：使用预分配的 buffer
                cell_buffer = buffer[:n_cells]
                cell_buffer.fill(0)
            else:
                # 超过 buffer 大小：动态分配临时 buffer
                cell_buffer = np.zeros((n_cells, n_vars), dtype=np.float32)

            local_rows = np.array([joinid_to_local[jid] for jid in row_indices])
            cell_buffer[local_rows, col_indices] = values

            # obs_idx -> local_row
            obs_to_local = {
                idx: joinid_to_local[soma_joinids[idx]]
                for idx in all_indices
            }

            # 生成 batch
            yield from self._create_batches(
                chains, times, cell_buffer, obs_to_local, shard_stage_id
            )

            # 释放动态分配的内存
            if n_cells > buffer.shape[0]:
                del cell_buffer
                gc.collect()

    def _create_batches(
        self,
        chains: List[List[int]],
        times: np.ndarray,
        cell_buffer: np.ndarray,
        obs_to_local: Dict[int, int],
        shard_stage_id: int,
    ):
        """
        从细胞链创建训练 batch。

        每个 batch 包含：
        - x_seq: [B, max_seq_len, D] 细胞序列
        - time_seq: [B, max_seq_len] 时间序列（归一化）
        - seq_len: [B] 每个序列的实际长度
        - stage: [B] 发育阶段
        """
        # 预先归一化所有时间（向量化）
        normalized_times = normalize_time_vec(times, self.use_log_time)

        # 随机打乱链
        random.shuffle(chains)

        # 从链中采样子序列
        sequences = []
        for chain in chains:
            # 随机选择子序列长度
            max_len = min(len(chain), self.max_seq_len)
            seq_len = random.randint(self.min_seq_len, max_len)

            # 随机选择起始位置
            start = random.randint(0, len(chain) - seq_len)
            sub_chain = chain[start:start + seq_len]
            sequences.append(sub_chain)

        # 按 batch_size 分组
        n_vars = cell_buffer.shape[1]

        for b_start in range(0, len(sequences), self.batch_size):
            b_end = min(b_start + self.batch_size, len(sequences))
            batch_seqs = sequences[b_start:b_end]
            batch_size = len(batch_seqs)

            if batch_size < 2:
                continue

            # 找到这个 batch 的最大序列长度
            max_len = max(len(s) for s in batch_seqs)

            # 初始化 batch tensors
            x_seq = np.zeros((batch_size, max_len, n_vars), dtype=np.float32)
            time_seq = np.zeros((batch_size, max_len), dtype=np.float32)
            seq_lens = np.zeros(batch_size, dtype=np.int64)

            # 向量化填充数据
            for i, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                seq_lens[i] = seq_len

                # 批量获取 local indices
                local_indices = [obs_to_local[obs_idx] for obs_idx in seq]

                # 向量化复制表达数据
                x_seq[i, :seq_len] = cell_buffer[local_indices]

                # 向量化获取时间
                time_seq[i, :seq_len] = normalized_times[seq]

            yield {
                'x_seq': torch.from_numpy(x_seq),
                'time_seq': torch.from_numpy(time_seq),
                'seq_len': torch.from_numpy(seq_lens),
                'stage': torch.from_numpy(np.full(batch_size, shard_stage_id, dtype=np.int64)),
            }
