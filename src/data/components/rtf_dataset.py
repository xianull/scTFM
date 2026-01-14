import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import pandas as pd
import math
import os
import random
import gc
from typing import Optional, Dict, List

# ============================================================================
# 时间编码常量和函数
# ============================================================================
MAX_TIME_DAYS = 36500.0  # 100 岁，作为归一化上界
MAX_DELTA_DAYS = 36500.0  # delta_t 也用相同范围

def normalize_time(time_days: float, use_log: bool = True) -> float:
    """将绝对时间（天）归一化到 [0, 1] 范围。"""
    if use_log:
        return math.log(1.0 + time_days) / math.log(1.0 + MAX_TIME_DAYS)
    else:
        return min(time_days / MAX_TIME_DAYS, 1.0)


def normalize_delta_t(delta_days: float, use_log: bool = True) -> float:
    """将时间差（天）归一化到 [-1, 1] 范围。"""
    if use_log:
        sign = 1.0 if delta_days >= 0 else -1.0
        abs_delta = abs(delta_days)
        return sign * math.log(1.0 + abs_delta) / math.log(1.0 + MAX_DELTA_DAYS)
    else:
        return max(min(delta_days / MAX_DELTA_DAYS, 1.0), -1.0)


# 向量化版本（用于批量处理）
def normalize_time_vec(time_days: np.ndarray, use_log: bool = True) -> np.ndarray:
    """向量化时间归一化"""
    if use_log:
        return np.log1p(time_days) / np.log1p(MAX_TIME_DAYS)
    else:
        return np.minimum(time_days / MAX_TIME_DAYS, 1.0)


def normalize_delta_t_vec(delta_days: np.ndarray, use_log: bool = True) -> np.ndarray:
    """向量化 delta_t 归一化"""
    if use_log:
        sign = np.sign(delta_days)
        sign[sign == 0] = 1.0  # 0 视为正
        abs_delta = np.abs(delta_days)
        return sign * np.log1p(abs_delta) / np.log1p(MAX_DELTA_DAYS)
    else:
        return np.clip(delta_days / MAX_DELTA_DAYS, -1.0, 1.0)


# ============================================================================
# 全局类别映射
# ============================================================================
STAGE_CATEGORIES = [
    "Unknown", "Embryonic", "Fetal", "Newborn", "Paediatric", "Adult",
]

STAGE_TO_ID = {name: idx for idx, name in enumerate(STAGE_CATEGORIES)}
STAGE_TO_ID.update({
    "unknown": 0, "embryonic": 1, "fetal": 2, "newborn": 3, "neonatal": 3,
    "paediatric": 4, "pediatric": 4, "child": 4, "adult": 5,
})

def encode_stage(stage_name: str) -> int:
    """将 Stage 名称编码为整数 ID"""
    if stage_name is None:
        return 0
    stage_str = str(stage_name).strip()
    if stage_str in STAGE_TO_ID:
        return STAGE_TO_ID[stage_str]
    stage_lower = stage_str.lower()
    if stage_lower in STAGE_TO_ID:
        return STAGE_TO_ID[stage_lower]
    return 0


TISSUE_CATEGORIES = [
    "Unknown", "Bone Marrow", "Epiblast", "Eye", "Intestine", "Kidney",
    "Placenta", "Primitive Endoderm", "Prostate", "Thymus", "Trophectoderm",
]

CELLTYPE_CATEGORIES = [
    "Unknown", "AFP_ALB Positive Cell", "Amacrine Cell", "Antigen Presenting Cell",
    "Astrocyte", "B Cell", "Bipolar Cell", "Corneal and Conjunctival Epithelial Cell",
    "Dendritic Cell", "Endothelial Cell", "Endothelial Cell (Lymphatic)",
    "Endothelial Cell (Vascular)", "Eosinophil/Basophil/Mast Cell", "Epiblast",
    "Epithelial Cell", "Epithelial Cell (Basal)", "Epithelial Cell (Club)",
    "Epithelial Cell (Hillock)", "Epithelial Cell (Luminal)", "Erythroid Cell",
    "Erythroid Progenitor Cell", "Fibroblast", "Ganglion Cell",
    "Hematopoietic Stem Cell", "Horizontal Cell", "IGFBP1_DKK1 Positive Cell",
    "Lens Fibre Cell", "Lymphoid Cell", "Macrophage", "Mast Cell", "Megakaryocyte",
    "Mesangial Cell", "Mesodermal Killer Cell", "Mesothelial Cell", "Metanephric Cell",
    "Microglia", "Monocyte", "Myeloid Cell", "Myocyte (Skeletal Muscle)",
    "Myocyte (Smooth Muscle)", "Natural Killer Cell", "Natural Killer T Cell",
    "Neuroendocrine Cell", "Neuron", "Neutrophil", "PAEP_MECOM Positive Cell",
    "PDE11A_FAM19A2 Positive Cell", "Pericyte", "Pericyte/Myocyte (Smooth Muscle)",
    "Photoreceptor Cell", "Primitive Endoderm", "Proliferating T Cell",
    "Retinal Pigment Cell", "Retinal Progenitor and Muller Glia", "Schwann Cell",
    "Stroma", "Syncytiotrophoblasts and Villous Cytotrophoblasts", "T Cell",
    "Trophectoderm", "Trophoblast Giant Cell", "Ureteric Bud Cell",
]

TISSUE_TO_ID = {name: idx for idx, name in enumerate(TISSUE_CATEGORIES)}
CELLTYPE_TO_ID = {name: idx for idx, name in enumerate(CELLTYPE_CATEGORIES)}

def encode_tissue(tissue_name: str) -> int:
    return TISSUE_TO_ID.get(str(tissue_name), 0)

def encode_celltype(celltype_name: str) -> int:
    return CELLTYPE_TO_ID.get(str(celltype_name), 0)


def load_stage_mapping(csv_path: str) -> Dict[str, int]:
    """从 tedd_info.csv 加载 Stage 映射。"""
    if not os.path.exists(csv_path):
        print(f"⚠️ Stage info CSV not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        stage_map = {}

        for _, row in df.iterrows():
            original_id = str(row.get('ID', ''))
            stage_raw = row.get('Stage', 'Unknown')

            if not original_id or pd.isna(original_id):
                continue

            normalized_id = original_id.replace('.', '_').replace('-', '_')

            if pd.isna(stage_raw):
                stage = 'Unknown'
            else:
                stage = str(stage_raw).split(',')[0].strip()

            stage_id = encode_stage(stage)
            stage_map[normalized_id] = stage_id
            stage_map[original_id] = stage_id

        print(f"✅ Loaded Stage mapping for {len(stage_map)} entries")
        return stage_map

    except Exception as e:
        print(f"⚠️ Failed to load Stage mapping: {e}")
        return {}


DEFAULT_STAGE_INFO_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/tedd_info.csv"
_STAGE_MAP_CACHE: Optional[Dict[str, int]] = None


def get_stage_map(csv_path: Optional[str] = None) -> Dict[str, int]:
    """获取 Stage 映射（带缓存）"""
    global _STAGE_MAP_CACHE
    if _STAGE_MAP_CACHE is None:
        path = csv_path or DEFAULT_STAGE_INFO_PATH
        _STAGE_MAP_CACHE = load_stage_mapping(path)
    return _STAGE_MAP_CACHE


class SomaRTFDataset(IterableDataset):
    """
    用于 Rectified Flow 训练的 TileDB-SOMA 数据集（优化版本）。

    核心优化：
    1. 只读取必要的 obs 列，避免读取所有元数据
    2. 使用 numpy 向量化操作，避免 Python iterrows
    3. 分 chunk 读取，避免一次性加载整个 shard
    4. 预分配内存缓冲区复用
    """

    def __init__(
        self,
        root_dir: str,
        split_label: int = 0,
        io_chunk_size: int = 16384,
        batch_size: int = 256,
        measurement_name: str = "RNA",
        latent_key: Optional[str] = None,
        direction: str = "forward",
        preloaded_sub_uris: Optional[List[str]] = None,
        shard_assignment: Optional[Dict[str, List[str]]] = None,
        stage_info_path: Optional[str] = None,
        use_log_time: bool = True,
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self.latent_key = latent_key
        self.direction = direction
        self._n_vars = None
        self.shard_assignment = shard_assignment
        self.use_log_time = use_log_time

        self.stage_map = get_stage_map(stage_info_path)

        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None

        if not os.path.exists(root_dir):
            raise ValueError(f"❌ 路径不存在: {root_dir}")

    @property
    def sub_uris(self):
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
        """返回优化后的 TileDB Context"""
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

        # 3. 预分配内存缓冲区
        n_vars = self.n_vars
        # 需要同时存储 curr 和 next 细胞，所以 *2
        dense_buffer = np.zeros((self.io_chunk_size * 2, n_vars), dtype=np.float32)

        try:
            for uri in my_worker_uris:
                try:
                    yield from self._process_shard_fast(uri, ctx, dense_buffer, global_worker_id)
                except Exception as e:
                    print(f"⚠️ [Worker {global_worker_id}] 读取 Shard {os.path.basename(uri)} 失败: {e}")
                    continue
        finally:
            del dense_buffer
            gc.collect()

    def _process_shard_fast(self, uri: str, ctx, dense_buffer: np.ndarray, worker_id: int):
        """处理单个 shard（优化版本）

        核心优化：
        1. 只读取必要的 obs 列
        2. 使用 numpy 向量化构建 pair
        3. 分 chunk 读取数据
        """
        shard_name = os.path.basename(uri)
        shard_stage_id = self.stage_map.get(shard_name, 0)

        with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
            # 1. 只读取必要的 obs 列（关键优化！）
            required_cols = ["soma_joinid", "split_label", "time"]

            # 根据 direction 决定需要哪些列
            if self.direction in ("forward", "both"):
                required_cols.append("next_cell_id")
            if self.direction in ("backward", "both"):
                required_cols.append("prev_cell_id")

            # 检查可用的列（使用 schema）
            try:
                available_cols = [field.name for field in exp.obs.schema]
            except Exception:
                return

            # 过滤出实际存在的列
            actual_cols = [c for c in required_cols if c in available_cols]
            if "soma_joinid" not in actual_cols:
                return

            # 检查 cell_id 列（用于匹配 next_cell_id）
            cell_id_col = None
            for col in ["obs_id", "new_index"]:
                if col in available_cols:
                    cell_id_col = col
                    actual_cols.append(col)
                    break

            if cell_id_col is None:
                return

            # 2. 读取 obs 数据（只读必要列）
            try:
                obs_table = exp.obs.read(column_names=actual_cols).concat()
                obs_df = obs_table.to_pandas()
            except Exception:
                return

            if len(obs_df) == 0:
                return

            # 3. 构建 cell_id -> soma_joinid 的映射（向量化）
            cell_ids = obs_df[cell_id_col].astype(str).values
            soma_joinids = obs_df["soma_joinid"].values
            split_labels = obs_df["split_label"].values
            times = obs_df["time"].values.astype(np.float64)
            times = np.nan_to_num(times, nan=0.0)

            # 创建快速查找字典
            cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

            # 4. 筛选当前 split 的细胞（向量化）
            curr_mask = split_labels == self.split_label
            curr_indices = np.where(curr_mask)[0]

            if len(curr_indices) == 0:
                return

            # 5. 构建细胞对（向量化）
            pairs_curr_idx = []  # 索引到 obs_df
            pairs_next_idx = []

            if self.direction in ("forward", "both") and "next_cell_id" in obs_df.columns:
                next_cell_ids = obs_df["next_cell_id"].values
                for i in curr_indices:
                    next_id = next_cell_ids[i]
                    if next_id is not None and not pd.isna(next_id):
                        next_id_str = str(next_id)
                        if next_id_str in cell_to_idx:
                            pairs_curr_idx.append(i)
                            pairs_next_idx.append(cell_to_idx[next_id_str])

            if self.direction in ("backward", "both") and "prev_cell_id" in obs_df.columns:
                prev_cell_ids = obs_df["prev_cell_id"].values
                for i in curr_indices:
                    prev_id = prev_cell_ids[i]
                    if prev_id is not None and not pd.isna(prev_id):
                        prev_id_str = str(prev_id)
                        if prev_id_str in cell_to_idx:
                            # backward: prev -> curr
                            pairs_curr_idx.append(cell_to_idx[prev_id_str])
                            pairs_next_idx.append(i)

            if len(pairs_curr_idx) == 0:
                return

            pairs_curr_idx = np.array(pairs_curr_idx, dtype=np.int64)
            pairs_next_idx = np.array(pairs_next_idx, dtype=np.int64)

            # 6. 随机打乱 pairs
            perm = np.random.permutation(len(pairs_curr_idx))
            pairs_curr_idx = pairs_curr_idx[perm]
            pairs_next_idx = pairs_next_idx[perm]

            # 7. 分 chunk 处理 pairs
            x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")
            n_vars = dense_buffer.shape[1]
            n_pairs = len(pairs_curr_idx)

            # 每个 chunk 处理的 pair 数量（考虑到需要读 curr 和 next）
            pairs_per_chunk = self.io_chunk_size // 2

            with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                for chunk_start in range(0, n_pairs, pairs_per_chunk):
                    chunk_end = min(chunk_start + pairs_per_chunk, n_pairs)
                    chunk_curr_idx = pairs_curr_idx[chunk_start:chunk_end]
                    chunk_next_idx = pairs_next_idx[chunk_start:chunk_end]

                    # 收集需要读取的所有 soma_joinid（去重）
                    all_idx = np.unique(np.concatenate([chunk_curr_idx, chunk_next_idx]))
                    read_joinids = soma_joinids[all_idx]
                    read_joinids_sorted = np.sort(read_joinids)

                    # 读取 X 数据
                    try:
                        data = X.read(coords=(read_joinids_sorted, slice(None))).tables().concat()
                    except Exception:
                        continue

                    row_indices = data["soma_dim_0"].to_numpy()
                    col_indices = data["soma_dim_1"].to_numpy()
                    values = data["soma_data"].to_numpy()

                    # 构建 soma_joinid -> local_row 映射
                    joinid_to_local = {jid: i for i, jid in enumerate(read_joinids_sorted)}

                    # 填充 dense buffer
                    n_cells = len(read_joinids_sorted)
                    cell_buffer = dense_buffer[:n_cells]
                    cell_buffer.fill(0)

                    local_rows = np.array([joinid_to_local[jid] for jid in row_indices])
                    cell_buffer[local_rows, col_indices] = values

                    # 构建 obs_idx -> local_row 映射
                    obs_idx_to_local = {
                        idx: joinid_to_local[soma_joinids[idx]]
                        for idx in all_idx
                    }

                    # 生成 batches
                    yield from self._yield_batches_fast(
                        chunk_curr_idx, chunk_next_idx,
                        times, cell_buffer, obs_idx_to_local,
                        shard_stage_id
                    )

    def _yield_batches_fast(
        self,
        curr_indices: np.ndarray,
        next_indices: np.ndarray,
        times: np.ndarray,
        cell_buffer: np.ndarray,
        obs_idx_to_local: Dict[int, int],
        shard_stage_id: int,
    ):
        """生成训练批次（向量化版本）"""
        n_pairs = len(curr_indices)

        for b_start in range(0, n_pairs, self.batch_size):
            b_end = min(b_start + self.batch_size, n_pairs)

            if b_end - b_start <= 1:
                continue

            batch_curr_idx = curr_indices[b_start:b_end]
            batch_next_idx = next_indices[b_start:b_end]
            batch_size = len(batch_curr_idx)

            # 获取表达数据（向量化）
            curr_local = np.array([obs_idx_to_local[i] for i in batch_curr_idx])
            next_local = np.array([obs_idx_to_local[i] for i in batch_next_idx])

            x_curr = cell_buffer[curr_local].copy()
            x_next = cell_buffer[next_local].copy()

            # 获取时间（向量化）
            time_curr_raw = times[batch_curr_idx]
            time_next_raw = times[batch_next_idx]
            delta_t_raw = time_next_raw - time_curr_raw

            # 归一化时间（向量化）
            time_curr_norm = normalize_time_vec(time_curr_raw, self.use_log_time)
            time_next_norm = normalize_time_vec(time_next_raw, self.use_log_time)
            delta_t_norm = normalize_delta_t_vec(delta_t_raw, self.use_log_time)

            # Stage（所有细胞使用 shard 级别的 stage）
            stage_ids = np.full(batch_size, shard_stage_id, dtype=np.int64)

            # Tissue 和 Celltype 暂时用 0（可以后续扩展）
            tissue_ids = np.zeros(batch_size, dtype=np.int64)
            celltype_ids = np.zeros(batch_size, dtype=np.int64)

            batch = {
                'x_curr': torch.from_numpy(x_curr),
                'x_next': torch.from_numpy(x_next),
                'cond_meta': {
                    'time_curr': torch.from_numpy(time_curr_norm.astype(np.float32)),
                    'time_next': torch.from_numpy(time_next_norm.astype(np.float32)),
                    'delta_t': torch.from_numpy(delta_t_norm.astype(np.float32)),
                    'tissue': torch.from_numpy(tissue_ids),
                    'celltype': torch.from_numpy(celltype_ids),
                    'stage': torch.from_numpy(stage_ids),
                }
            }

            yield batch
