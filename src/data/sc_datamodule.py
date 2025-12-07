import tiledb
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.components.tiledb_dataset import TileDBDataset, TileDBCollator

class SingleCellDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 16,
        train_val_split: float = 0.95,
        pin_memory: bool = False,
        tile_cache_size: int = 4000000000, # Default 4GB
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train = None
        self.data_val = None

    def setup(self, stage=None):
        """
        在 DDP 模式下，setup 会在每张显卡的进程里都运行一次。
        如果有 8 张卡，就会有 8 个进程同时读取 GPFS。
        必须强制关闭 TileDB 文件锁，否则会发生死锁 (Deadlock)。
        """
        # 防止重复 setup
        if self.data_train and self.data_val:
            return

        counts_uri = f"{self.hparams.data_dir}/counts"
        meta_uri = f"{self.hparams.data_dir}/cell_metadata"
        
        # -----------------------------------------------------------------
        # 【关键修复】定义全局无锁 Context
        # 在 GPFS/S3 等网络存储上，必须禁用文件锁 (vfs.file.enable_filelocks)
        # 增加 Tile Cache 减少 GPFS 访问
        # -----------------------------------------------------------------
        no_lock_cfg = tiledb.Config({
            "sm.compute_concurrency_level": "2",
            "sm.io_concurrency_level": "2",
            "vfs.file.enable_filelocks": "false",  # <--- 核心：强制关锁
            "sm.tile_cache_size": str(self.hparams.tile_cache_size), # <--- Tile 缓存
        })
        # 实例化 Context
        ctx = tiledb.Ctx(no_lock_cfg)

        # 1. 读取 Schema 获取基因数量 (必须传入 ctx)
        try:
            with tiledb.open(counts_uri, mode='r', ctx=ctx) as A:
                n_genes = A.schema.domain.dim("gene_index").domain[1] + 1
        except Exception as e:
            raise FileNotFoundError(f"Could not open TileDB at {counts_uri}. Check path!") from e

        # 2. 读取 Metadata 进行过滤 (必须传入 ctx)
        print(f"Loading metadata from {meta_uri}...")
        try:
            with tiledb.open(meta_uri, mode='r', ctx=ctx) as A:
                # 只读取需要的列，减少 IO
                is_ood = A.query(attrs=["is_ood"])[:]["is_ood"]
        except Exception as e:
             raise FileNotFoundError(f"Could not read metadata at {meta_uri}") from e
        
        # 3. 筛选 In-Distribution 数据 (is_ood == 0)
        valid_indices = np.where(is_ood == 0)[0]
        total_valid = len(valid_indices)
        
        # 4. 固定种子 Shuffle 并划分 (使用 Chunked Shuffle 优化 GPFS 性能)
        print(f"Applying Chunked Shuffle optimization for GPFS...")
        
        # A. 确保物理顺序 (利用 TileDB 的空间局部性)
        valid_indices.sort()
        
        # B. 定义块大小 (Tile Extent 4096 的倍数)
        # 4096 * 20 ≈ 80k 细胞。在这个范围内随机，既保证了 Batch 的随机性，
        # 又保证了读取只会命中 ~20 个 Tile，极大提高 Cache 命中率并减少读放大。
        chunk_size = 4096 * 20 
        rng = np.random.default_rng(seed=42)
        
        if len(valid_indices) > chunk_size:
            n_chunks = len(valid_indices) // chunk_size
            
            # 分割主数据和剩余数据
            main_part = valid_indices[:n_chunks * chunk_size]
            rest_part = valid_indices[n_chunks * chunk_size:]
            
            # Reshape 成 (n_chunks, chunk_size)
            # 注意：创建副本以避免 View 问题
            chunks = main_part.reshape(n_chunks, chunk_size).copy()
            
            # C. 块间 Shuffle (宏观随机：决定先读哪一块 80k 细胞)
            rng.shuffle(chunks)
            
            # D. 块内 Shuffle (微观随机：块内 80k 细胞完全打乱)
            for i in range(n_chunks):
                rng.shuffle(chunks[i])
            
            shuffled_main = chunks.flatten()
            
            # 剩余部分也 shuffle
            rng.shuffle(rest_part)
            
            valid_indices = np.concatenate([shuffled_main, rest_part])
        else:
            # 数据量太小，直接全局 Shuffle
            rng.shuffle(valid_indices)
        
        n_train = int(total_valid * self.hparams.train_val_split)
        train_idxs = valid_indices[:n_train]
        val_idxs = valid_indices[n_train:]
        
        print(f"Dataset Setup: {total_valid} cells (Filtered OOD).")
        print(f"Train: {len(train_idxs)} | Val: {len(val_idxs)}")

        # 5. 实例化 Dataset
        # 注意：TileDBDataset 内部的 __getitem__ 也必须有关锁逻辑
        self.data_train = TileDBDataset(counts_uri, train_idxs, n_genes)
        self.data_val = TileDBDataset(counts_uri, val_idxs, n_genes)

        # 6. 实例化 Collator (用于 Batch 读取 TileDB)
        # 将配置转换为字典传入，确保可序列化
        collator_cfg = {
            "sm.compute_concurrency_level": "2",
            "sm.io_concurrency_level": "2",
            "vfs.file.enable_filelocks": "false",
            "sm.tile_cache_size": str(self.hparams.tile_cache_size),
        }
        self.collator = TileDBCollator(counts_uri, n_genes, ctx_cfg=collator_cfg)

    def train_dataloader(self):
            # 1. 基础参数
            loader_args = dict(
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collator,  # <--- 使用自定义 Collator
            )
            
            # 2. 只有在有 Worker 的时候才启用 spawn 和持久化
            if self.hparams.num_workers > 0:
                loader_args['persistent_workers'] = True
                loader_args['multiprocessing_context'] = 'spawn'  # <--- 动态添加
                
            return DataLoader(self.data_train, **loader_args)

    def val_dataloader(self):
        loader_args = dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,  # <--- 使用自定义 Collator
        )
        
        if self.hparams.num_workers > 0:
            loader_args['persistent_workers'] = True
            loader_args['multiprocessing_context'] = 'spawn' # <--- 动态添加
            
        return DataLoader(self.data_val, **loader_args)