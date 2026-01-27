"""NeighborhoodDataModule - 用于 SetSCAE 的微环境数据模块"""

from typing import Optional, Dict
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.neighborhood_dataset import SomaNeighborhoodDataset


class NeighborhoodDataModule(LightningDataModule):
    """
    微环境数据的 PyTorch Lightning DataModule。

    用于训练 SetSCAE（Set Single-Cell Autoencoder）。
    每个 batch 返回 (batch_size, bag_size, n_genes) 的细胞集合。

    Split Labels:
    0: Train (ID) - 用于训练
    1: Val (ID)   - 用于验证
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        io_chunk_size: int = 16384,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        split_label_train: int = 0,
        split_label_val: int = 1,
        shard_assignment: Optional[Dict] = None,
        # 微环境参数
        bag_size: int = 16,
        set_size: int = 16, # Alias for bag_size
        mask_ratio: float = 0.0,
        mask_strategy: str = "random",
    ):
        """
        Args:
            data_dir: 数据集根目录
            batch_size: 每个 batch 的 bag 数量
            num_workers: DataLoader 的 worker 数量
            pin_memory: 是否将数据锁在内存中
            io_chunk_size: TileDB 读取时的 chunk 大小
            prefetch_factor: 每个 worker 预加载的 batch 数量
            persistent_workers: 是否保持 workers 存活
            split_label_train: 训练集标签
            split_label_val: 验证集标签
            shard_assignment: 智能负载均衡的 shard 分配方案
            bag_size: 每个微环境 bag 中的细胞数量
            set_size: bag_size 的别名，优先使用 bag_size (config 中可能两个都有)
            mask_ratio: Masked AE 的掩码比例
            mask_strategy: 掩码策略
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Resolve bag_size / set_size alias
        # If set_size passed but bag_size is default, prefer set_size.
        # However, Hydra instantiates with whatever is in config.
        # Usually 'set_size' is the parameter we want to use.
        if set_size != 16 and bag_size == 16:
             self.hparams.bag_size = set_size
        # Or just sync them
        self.hparams.bag_size = set_size # Enforce set_size as primary if passed

        self.data_train: Optional[SomaNeighborhoodDataset] = None
        self.data_val: Optional[SomaNeighborhoodDataset] = None
        self._cached_sub_uris: Optional[list] = None

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if not self.data_train and not self.data_val:
            # 预扫描 Shards
            if self._cached_sub_uris is None:
                print(f"🔍 [NeighborhoodDataModule] Pre-scanning shards in {self.hparams.data_dir}...")
                self._cached_sub_uris = sorted([
                    os.path.join(self.hparams.data_dir, d)
                    for d in os.listdir(self.hparams.data_dir)
                    if os.path.isdir(os.path.join(self.hparams.data_dir, d))
                ])
                print(f"✅ [NeighborhoodDataModule] Found {len(self._cached_sub_uris)} shards")

            # 训练集
            self.data_train = SomaNeighborhoodDataset(
                root_dir=self.hparams.data_dir,
                split_label=self.hparams.split_label_train,
                bag_size=self.hparams.bag_size,
                batch_size=self.hparams.batch_size,
                io_chunk_size=self.hparams.io_chunk_size,
                preloaded_sub_uris=self._cached_sub_uris,
                shard_assignment=self.hparams.shard_assignment,
                mask_ratio=self.hparams.mask_ratio,
                mask_strategy=self.hparams.mask_strategy,
            )

            # 验证集
            self.data_val = SomaNeighborhoodDataset(
                root_dir=self.hparams.data_dir,
                split_label=self.hparams.split_label_val,
                bag_size=self.hparams.bag_size,
                batch_size=self.hparams.batch_size,
                io_chunk_size=self.hparams.io_chunk_size,
                preloaded_sub_uris=self._cached_sub_uris,
                shard_assignment=None,  # 验证集不需要负载均衡
                mask_ratio=self.hparams.mask_ratio,
                mask_strategy=self.hparams.mask_strategy,
            )

    def train_dataloader(self):
        """返回训练集的 DataLoader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,  # Dataset 已经处理了 batching
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        """返回验证集的 DataLoader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
