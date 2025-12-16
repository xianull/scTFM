from typing import Optional, Dict
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.ae_dataset import SomaCollectionDataset

class SingleCellDataModule(LightningDataModule):
    """
    单细胞数据的 PyTorch Lightning DataModule，使用 SomaCollectionDataset。
    
    关键特性：
    Dataset 会直接 yield 一个 batch 的数据，因此 DataLoader 初始化时必须设置 batch_size=None。
    
    Split Labels:
    0: Train (ID) - 用于训练
    1: Val (ID)   - 用于验证
    2: Test (ID)  - 用于测试 (同分布)
    3: Test (OOD) - 用于测试 (外分布) - 暂不需要
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        io_chunk_size: int = 16384,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        shard_assignment: Optional[Dict] = None,  # 新增：负载均衡分配方案
    ):
        """
        Args:
            data_dir: 数据集根目录
            batch_size: 每个 batch 的大小 (直接传递给 SomaCollectionDataset)
            num_workers: DataLoader 的 worker 数量
            pin_memory: 是否将数据锁在内存中 (建议 True)
            io_chunk_size: TileDB 读取时的 chunk 大小 (影响内存占用)
            prefetch_factor: 每个 worker 预加载的 batch 数量
            persistent_workers: 是否保持 workers 存活 (避免重复初始化开销)
            shard_assignment: 智能负载均衡的 shard 分配方案 (可选)
        """
        super().__init__()

        # 允许通过 self.hparams 访问 init 参数
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[SomaCollectionDataset] = None
        self.data_val: Optional[SomaCollectionDataset] = None
        # self.data_test: Optional[SomaCollectionDataset] = None # 暂时不需要Test，或者根据需求开启
        
        # 预扫描 Shards 列表（只在主进程执行一次，避免 64 个 workers 重复扫描）
        self._cached_sub_uris: Optional[list] = None

    def setup(self, stage: Optional[str] = None):
        """
        加载数据。设置变量: `self.data_train`, `self.data_val`.
        
        这个方法会被 trainer.fit() 和 trainer.test() 调用。
        根据用户指示，只需要读取 split_label 0 (Train) 和 1 (Val)。
        """
        # 仅当未加载时才加载数据集
        if not self.data_train and not self.data_val:
            # [关键优化] 在主进程中预扫描所有 Shards，避免 64 个 workers 重复扫描
            if self._cached_sub_uris is None:
                print(f"🔍 [DataModule] Pre-scanning shards in {self.hparams.data_dir}...")
                self._cached_sub_uris = sorted([
                    os.path.join(self.hparams.data_dir, d) 
                    for d in os.listdir(self.hparams.data_dir) 
                    if os.path.isdir(os.path.join(self.hparams.data_dir, d))
                ])
                print(f"✅ [DataModule] Found {len(self._cached_sub_uris)} shards (will be shared across all workers)")
            
            # 训练集 (split_label=0: Train ID)
            self.data_train = SomaCollectionDataset(
                root_dir=self.hparams.data_dir,
                split_label=0,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                preloaded_sub_uris=self._cached_sub_uris,  # 传入预扫描的列表
                shard_assignment=self.hparams.shard_assignment,  # 传入负载均衡方案
            )
            
            # 验证集 (split_label=1: Val ID)
            self.data_val = SomaCollectionDataset(
                root_dir=self.hparams.data_dir,
                split_label=1,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                preloaded_sub_uris=self._cached_sub_uris,  # 复用同一个列表
                shard_assignment=None,  # 验证集不需要负载均衡（数据量小）
            )
            
            # 注意：split_label 2 (Test ID) and 3 (Test OOD) 目前未加载
            # 如果后续需要测试，可以在这里添加

    def train_dataloader(self):
        """返回训练集的 DataLoader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,  # <--- 关键！Dataset 已经处理了 batching
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers, # 使用配置参数
        )

    def val_dataloader(self):
        """返回验证集的 DataLoader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=None,  # <--- 关键！
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def teardown(self, stage: Optional[str] = None):
        """fit 或 test 结束后的清理工作"""
        pass

    def state_dict(self):
        """保存到 checkpoint 的额外状态"""
        return {}

    def load_state_dict(self, state_dict):
        """加载 checkpoint 时的操作"""
        pass
