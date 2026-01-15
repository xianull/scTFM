"""ConsistencyDataModule - 支持时间链一致性训练的数据模块"""

from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.consistency_dataset import SomaConsistencyDataset


class ConsistencyDataModule(LightningDataModule):
    """
    支持时间链一致性训练的 DataModule。

    使用全部数据进行训练，不划分验证集。
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
        max_time_days: float = 100.0,
        min_seq_len: int = 2,
        max_seq_len: int = 5,
        latent_key: Optional[str] = None,
        shard_assignment: Optional[Dict[str, Any]] = None,
        stage_info_path: Optional[str] = None,
        use_log_time: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[SomaConsistencyDataset] = None

    def setup(self, stage: Optional[str] = None):
        """设置数据集 - 使用全部数据"""
        if self.data_train is None:
            self.data_train = SomaConsistencyDataset(
                root_dir=self.hparams.data_dir,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                max_time_days=self.hparams.max_time_days,
                min_seq_len=self.hparams.min_seq_len,
                max_seq_len=self.hparams.max_seq_len,
                latent_key=self.hparams.latent_key,
                shard_assignment=self.hparams.shard_assignment,
                stage_info_path=self.hparams.stage_info_path,
                use_log_time=self.hparams.use_log_time,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )
