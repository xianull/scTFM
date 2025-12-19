from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.rtf_dataset import SomaRTFDataset

class RTFDataModule(LightningDataModule):
    """
    用于 Rectified Flow 训练的 LightningDataModule。

    特点：
    1. 支持 Latent 和 Raw 两种模式（通过 latent_key 控制）
    2. 集成智能负载均衡（shard_assignment）
    3. 支持多方向细胞对（forward/backward/both）
    4. 支持时间归一化和 Stage 映射

    参数:
        data_dir: TileDB SOMA 根目录
        batch_size: 批次大小
        num_workers: DataLoader worker 数量
        pin_memory: 是否固定内存
        io_chunk_size: TileDB 读取块大小
        prefetch_factor: 预加载因子
        persistent_workers: 是否保持 worker 存活
        split_label_train: 训练集标签 (默认 0)
        split_label_val: 验证集标签 (默认 1)
        latent_key: 潜在空间键名 (None=Raw, "X_latent"=Latent)
        direction: 细胞对方向 ('forward', 'backward', 'both')
        shard_assignment: 智能负载均衡方案（由 train.py 注入）
        stage_info_path: Stage 信息 CSV 路径
        use_log_time: 是否使用 log-scale 时间归一化
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
        split_label_train: int = 0,
        split_label_val: int = 1,
        latent_key: Optional[str] = None,  # 关键：控制 Latent 还是 Raw
        direction: str = "forward",
        shard_assignment: Optional[Dict[str, Any]] = None,
        stage_info_path: Optional[str] = None,  # Stage 信息 CSV 路径
        use_log_time: bool = True,  # 是否使用 log-scale 时间归一化
        # 以下参数由 train.py 使用，DataModule 不需要
        raw_data_dir: Optional[str] = None,  # 仅供 train.py 推理 latent 目录
        latent_dir: Optional[str] = None,    # 仅供 train.py 使用
    ):
        super().__init__()

        # 保存所有参数到 hparams（Lightning 会自动处理）
        self.save_hyperparameters(logger=False)

        # 数据集占位符
        self.data_train: Optional[SomaRTFDataset] = None
        self.data_val: Optional[SomaRTFDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        设置数据集（在 Trainer 调用 fit/validate/test 之前执行）。

        注意：
        - train.py 已经完成了 shard_assignment 的计算和注入
        - 这里直接使用 self.hparams.shard_assignment
        """
        if not self.data_train and not self.data_val:
            # 训练集
            self.data_train = SomaRTFDataset(
                root_dir=self.hparams.data_dir,
                split_label=self.hparams.split_label_train,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                latent_key=self.hparams.latent_key,  # 关键：传递模式信息
                direction=self.hparams.direction,
                shard_assignment=self.hparams.shard_assignment,
                stage_info_path=self.hparams.stage_info_path,
                use_log_time=self.hparams.use_log_time,
            )

            # 验证集
            self.data_val = SomaRTFDataset(
                root_dir=self.hparams.data_dir,
                split_label=self.hparams.split_label_val,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                latent_key=self.hparams.latent_key,
                direction=self.hparams.direction,
                shard_assignment=self.hparams.shard_assignment,
                stage_info_path=self.hparams.stage_info_path,
                use_log_time=self.hparams.use_log_time,
            )

    def train_dataloader(self):
        """返回训练 DataLoader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,  # Dataset 内部已经处理了 batching
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        """返回验证 DataLoader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

