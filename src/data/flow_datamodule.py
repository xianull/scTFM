from typing import Optional, Dict

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.flow_dataset import FlowSomaDataset

class FlowDataModule(LightningDataModule):
    """
    用于 Flow Matching 训练的 DataModule。
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        latent_key: str = "X_latent",
        persistent_workers: bool = True,
        condition_keys: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[FlowSomaDataset] = None
        self.data_val: Optional[FlowSomaDataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val:
            self.data_train = FlowSomaDataset(
                root_dir=self.hparams.data_dir,
                split_label=0, # Train
                batch_size=self.hparams.batch_size,
                latent_key=self.hparams.latent_key,
                condition_keys=self.hparams.condition_keys,
            )
            self.data_val = FlowSomaDataset(
                root_dir=self.hparams.data_dir,
                split_label=1, # Val
                batch_size=self.hparams.batch_size,
                latent_key=self.hparams.latent_key,
                condition_keys=self.hparams.condition_keys,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
