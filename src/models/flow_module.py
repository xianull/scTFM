from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Optimizer

from src.models.components.flow.rectified_flow import RectifiedFlow

class FlowLitModule(LightningModule):
    """
    用于训练 Rectified Flow 的 LightningModule。
    """
    
    def __init__(
        self,
        net: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.flow = RectifiedFlow(backbone=net)
        
        if compile and hasattr(torch, "compile"):
            self.flow = torch.compile(self.flow)

    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any]):
        return self.flow(x1, cond_data)

    def model_step(self, batch: Any):
        x1 = batch['x_next']
        cond_data = batch['cond_meta']
        cond_data['x_curr'] = batch['x_curr']
        
        loss = self.flow(x1, cond_data)
        
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        # 添加 sync_dist=True
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
