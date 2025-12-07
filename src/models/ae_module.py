from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class AELitModule(LightningModule):
    """
    Autoencoder 的 PyTorch Lightning 模块。
    它作为一个系统，负责管理：训练循环、优化器配置、Loss 计算、日志记录。
    它不包含具体的网络架构，网络架构通过 `net` 参数注入。
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ):
        """
        Args:
            net: 具体的 Autoencoder 网络实例 (由 Hydra 递归实例化)
            optimizer: 优化器配置 (partial, 等待传入 params)
            scheduler: 调度器配置 (partial)
            compile: 是否使用 torch.compile 加速 (PyTorch 2.0+)
        """
        super().__init__()

        # 1. 保存超参数
        # logger=False 是关键，下面会详细解释
        self.save_hyperparameters(logger=False, ignore=["net"])

        # 2. 注入网络架构
        self.net = net

        # 3. 可选：PyTorch 2.0 编译加速
        if compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor):
        """
        前向传播，直接调用内部的 net
        """
        return self.net(x)

    def on_train_start(self):
        """
        训练开始前的钩子，用于初始化一些 metric 防止 crash
        """
        # 记录 val/loss 为 0，防止 wandb 在第一个 epoch 前没有这个字段报错
        self.log("val/loss", 0.0, sync_dist=True)

    def model_step(self, batch: Any):
        """
        通用的 step 逻辑，供 training/validation/test 共享
        """
        # TileDBDataset 返回的是 (dense_vector, global_index)
        x, _ = batch
        
        # Forward pass
        # 假设 net 的 forward 返回 (recon_x, z)
        recon_x, z = self.forward(x)

        # Loss 计算 (MSE Loss 适用于 log1p 后的数据)
        loss = F.mse_loss(recon_x, x)

        return loss, recon_x, z

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)

        # 记录训练 Loss
        # on_step=False, on_epoch=True 表示不在每个 step 画图，只在 epoch 结束画一个点，节省 I/O
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        配置优化器和调度器
        """
        # 实例化 partial optimizer
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss", # ReduceLROnPlateau 需要监控指标
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}