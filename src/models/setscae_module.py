"""
SetSCAE Lightning Module

用于训练 Set Single-Cell Autoencoder。
支持四种策略：graph, contrastive, masked, stack
"""

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.models.components.ae.set_scae import SetSCAE


class SetSCAELitModule(LightningModule):
    """
    SetSCAE 训练模块

    输入格式：
    - x_set: (batch, set_size, n_genes) - 细胞集合（中心 + 邻居）

    支持策略：
    - graph: GAT-based 图自编码器
    - contrastive: 对比学习自编码器
    - masked: 掩码自编码器
    - stack: 基于Stack论文的条件嵌入方法
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compile: bool = False,
        # 训练参数
        center_idx: int = 0,  # 中心细胞在集合中的索引
    ):
        """
        Args:
            net: SetSCAE 网络
            optimizer: 优化器配置
            scheduler: 学习率调度器配置
            compile: 是否使用 torch.compile
            center_idx: 中心细胞索引
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.center_idx = center_idx

        if compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def forward(self, x_set: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.net(x_set, center_idx=self.center_idx, **kwargs)

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """单步训练/验证"""
        # 获取数据 (兼容两种格式: x_set 或 x)
        x_set = batch.get("x_set", batch.get("x"))  # (batch, set_size, n_genes)

        # 可选的额外输入
        adj = batch.get("adj", None)  # 邻接矩阵 (graph 策略)
        raw_counts = batch.get("raw_counts", batch.get("counts", None))  # 兼容 counts
        negatives = batch.get("negatives", None)  # 负样本 (contrastive 策略)
        library_size = batch.get("library_size", None)
        neighbor_mask = batch.get("neighbor_mask", None)  # 邻居掩码 (stack 策略)

        # 前向传播
        outputs = self.net(
            x_set,
            center_idx=self.center_idx,
            adj=adj,
            library_size=library_size,
            neighbor_mask=neighbor_mask,
        )

        # 计算损失
        losses = self.net.compute_loss(
            x_set,
            outputs,
            raw_counts=raw_counts,
            adj=adj,
            negatives=negatives,
            center_idx=self.center_idx,
        )

        return losses

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """训练步骤"""
        losses = self.model_step(batch, batch_idx)

        # 记录损失
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if "recon_loss" in losses:
            self.log("train/recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, sync_dist=True)
        if "contrastive_loss" in losses:
            self.log("train/contrastive_loss", losses["contrastive_loss"], on_step=False, on_epoch=True, sync_dist=True)
        if "link_loss" in losses:
            self.log("train/link_loss", losses["link_loss"], on_step=False, on_epoch=True, sync_dist=True)
        if "mask_pred_loss" in losses:
            self.log("train/mask_pred_loss", losses["mask_pred_loss"], on_step=False, on_epoch=True, sync_dist=True)
        if "kl_loss" in losses:
            self.log("train/kl_loss", losses["kl_loss"], on_step=False, on_epoch=True, sync_dist=True)

        return losses["loss"]

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """验证步骤"""
        losses = self.model_step(batch, batch_idx)

        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if "recon_loss" in losses:
            self.log("val/recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, sync_dist=True)

        return losses["loss"]

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和调度器"""
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

    def get_latent(
        self,
        x_set: torch.Tensor,
        return_context: bool = False,
    ) -> torch.Tensor:
        """获取潜在表征"""
        return self.net.get_latent(x_set, return_context=return_context)
