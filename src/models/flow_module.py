from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Optimizer

from src.models.components.flow.rectified_flow import RectifiedFlow
from src.models.components.flow.flow_matching import FlowMatching, VelocityFlow

class FlowLitModule(LightningModule):
    """
    用于训练各种 Flow 模型的 LightningModule。
    
    支持的 Flow 类型:
    - rectified_flow: Rectified Flow (默认)
    - flow_matching: Conditional Flow Matching
    - velocity_flow: Velocity Flow
    
    参数:
        net: Backbone 模型（例如 DiT）
        flow_type: Flow 模型类型
        optimizer: 优化器（partial）
        scheduler: 学习率调度器（partial）
        compile: 是否使用 torch.compile 加速
    """
    
    def __init__(
        self,
        net: nn.Module,
        flow_type: str = "rectified_flow",
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        mode: Optional[str] = None,
        ae_ckpt_path: Optional[str] = None,
        scale_factor: float = 1.0,  # 新增缩放因子参数
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False, ignore=["net"])
        
        # 根据 flow_type 创建对应的 Flow 模型
        if flow_type == "rectified_flow":
            self.flow = RectifiedFlow(backbone=net)
        elif flow_type == "flow_matching":
            self.flow = FlowMatching(backbone=net)
        elif flow_type == "velocity_flow":
            self.flow = VelocityFlow(backbone=net)
        else:
            raise ValueError(f"❌ 不支持的 Flow 类型: {flow_type}")
        
        if compile and hasattr(torch, "compile"):
            self.flow = torch.compile(self.flow)

    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any]):
        return self.flow(x1, cond_data)

    def model_step(self, batch: Any):
        x1 = batch['x_next']
        cond_data = batch['cond_meta']
        cond_data['x_curr'] = batch['x_curr']
        
        # [Critical Fix]
        # Allow x0 (source) to be x_curr if available, enabling Data-to-Data flow
        # for both 'raw' (gene space) and 'latent' (embedding space) trajectories.
        # This is essential for modeling cell transitions (Latent t -> Latent t+1).
        x0 = None
        if self.hparams.mode in ['raw', 'latent']:
            x0 = batch['x_curr']
            
        # [Scaling for Raw Mode]
        # If input is log1p transformed (range ~0-10), scale to ~0-1 for stability
        if self.hparams.mode == 'raw':
            scale_factor = self.hparams.scale_factor
            x1 = x1 / scale_factor
            if x0 is not None:
                x0 = x0 / scale_factor
            cond_data['x_curr'] = cond_data['x_curr'] / scale_factor

        loss = self.flow(x1, cond_data, x0=x0)
        
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
