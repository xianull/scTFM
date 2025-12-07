from typing import Any, Tuple, Optional
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class RectifiedFlowLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def forward(self, z, t, cond):
        return self.net(z, t, cond)

    def model_step(self, batch: Any):
        # Rectified Flow: 1-Rectified Flow (Straight Line)
        # Source Distribution (pi_0): N(0, I)
        # Target Distribution (pi_1): Real Data
        
        x1, _ = batch # (B, D) Real Data
        B = x1.shape[0]
        
        # 1. Sample x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 2. Sample t ~ Uniform[0, 1]
        t = torch.rand(B, device=self.device)
        
        # 3. Interpolation: z_t = t * x1 + (1-t) * x0
        t_expand = t.view(B, 1)
        z_t = t_expand * x1 + (1 - t_expand) * x0
        
        # 4. Target Vector Field: v = x1 - x0
        v_target = x1 - x0
        
        # 5. Condition Handling (目前这里还没有 Conditional 的具体数据流，暂时用空)
        # 实际使用中，cond 可能是 class label embedding 或其他 metadata
        # 必须传入一个空的 tensor 给 MLP，如果 MLP 不需要 condition，cond_dim 设为 0
        cond = torch.empty(B, 0, device=self.device)
        
        # 6. Predict
        v_pred = self.forward(z_t, t, cond)
        
        # 7. Loss (MSE)
        loss = F.mse_loss(v_pred, v_target)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
    
    @torch.no_grad()
    def sample(self, z0: torch.Tensor, cond: Optional[torch.Tensor] = None, n_steps: int = 100):
        """Euler ODE Solver for Sampling"""
        dt = 1.0 / n_steps
        z = z0
        batch_size = z.shape[0]
        
        if cond is None:
             cond = torch.empty(batch_size, 0, device=self.device)
        
        # Trajectory storage for visualization
        trajectory = [z.cpu()]
        
        for i in range(n_steps):
            t_value = i / n_steps
            t = torch.full((batch_size,), t_value, device=self.device)
            
            v_pred = self.forward(z, t, cond)
            z = z + v_pred * dt
            
            trajectory.append(z.cpu())
            
        return z, trajectory

