import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from src.models.components.flow.base_flow import BaseFlow


class RectifiedFlow(BaseFlow):
    """
    Rectified Flow (Reflow) 核心逻辑实现。

    Paper: Flow Straight and Fast: Learning to Generate with Rectified Flow (ICLR 2023)
    Target: 学习从 Source Distribution (x0) 到 Target Distribution (x1) 的直线轨迹。

    核心思想：
    - z_t = t * x1 + (1-t) * x0 (线性插值)
    - v_target = x1 - x0 (恒定速度)
    - Loss = MSE(v_pred, v_target)

    新增功能：
    - 支持 Classifier-Free Guidance (CFG) 采样
    """

    def __init__(self, backbone: nn.Module):
        super().__init__(backbone)

    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any] = None, x0: torch.Tensor = None) -> torch.Tensor:
        """
        训练阶段前向传播：计算 MSE Loss。

        Args:
            x1: Target data (x_next)
            cond_data: Condition data
            x0: Source data (x_curr). If None, use Gaussian Noise (standard Reflow).
        """
        batch_size = x1.shape[0]
        device = x1.device

        # 1. 采样时间 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)

        # 2. 准备 Source 样本 x0
        if x0 is None:
            # Standard Reflow: Source=Noise, Target=Data
            x0 = torch.randn_like(x1)
        # Else: Data-to-Data Flow: Source=x_curr, Target=x_next

        # 3. 线性插值构建 z_t
        t_expand = t.view(batch_size, 1)
        z_t = t_expand * x1 + (1 - t_expand) * x0

        # 4. 计算目标速度 v_target
        # v = d(z_t)/dt = x1 - x0
        v_target = x1 - x0

        # 5. 模型预测
        v_pred = self.backbone(z_t, t, cond_data)

        # 6. 计算 Loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def _get_velocity(
        self,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        cond_data: Dict[str, Any],
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        获取速度预测，支持 CFG。

        CFG 公式: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                   = (1 - cfg_scale) * v_uncond + cfg_scale * v_cond

        当 cfg_scale = 1.0 时，等价于普通的条件采样。
        当 cfg_scale > 1.0 时，增强条件的影响。
        """
        if cfg_scale == 1.0:
            # 普通采样，不需要 CFG
            return self.backbone(z, t_batch, cond_data)

        # CFG 采样：需要同时计算 conditional 和 unconditional
        # 检查 backbone 是否支持 force_drop_cond 参数
        if hasattr(self.backbone, 'forward'):
            import inspect
            sig = inspect.signature(self.backbone.forward)
            supports_cfg = 'force_drop_cond' in sig.parameters
        else:
            supports_cfg = False

        if not supports_cfg:
            # Backbone 不支持 CFG，回退到普通采样
            return self.backbone(z, t_batch, cond_data)

        # Conditional prediction
        v_cond = self.backbone(z, t_batch, cond_data, force_drop_cond=False)

        # Unconditional prediction
        v_uncond = self.backbone(z, t_batch, cond_data, force_drop_cond=True)

        # CFG combination
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        return v_pred

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        cond_data: Dict[str, Any],
        steps: int = 50,
        method: str = 'euler',
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        采样/生成逻辑 (ODE Solver)，支持 Classifier-Free Guidance。

        Args:
            x0: 初始噪声 (B, D)
            cond_data: 条件数据字典
            steps: ODE 求解步数
            method: 'euler' 或 'rk4'
            cfg_scale: CFG 强度，1.0 表示不使用 CFG，>1.0 增强条件

        Returns:
            生成的样本 (B, D)
        """
        batch_size = x0.shape[0]
        device = x0.device

        z = x0.clone()
        dt = 1.0 / steps

        # 时间网格: 0 -> 1
        time_steps = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t_curr = time_steps[i]

            # 扩展 t 为 batch
            t_batch = torch.ones(batch_size, device=device) * t_curr

            if method == 'euler':
                v_pred = self._get_velocity(z, t_batch, cond_data, cfg_scale)
                z = z + v_pred * dt

            elif method == 'rk4':
                # Runge-Kutta 4 with CFG
                k1 = self._get_velocity(z, t_batch, cond_data, cfg_scale)

                t2 = t_batch + 0.5 * dt
                z2 = z + 0.5 * dt * k1
                k2 = self._get_velocity(z2, t2, cond_data, cfg_scale)

                t3 = t_batch + 0.5 * dt
                z3 = z + 0.5 * dt * k2
                k3 = self._get_velocity(z3, t3, cond_data, cfg_scale)

                t4 = t_batch + dt
                z4 = z + dt * k3
                k4 = self._get_velocity(z4, t4, cond_data, cfg_scale)

                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return z
