import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class RectifiedFlow(nn.Module):
    """
    Rectified Flow (Reflow) 核心逻辑实现。
    
    Paper: Flow Straight and Fast: Learning to Generate with Rectified Flow (ICLR 2023)
    Target: 学习从 Source Distribution (x0) 到 Target Distribution (x1) 的直线轨迹。
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any] = None) -> torch.Tensor:
        """
        训练阶段前向传播：计算 MSE Loss。
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 1. 采样时间 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        
        # 2. 准备 Source 样本 x0
        # Conditional Flow Matching: Source=Noise, Target=Data
        x0 = torch.randn_like(x1)
        
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

    @torch.no_grad()
    def sample(
        self, 
        x0: torch.Tensor, 
        cond_data: Dict[str, Any], 
        steps: int = 50, 
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        采样/生成逻辑 (ODE Solver)。
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
                v_pred = self.backbone(z, t_batch, cond_data)
                z = z + v_pred * dt
                
            elif method == 'rk4':
                # Runge-Kutta 4
                k1 = self.backbone(z, t_batch, cond_data)
                
                t2 = t_batch + 0.5 * dt
                z2 = z + 0.5 * dt * k1
                k2 = self.backbone(z2, t2, cond_data)
                
                t3 = t_batch + 0.5 * dt
                z3 = z + 0.5 * dt * k2
                k3 = self.backbone(z3, t3, cond_data)
                
                t4 = t_batch + dt
                z4 = z + dt * k3
                k4 = self.backbone(z4, t4, cond_data)
                
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
        return z
