import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.models.components.flow.base_flow import BaseFlow

class FlowMatching(BaseFlow):
    """
    Flow Matching (Conditional Flow Matching, CFM)。
    
    Paper: Flow Matching for Generative Modeling (ICLR 2023)
    Target: 学习条件最优传输路径。
    
    核心思想：
    - 使用 Conditional Probability Paths (CPP)
    - z_t = t * x1 + (1-t) * x0 + sigma_t * epsilon
    - v_target = (x1 - z_t) / (1 - t + eps)
    - 支持可学习的时间依赖方差
    """
    
    def __init__(
        self, 
        backbone: nn.Module,
        sigma_min: float = 0.001,  # 最小噪声水平
        sigma_max: float = 0.1,    # 最大噪声水平
    ):
        super().__init__(backbone)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算时间依赖的噪声水平。
        
        Args:
            t: 时间步 (batch_size,)
        
        Returns:
            sigma_t: 噪声水平 (batch_size,)
        """
        # 线性插值：sigma_t = sigma_max * (1-t) + sigma_min * t
        return self.sigma_max * (1 - t) + self.sigma_min * t
    
    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any] = None) -> torch.Tensor:
        """
        训练阶段前向传播：计算 Flow Matching Loss。
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 1. 采样时间 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        t_expand = t.view(batch_size, 1)
        
        # 2. 准备 Source 样本 x0（噪声或条件输入）
        if cond_data and 'x_curr' in cond_data:
            # 如果提供了起始细胞，使用它作为 source
            x0 = cond_data['x_curr']
        else:
            # 否则使用纯噪声
            x0 = torch.randn_like(x1)
        
        # 3. 添加时间依赖噪声
        sigma_t = self.get_sigma_t(t).view(batch_size, 1)
        epsilon = torch.randn_like(x1)
        
        # 4. 构建 z_t（带噪声的线性插值）
        z_t = t_expand * x1 + (1 - t_expand) * x0 + sigma_t * epsilon
        
        # 5. 计算目标速度（条件流）
        # v_target = (x1 - z_t) / (1 - t + eps)
        # 这里使用 eps 避免除零
        eps = 1e-5
        v_target = (x1 - z_t) / (1 - t_expand + eps)
        
        # 6. 模型预测
        v_pred = self.backbone(z_t, t, cond_data)
        
        # 7. 计算 Loss
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
        采样/生成逻辑 (ODE Solver with stochastic noise)。
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
            
            # 预测速度
            v_pred = self.backbone(z, t_batch, cond_data)
            
            if method == 'euler':
                z = z + v_pred * dt
            elif method == 'rk4':
                # Runge-Kutta 4
                k1 = v_pred
                
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


class VelocityFlow(BaseFlow):
    """
    Velocity Flow (Velocity Matching)。
    
    核心思想：
    - 直接学习速度场 v(z_t, t)
    - z_t = x0 + t * v_avg
    - v_target = d(z_t)/dt
    - 更关注轨迹的动力学特性
    """
    
    def __init__(self, backbone: nn.Module):
        super().__init__(backbone)
    
    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any] = None) -> torch.Tensor:
        """
        训练阶段前向传播：计算 Velocity Matching Loss。
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 1. 采样时间 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        t_expand = t.view(batch_size, 1)
        
        # 2. 准备 Source 样本 x0
        if cond_data and 'x_curr' in cond_data:
            x0 = cond_data['x_curr']
        else:
            x0 = torch.randn_like(x1)
        
        # 3. 计算平均速度
        v_avg = x1 - x0
        
        # 4. 构建 z_t（基于平均速度的轨迹）
        z_t = x0 + t_expand * v_avg
        
        # 5. 目标速度（在这个简单版本中，就是平均速度）
        v_target = v_avg
        
        # 6. 模型预测
        v_pred = self.backbone(z_t, t, cond_data)
        
        # 7. 计算 Loss
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
        采样/生成逻辑 (Velocity-based ODE Solver)。
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
            
            # 预测速度
            v_pred = self.backbone(z, t_batch, cond_data)
            
            # Euler 步进
            z = z + v_pred * dt
        
        return z

