import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseFlow(nn.Module, ABC):
    """
    Flow Model 基类，定义统一接口。
    
    所有 Flow 变体都需要实现:
    1. forward(): 训练时的损失计算
    2. sample(): 采样/生成逻辑
    """
    
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    
    @abstractmethod
    def forward(self, x1: torch.Tensor, cond_data: Dict[str, Any] = None) -> torch.Tensor:
        """
        训练阶段前向传播：计算损失。
        
        Args:
            x1: 目标数据 (batch_size, dim)
            cond_data: 条件信息字典，包含 x_curr, time_curr, time_next 等
        
        Returns:
            loss: 标量损失值
        """
        pass
    
    @abstractmethod
    def sample(
        self, 
        x0: torch.Tensor, 
        cond_data: Dict[str, Any], 
        steps: int = 50, 
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        采样/生成逻辑。
        
        Args:
            x0: 初始状态（通常是噪声或起始细胞）
            cond_data: 条件信息
            steps: ODE solver 步数
            method: 求解方法 ('euler', 'rk4')
        
        Returns:
            x1: 生成的目标状态
        """
        pass

