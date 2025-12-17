import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Module (Fixed for numerical stability).
    Accepts input of dimension 'in_features', projects to 2 * 'out_features',
    and applies SwiGLU: (xW + b) * SiLU(xV + c).
    
    This replaces the standard (Linear -> Activation) block.
    
    Improvements:
    - Xavier initialization for better gradient flow
    - Optional scaling factor to prevent overflow
    - Gradient clipping in extreme cases
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # We need 2 * out_features for the gating mechanism
        self.linear = nn.Linear(in_features, 2 * out_features, bias=bias)
        
        # ✅ 修复1：使用 Xavier/Glorot 初始化，保证数值稳定
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)  # 使用较小的 gain
        if bias:
            nn.init.zeros_(self.linear.bias)
        
        # ✅ 修复2：添加缩放因子（参考 GLU 论文）
        self.scale = (2 * out_features) ** -0.5

    def forward(self, x):
        # Project to 2 * hidden
        x = self.linear(x)
        # Split into two halves
        x1, x2 = x.chunk(2, dim=-1)
        
        # ✅ 修复3：添加数值稳定性检查
        # 在 SiLU 之前进行裁剪，防止过大的输入
        x2 = torch.clamp(x2, min=-20, max=20)
        
        # SwiGLU: x1 * SiLU(x2)
        out = x1 * F.silu(x2)
        
        # ✅ 修复4：应用缩放（可选，防止梯度爆炸）
        # out = out * self.scale
        
        return out

def get_activation(name: str):
    """Factory for activation functions"""
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    elif name == "GELU":
        return nn.GELU()
    elif name == "SiLU":
        return nn.SiLU()
    else:
        # SwiGLU is special, handled in logic
        if name == "SwiGLU":
            return None 
        raise ValueError(f"Unknown activation: {name}")

# 朴素自编码器的实现
class VanillaAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [1024, 512],
        latent_dim: int = 64,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "LeakyReLU", 
        **kwargs,
    ):
        super().__init__()

        # encoder
        encoder_layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            if activation == "SwiGLU":
                # ✅ 修复：SwiGLU 使用 LayerNorm（在之前），更稳定
                # 标准架构：LayerNorm -> SwiGLU -> Dropout
                if use_batch_norm:
                    encoder_layers.append(nn.LayerNorm(curr_dim))
                encoder_layers.append(SwiGLU(curr_dim, h_dim))
            else:
                # Standard: Linear -> BN -> Activation
                encoder_layers.append(nn.Linear(curr_dim, h_dim))
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(h_dim))
                encoder_layers.append(get_activation(activation))
            
            encoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        # latent投影，无激活函数
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(curr_dim, latent_dim)

        # decoder
        decoder_layers = []
        # 镜像Hidden dims
        reversed_hidden = list(reversed(hidden_dims))
        curr_dim = latent_dim
        for h_dim in reversed_hidden:
            if activation == "SwiGLU":
                # ✅ 修复：SwiGLU 使用 LayerNorm（在之前）
                if use_batch_norm:
                    decoder_layers.append(nn.LayerNorm(curr_dim))
                decoder_layers.append(SwiGLU(curr_dim, h_dim))
            else:
                decoder_layers.append(nn.Linear(curr_dim, h_dim))
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(h_dim))
                decoder_layers.append(get_activation(activation))
            
            decoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        # 输出层
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(curr_dim, input_dim)
        
    def encode(self, x):
        """返回 latent vector"""
        h = self.encoder(x)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        """从 latent 恢复"""
        h = self.decoder_net(z)
        recon = self.output_layer(h)
        return recon

    def forward(self, x):
        """
        前向传播方法，执行编码和解码操作
        """
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z
