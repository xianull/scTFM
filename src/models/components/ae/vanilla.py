import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Module.
    Accepts input of dimension 'in_features', projects to 2 * 'out_features',
    and applies SwiGLU: (xW + b) * SiLU(xV + c).
    
    This replaces the standard (Linear -> Activation) block.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # We need 2 * out_features for the gating mechanism
        self.linear = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, x):
        # Project to 2 * hidden
        x = self.linear(x)
        # Split into two halves
        x1, x2 = x.chunk(2, dim=-1)
        # SwiGLU: x1 * SiLU(x2) (or vice versa, symmetric)
        return x1 * F.silu(x2)

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
                # SwiGLU replaces (Linear + Activation)
                # Note: BatchNorm usually comes AFTER linear projection but BEFORE activation in standard ResNet
                # In SwiGLU, the "activation" is built-in.
                # Standard Pre-Norm Transformer: Norm -> Linear -> SwiGLU
                # Standard Post-Norm: Linear -> SwiGLU -> Norm
                # Here we follow the existing pattern: Linear -> BN -> Act -> Dropout
                # SwiGLU combines Linear + Act.
                # So we do: SwiGLU(in->out) -> BN -> Dropout? 
                # Or SwiGLU includes the "Linear" part.
                # Let's use: SwiGLU(curr -> h) -> BN -> Dropout.
                
                encoder_layers.append(SwiGLU(curr_dim, h_dim))
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(h_dim))
                # No separate activation layer needed
            else:
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
                decoder_layers.append(SwiGLU(curr_dim, h_dim))
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(h_dim))
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
