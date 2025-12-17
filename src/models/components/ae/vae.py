import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Module (Fixed for numerical stability).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features, bias=bias)
        
        # ✅ 修复1：使用 Xavier 初始化
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        if bias:
            nn.init.zeros_(self.linear.bias)
        
        self.scale = (2 * out_features) ** -0.5

    def forward(self, x):
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=-1)
        
        # ✅ 修复2：数值裁剪，防止 NaN
        x2 = torch.clamp(x2, min=-20, max=20)
        
        out = x1 * F.silu(x2)
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
        if name == "SwiGLU":
            return None
        raise ValueError(f"Unknown activation: {name}")

class VariationalAE(nn.Module):
    """
    Variational Autoencoder (VAE) 实现。
    """
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

        # Encoder
        encoder_layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            if activation == "SwiGLU":
                # ✅ 修复：LayerNorm 在 SwiGLU 之前
                if use_batch_norm:
                    encoder_layers.append(nn.LayerNorm(curr_dim))
                encoder_layers.append(SwiGLU(curr_dim, h_dim))
            else:
                encoder_layers.append(nn.Linear(curr_dim, h_dim))
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(h_dim))
                encoder_layers.append(get_activation(activation))
            
            encoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        
        # VAE 特有的两个投影头：Mu (均值) 和 LogVar (对数方差)
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)

        # Decoder
        decoder_layers = []
        reversed_hidden = list(reversed(hidden_dims))
        
        # Latent -> First Hidden
        curr_dim = latent_dim
        
        for h_dim in reversed_hidden:
            if activation == "SwiGLU":
                # ✅ 修复：LayerNorm 在 SwiGLU 之前
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
            
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(curr_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_net(z)
        recon = self.output_layer(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, z, mu, logvar
