import torch
import torch.nn as nn
import torch.nn.functional as F

# 朴素自编码器的实现
class VanillaAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [1024, 512],
        latent_dim: int = 64,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        # encoder
        encoder_layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU())
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
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU())
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