import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch_size, )
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dims: list = [1024, 1024, 1024],
        dropout: float = 0.1,
        time_emb_dim: int = 64
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Concat 注入： Input = z + cond + time_emb
        input_total_dim = in_dim + cond_dim + time_emb_dim
        
        layers = []
        curr_dim = input_total_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) # Rectified Flow 常用 LayerNorm
            layers.append(nn.SiLU()) # SiLU (Swish) 优于 ReLU
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        
        layers.append(nn.Linear(curr_dim, in_dim)) # 输出 v 场，维度与 z 相同
        
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, cond):
        # z: (B, D)
        # t: (B, )
        # cond: (B, C)
        
        t_emb = self.time_mlp(t)
        
        # 确保 condition 维度匹配 (如果 cond 是 None 或者维度不对，需要处理)
        # 这里假设调用者保证 cond 是 Tensor
        if cond is None:
             # 创建一个全 0 的 dummy condition
             cond = torch.zeros(z.shape[0], 0, device=z.device)
             
        x = torch.cat([z, cond, t_emb], dim=-1)
        
        out = self.net(x)
        return out

