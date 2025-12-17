import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        t = t * 1000.0 
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer Backbone.
    """
    def __init__(
        self,
        input_dim,
        hidden_size=384,
        depth=6,
        num_heads=6,
        n_tissues=50,
        n_celltypes=100,
        cond_dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # 1. Input Embedding
        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 2. Condition Embeddings
        self.tissue_emb = nn.Embedding(n_tissues + 1, hidden_size)
        self.celltype_emb = nn.Embedding(n_celltypes + 1, hidden_size)
        self.x_curr_embedder = nn.Linear(input_dim, hidden_size)
        self.abs_time_embedder = TimestepEmbedder(hidden_size)
        self.dt_embedder = TimestepEmbedder(hidden_size)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 4. Final Layer
        self.final_layer = FinalLayer(hidden_size, input_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, cond_data=None):
        """
        前向传播。
        
        Args:
            x: 输入数据 (N, input_dim)
            t: 时间步 (N,)
            cond_data: 条件信息字典（可选）
                - x_curr: 当前细胞状态 (N, input_dim)
                - time_curr: 当前时间 (N,) 或 'time'
                - delta_t: 时间差 (N,) 或 'dt'
                - tissue: 组织ID (N,) [可选]
                - celltype: 细胞类型ID (N,) [可选]
        
        Returns:
            v_pred: 预测的速度场 (N, input_dim)
        """
        # 1. Embed Input -> (N, 1, D)
        x_emb = self.x_embedder(x).unsqueeze(1)
        
        # 2. Embed Flow Time
        t_emb = self.t_embedder(t)
        
        # 3. Embed Conditions (灵活处理可能缺失的键)
        c = t_emb  # 基础条件
        
        if cond_data is not None:
            # x_curr (当前细胞状态)
            if 'x_curr' in cond_data:
                c_xcurr = self.x_curr_embedder(cond_data['x_curr'])
                c = c + c_xcurr
            
            # 绝对时间 (支持 'time_curr' 或 'time')
            time_key = 'time_curr' if 'time_curr' in cond_data else 'time'
            if time_key in cond_data:
                c_abs_time = self.abs_time_embedder(cond_data[time_key])
                c = c + c_abs_time
            
            # 时间差 (支持 'delta_t' 或 'dt')
            dt_key = 'delta_t' if 'delta_t' in cond_data else 'dt'
            if dt_key in cond_data:
                c_dt = self.dt_embedder(cond_data[dt_key])
                c = c + c_dt
            
            # 组织 (可选)
            if 'tissue' in cond_data:
                tissue_ids = cond_data['tissue']
                # 处理非整数情况（例如字符串或缺失值）
                if not torch.is_tensor(tissue_ids):
                    tissue_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                c_tissue = self.tissue_emb(tissue_ids)
                c = c + c_tissue
            
            # 细胞类型 (可选)
            if 'celltype' in cond_data:
                celltype_ids = cond_data['celltype']
                if not torch.is_tensor(celltype_ids):
                    celltype_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                c_ctype = self.celltype_emb(celltype_ids)
                c = c + c_ctype
        
        # 4. Transformer Blocks
        for block in self.blocks:
            x_emb = block(x_emb, c)
            
        # 5. Final Layer
        v_pred = self.final_layer(x_emb, c)
        
        return v_pred.squeeze(1)

