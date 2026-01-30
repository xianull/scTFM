"""
DiT with Cross-Attention Conditioning and Classifier-Free Guidance Support.

核心设计：
1. x_curr: 通过 Cross-Attention 注入（核心条件）
2. tissue + celltype + stage: 组合成 context tokens，也通过 Cross-Attention 注入
3. time_curr + delta_t: 通过 adaLN 调制（与 flow time t 一起）
4. CFG: 支持对所有条件进行 dropout

条件重要性分析：
- x_curr: 当前细胞状态，是预测的核心输入（高维，需要 attention）
- tissue + celltype + stage: 决定分化规则/轨迹模式（类别，需要语义理解）
- delta_t: 时间步长，决定变化幅度（标量，适合 adaLN）
- time_curr: 绝对时间点，决定发育阶段（标量，适合 adaLN）

时间编码说明：
- flow_time (t): [0, 1] 范围，乘以 1000 后做 sinusoidal embedding
- time_curr: [0, 1] 范围（已 log-scale 归一化），乘以 1000 后做 sinusoidal embedding
- delta_t: [-1, 1] 范围（已对称 log-scale 归一化），乘以 1000 后做 sinusoidal embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    输入期望在 [0, 1] 或 [-1, 1] 范围内（已归一化）。
    内部会乘以 scale_factor 扩展到更大范围，然后做 sinusoidal embedding。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, scale_factor=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale_factor = scale_factor

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, scale_factor=1000.0):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (B,) 时间值，期望在 [0, 1] 或 [-1, 1] 范围
            dim: embedding 维度
            max_period: sinusoidal 的最大周期
            scale_factor: 将输入缩放到更大范围（默认 1000）
        """
        t = t * scale_factor  # [0, 1] -> [0, 1000] 或 [-1, 1] -> [-1000, 1000]
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, scale_factor=self.scale_factor)
        t_emb = self.mlp(t_freq)
        return t_emb


class StageEmbedder(nn.Module):
    """
    Stage (发育阶段) 的顺序编码器。

    发育阶段有天然的顺序关系：
    Unknown(0) < Embryonic(1) < Fetal(2) < Newborn(3) < Paediatric(4) < Adult(5)

    编码策略：
    1. 将离散的 stage_id 转换为 [0, 1] 范围的连续值
    2. 使用 sinusoidal embedding（与时间编码一致）
    3. 通过 MLP 投影到 hidden_size

    这样做的好处：
    - 保留了阶段间的顺序关系（Embryonic < Fetal < Adult）
    - 与时间编码方式一致，便于模型学习
    - 相邻阶段的 embedding 更相似
    """

    def __init__(self, hidden_size, n_stages=6, frequency_embedding_size=256):
        super().__init__()
        self.n_stages = n_stages
        self.frequency_embedding_size = frequency_embedding_size

        # MLP 将 sinusoidal embedding 投影到 hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, stage_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stage_ids: (B,) long tensor, 值在 [0, n_stages] 范围

        Returns:
            stage_emb: (B, hidden_size)
        """
        # 1. 顺序编码：将 stage_id 归一化到 [0, 1]
        # Unknown(0) -> 0.0, Embryonic(1) -> 0.2, ..., Adult(5) -> 1.0
        stage_normalized = stage_ids.float() / max(self.n_stages - 1, 1)

        # 2. Sinusoidal embedding（与 TimestepEmbedder 一致）
        stage_freq = TimestepEmbedder.timestep_embedding(
            stage_normalized,
            self.frequency_embedding_size,
            scale_factor=1000.0  # [0, 1] -> [0, 1000]
        )

        # 3. MLP 投影
        stage_emb = self.mlp(stage_freq)

        return stage_emb


class CrossAttention(nn.Module):
    """
    Cross-Attention 模块：Query 来自 x，Key/Value 来自条件 context。
    """

    def __init__(self, hidden_size, num_heads, context_dim=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        context_dim = context_dim or hidden_size

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_size, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        Args:
            x: (B, L, D) - query tokens
            context: (B, L_ctx, D_ctx) - key/value tokens
        Returns:
            (B, L, D)
        """
        B, L, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.to_out(out)


class DiTBlockWithCrossAttn(nn.Module):
    """
    DiT Block with:
    1. Self-Attention (original)
    2. Cross-Attention to context (x_curr + tissue/celltype)
    3. MLP (original)

    All modulated by adaLN-Zero using scalar conditions (t, time_curr, delta_t)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        context_dim=None,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        # Cross-Attention
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads,
            context_dim=context_dim or hidden_size,
            dropout=dropout
        )

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN-Zero modulation: 8 outputs
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size, bias=True)
        )

    def forward(self, x, c, context=None):
        """
        Args:
            x: (B, L, D) - input tokens
            c: (B, D) - scalar condition embedding (t, time_curr, delta_t)
            context: (B, L_ctx, D) - cross-attention context (x_curr + tissue/celltype)
        """
        modulation = self.adaLN_modulation(c).chunk(8, dim=1)
        shift_sa, scale_sa, gate_sa, gate_cross, shift_mlp, scale_mlp, gate_mlp, scale_cross = modulation

        # 1. Self-Attention
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # 2. Cross-Attention (if context provided)
        if context is not None:
            x_norm_cross = self.norm_cross(x)
            cross_out = self.cross_attn(x_norm_cross, context * (1 + scale_cross.unsqueeze(1)))
            x = x + gate_cross.unsqueeze(1) * cross_out

        # 3. MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN-Zero modulation."""

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


class DiTCrossAttn(nn.Module):
    """
    Diffusion Transformer with Enhanced Conditioning.

    条件注入策略：
    ┌─────────────────┬────────────────────┬─────────────────────────────┐
    │ 条件            │ 注入方式           │ 原因                        │
    ├─────────────────┼────────────────────┼─────────────────────────────┤
    │ x_curr          │ Cross-Attention    │ 高维，需要学习关注哪些特征  │
    │ tissue          │ Cross-Attention    │ 类别，与 x_curr 组合        │
    │ celltype        │ Cross-Attention    │ 类别，与 x_curr 组合        │
    │ stage           │ Cross-Attention    │ 发育阶段，与 x_curr 组合    │
    │ time_curr       │ adaLN              │ 标量，全局调制              │
    │ delta_t         │ adaLN              │ 标量，全局调制              │
    │ flow_time (t)   │ adaLN              │ 标量，全局调制              │
    └─────────────────┴────────────────────┴─────────────────────────────┘

    CFG 策略：
    - 训练时以 cond_dropout 概率将条件替换为 null embeddings
    - 采样时可以用 cfg_scale > 1.0 增强条件
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        n_tissues: int = 50,
        n_celltypes: int = 100,
        n_stages: int = 6,
        n_xcurr_tokens: int = 32,
        frequency_embedding_size: int = 256,
        cond_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.cond_dropout = cond_dropout

        # ============================================================
        # 1. Input Embedding
        # ============================================================
        self.x_embedder = nn.Linear(input_dim, hidden_size)

        # ============================================================
        # 2. Scalar Condition Embeddings (for adaLN)
        # ============================================================
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)  # Flow time t
        self.abs_time_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)  # Absolute time
        self.dt_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)  # Delta time

        # ============================================================
        # 3. Context Embeddings (for Cross-Attention)
        # ============================================================
        # x_curr -> multiple tokens
        self.n_xcurr_tokens = n_xcurr_tokens
        self.x_curr_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size * self.n_xcurr_tokens),
            nn.SiLU(),
            nn.Linear(hidden_size * self.n_xcurr_tokens, hidden_size * self.n_xcurr_tokens),
        )

        # Tissue embedding -> 1 token
        self.tissue_emb = nn.Embedding(n_tissues + 1, hidden_size)

        # Celltype embedding -> 1 token
        self.celltype_emb = nn.Embedding(n_celltypes + 1, hidden_size)

        # Stage embedding -> 1 token (发育阶段，使用顺序编码)
        self.stage_emb = StageEmbedder(hidden_size, n_stages=n_stages, frequency_embedding_size=frequency_embedding_size)

        # Null embeddings for CFG
        self.null_xcurr = nn.Parameter(torch.randn(1, self.n_xcurr_tokens, hidden_size) * 0.02)
        self.null_tissue = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.null_celltype = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.null_stage = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Context projection (to refine combined context)
        # Total context tokens: n_xcurr_tokens + 1 (tissue) + 1 (celltype) + 1 (stage) = 7
        self.n_context_tokens = self.n_xcurr_tokens + 3
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ============================================================
        # 4. Transformer Blocks
        # ============================================================
        self.blocks = nn.ModuleList([
            DiTBlockWithCrossAttn(
                hidden_size,
                num_heads,
                context_dim=hidden_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ============================================================
        # 5. Final Layer
        # ============================================================
        self.final_layer = FinalLayer(hidden_size, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # TimestepEmbedder and StageEmbedder
        for embedder in [self.t_embedder, self.abs_time_embedder, self.dt_embedder, self.stage_emb]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        # Zero-init adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _prepare_context(
        self,
        x_curr: torch.Tensor,
        tissue: Optional[torch.Tensor],
        celltype: Optional[torch.Tensor],
        stage: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        准备 Cross-Attention 的 context。

        Context 组成：
        - x_curr tokens: [n_xcurr_tokens, hidden_size]
        - tissue token: [1, hidden_size]
        - celltype token: [1, hidden_size]
        - stage token: [1, hidden_size]

        Args:
            x_curr: (B, input_dim)
            tissue: (B,) long tensor or None
            celltype: (B,) long tensor or None
            stage: (B,) long tensor or None
            drop_mask: (B,) bool - True = drop condition (for CFG)

        Returns:
            context: (B, n_context_tokens, hidden_size)
        """
        B = x_curr.shape[0]
        device = x_curr.device

        # 1. x_curr tokens
        xcurr_tokens = self.x_curr_proj(x_curr)
        xcurr_tokens = xcurr_tokens.view(B, self.n_xcurr_tokens, self.hidden_size)

        # 2. Tissue token
        if tissue is not None and torch.is_tensor(tissue):
            tissue_token = self.tissue_emb(tissue).unsqueeze(1)  # (B, 1, D)
        else:
            tissue_token = self.null_tissue.expand(B, -1, -1)

        # 3. Celltype token
        if celltype is not None and torch.is_tensor(celltype):
            celltype_token = self.celltype_emb(celltype).unsqueeze(1)  # (B, 1, D)
        else:
            celltype_token = self.null_celltype.expand(B, -1, -1)

        # 4. Stage token
        if stage is not None and torch.is_tensor(stage):
            stage_token = self.stage_emb(stage).unsqueeze(1)  # (B, 1, D)
        else:
            stage_token = self.null_stage.expand(B, -1, -1)

        # 5. Concatenate all context tokens
        context = torch.cat([xcurr_tokens, tissue_token, celltype_token, stage_token], dim=1)

        # 6. Apply CFG dropout
        if drop_mask is not None and drop_mask.any():
            null_xcurr = self.null_xcurr.expand(B, -1, -1)
            null_tissue = self.null_tissue.expand(B, -1, -1)
            null_celltype = self.null_celltype.expand(B, -1, -1)
            null_stage = self.null_stage.expand(B, -1, -1)
            null_context = torch.cat([null_xcurr, null_tissue, null_celltype, null_stage], dim=1)

            # Replace with null context where drop_mask is True
            context = torch.where(
                drop_mask.view(B, 1, 1),
                null_context,
                context
            )

        # 7. Project context
        context = self.context_proj(context)

        return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_data: Optional[Dict[str, Any]] = None,
        force_drop_cond: bool = False,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: (B, input_dim) - 输入（噪声/中间状态）
            t: (B,) - Flow 时间步
            cond_data: 条件字典
                - x_curr: (B, input_dim) 当前细胞状态
                - time_curr: (B,) 绝对时间（已归一化到 [0, 1]）
                - delta_t: (B,) 时间差（已归一化到 [-1, 1]）
                - tissue: (B,) 组织 ID
                - celltype: (B,) 细胞类型 ID
                - stage: (B,) 发育阶段 ID
            force_drop_cond: 强制 drop 所有条件（CFG unconditional）

        Returns:
            v_pred: (B, input_dim)
        """
        B = x.shape[0]
        device = x.device

        # ============================================================
        # 1. Embed Input
        # ============================================================
        x_emb = self.x_embedder(x).unsqueeze(1)  # (B, 1, D)

        # ============================================================
        # 2. Scalar Conditions -> adaLN
        # ============================================================
        c = self.t_embedder(t)

        if cond_data is not None:
            # Absolute time (已归一化到 [0, 1])
            time_key = 'time_curr' if 'time_curr' in cond_data else 'time'
            if time_key in cond_data:
                c = c + self.abs_time_embedder(cond_data[time_key])

            # Delta time (已归一化到 [-1, 1])
            dt_key = 'delta_t' if 'delta_t' in cond_data else 'dt'
            if dt_key in cond_data:
                c = c + self.dt_embedder(cond_data[dt_key])

        # ============================================================
        # 3. High-dim Conditions -> Cross-Attention Context
        # ============================================================
        context = None
        if cond_data is not None and 'x_curr' in cond_data:
            x_curr = cond_data['x_curr']
            tissue = cond_data.get('tissue')
            celltype = cond_data.get('celltype')
            stage = cond_data.get('stage')

            # CFG dropout
            if force_drop_cond:
                drop_mask = torch.ones(B, dtype=torch.bool, device=device)
            elif self.training and self.cond_dropout > 0:
                drop_mask = torch.rand(B, device=device) < self.cond_dropout
            else:
                drop_mask = None

            context = self._prepare_context(x_curr, tissue, celltype, stage, drop_mask)

        # ============================================================
        # 4. Transformer Blocks
        # ============================================================
        for block in self.blocks:
            x_emb = block(x_emb, c, context)

        # ============================================================
        # 5. Final Layer
        # ============================================================
        v_pred = self.final_layer(x_emb, c)

        return v_pred.squeeze(1)


def build_dit_cross_attn(
    input_dim: int,
    hidden_size: int = 512,
    depth: int = 12,
    num_heads: int = 8,
    n_tissues: int = 12,
    n_celltypes: int = 62,
    n_stages: int = 6,
    n_xcurr_tokens: int = 32,
    frequency_embedding_size: int = 256,
    cond_dropout: float = 0.1,
    **kwargs
) -> DiTCrossAttn:
    """Factory function for Hydra instantiation."""
    return DiTCrossAttn(
        input_dim=input_dim,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        n_tissues=n_tissues,
        n_celltypes=n_celltypes,
        n_stages=n_stages,
        n_xcurr_tokens=n_xcurr_tokens,
        frequency_embedding_size=frequency_embedding_size,
        cond_dropout=cond_dropout,
        **kwargs
    )
