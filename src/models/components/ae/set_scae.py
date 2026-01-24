"""
Set Single-Cell Autoencoder (SetSCAE)

用于建模细胞微环境的自编码器框架。
支持四种策略：
1. GraphSetAE - 基于GAT的图自编码器（带边特征）
2. ContrastiveSetAE - 差异感知的对比学习自编码器
3. MaskedSetAE - 差异感知的掩码自编码器
4. StackSetAE - 差异感知的条件嵌入方法

核心改进（2026-01-23）：
- 所有策略都加入了"中心-邻居差异"信息
- 解决同一 niche 中细胞 z_center 过于相似的问题
- 让 context 依赖于中心细胞的视角

核心思想：
- 输入是一个细胞集合（中心细胞 + 邻居细胞）
- 同时建模单细胞表达和微环境上下文
- **新增**：建模中心-邻居的差异关系
- 输出包含细胞级别和集合级别的表征

参考:
- scVI: https://docs.scvi-tools.org
- Set Transformer: https://arxiv.org/abs/1810.00825
- GAT: https://arxiv.org/abs/1710.10903
- Stack: Spatial Transcriptomics Analysis via Conditional Knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, Dict, List
from abc import ABC, abstractmethod
import math

from .scvi_ae import Encoder, Decoder, ResidualBlock


# =============================================================================
# 共享组件
# =============================================================================

class SetEncoder(nn.Module):
    """
    集合编码器 - 将细胞集合编码为上下文感知的表征

    使用 Set Transformer 风格的注意力机制
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.n_hidden = n_hidden

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_heads,
            dim_feedforward=n_hidden * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(n_hidden)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, set_size, n_genes) - 细胞集合的基因表达
            mask: (batch, set_size) - padding mask, True表示有效位置

        Returns:
            h: (batch, set_size, n_hidden) - 上下文感知的表征
        """
        # 投影到隐藏空间
        h = self.input_proj(x)  # (batch, set_size, n_hidden)

        # 创建attention mask (True表示需要mask的位置)
        if mask is not None:
            attn_mask = ~mask  # 反转，因为transformer期望True表示mask
        else:
            attn_mask = None

        # Transformer编码
        h = self.transformer(h, src_key_padding_mask=attn_mask)
        h = self.final_norm(h)

        return h


class PositionalEncoding(nn.Module):
    """位置编码 - 用于区分中心细胞和邻居细胞"""

    def __init__(self, n_hidden: int, max_len: int = 100):
        super().__init__()

        # 可学习的位置嵌入
        self.center_emb = nn.Parameter(torch.randn(1, 1, n_hidden) * 0.02)
        self.neighbor_emb = nn.Parameter(torch.randn(1, 1, n_hidden) * 0.02)

    def forward(self, x: torch.Tensor, center_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x: (batch, set_size, n_hidden)
            center_idx: 中心细胞的索引位置

        Returns:
            x + positional encoding
        """
        batch_size, set_size, n_hidden = x.shape

        # 创建位置编码
        pos_enc = self.neighbor_emb.expand(batch_size, set_size, -1).clone()
        pos_enc[:, center_idx, :] = self.center_emb.squeeze(1)

        return x + pos_enc


class CellTypeEmbedding(nn.Module):
    """细胞类型嵌入 - 可选的条件信息"""

    def __init__(self, n_types: int, n_hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(n_types, n_hidden)

    def forward(self, cell_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_types: (batch, set_size) - 细胞类型索引

        Returns:
            (batch, set_size, n_hidden) - 类型嵌入
        """
        return self.embedding(cell_types)


class DifferenceAwareAggregator(nn.Module):
    """
    差异感知聚合器 - 所有策略共享的核心组件

    核心思想：
    - 不仅聚合邻居的表征，还聚合"中心-邻居差异"
    - 这样不同中心细胞即使有相同邻居，也会得到不同的 context

    聚合信息 = f(邻居表征, 中心-邻居差异)
    """

    def __init__(
        self,
        n_hidden: int,
        aggregation: Literal['mean', 'attention', 'max'] = 'attention',
        n_heads: int = 4,
        dropout_rate: float = 0.1,
        use_difference: bool = True,  # 是否使用差异信息
    ):
        super().__init__()

        self.n_hidden = n_hidden
        self.aggregation = aggregation
        self.use_difference = use_difference

        if use_difference:
            # 差异信息投影：将 [neighbor; diff] 投影回 n_hidden
            self.diff_proj = nn.Sequential(
                nn.Linear(n_hidden * 2, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )

        if aggregation == 'attention':
            self.n_heads = n_heads
            self.head_dim = n_hidden // n_heads

            self.query = nn.Linear(n_hidden, n_hidden)
            self.key = nn.Linear(n_hidden, n_hidden)
            self.value = nn.Linear(n_hidden, n_hidden)
            self.out_proj = nn.Linear(n_hidden, n_hidden)

    def forward(
        self,
        h_center: torch.Tensor,
        h_neighbors: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_center: (batch, n_hidden) - 中心细胞表征
            h_neighbors: (batch, n_neighbors, n_hidden) - 邻居细胞表征
            neighbor_mask: (batch, n_neighbors) - 邻居掩码，True 表示有效

        Returns:
            context: (batch, n_hidden) - 差异感知的上下文向量
        """
        batch_size, n_neighbors, n_hidden = h_neighbors.shape

        # 计算差异信息
        if self.use_difference:
            # 中心与每个邻居的差异
            diff = h_neighbors - h_center.unsqueeze(1)  # (batch, n_neighbors, n_hidden)

            # 拼接邻居表征和差异
            combined = torch.cat([h_neighbors, diff], dim=-1)  # (batch, n_neighbors, 2*n_hidden)

            # 投影回 n_hidden
            h_neighbors = self.diff_proj(combined)  # (batch, n_neighbors, n_hidden)

        # 聚合
        if self.aggregation == 'mean':
            if neighbor_mask is not None:
                mask = neighbor_mask.unsqueeze(-1).float()
                context = (h_neighbors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                context = h_neighbors.mean(dim=1)

        elif self.aggregation == 'max':
            if neighbor_mask is not None:
                h_neighbors = h_neighbors.masked_fill(
                    ~neighbor_mask.unsqueeze(-1), float('-inf')
                )
            context = h_neighbors.max(dim=1)[0]

        elif self.aggregation == 'attention':
            # 多头注意力：中心细胞作为 Query，邻居作为 Key/Value
            Q = self.query(h_center)  # (batch, n_hidden)
            K = self.key(h_neighbors)  # (batch, n_neighbors, n_hidden)
            V = self.value(h_neighbors)  # (batch, n_neighbors, n_hidden)

            # 重塑为多头
            Q = Q.view(batch_size, 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, n_neighbors, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
            V = V.view(batch_size, n_neighbors, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # 注意力分数
            attn = torch.matmul(Q, K) / math.sqrt(self.head_dim)  # (batch, heads, 1, n_neighbors)

            if neighbor_mask is not None:
                mask = neighbor_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_neighbors)
                attn = attn.masked_fill(~mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)

            # 聚合
            context = torch.matmul(attn, V)  # (batch, heads, 1, head_dim)
            context = context.permute(0, 2, 1, 3).reshape(batch_size, n_hidden)
            context = self.out_proj(context)

        return context


# =============================================================================
# 基础 SetSCAE 类
# =============================================================================

class BaseSetSCAE(nn.Module, ABC):
    """
    SetSCAE 基础类

    定义通用接口和共享组件。

    重要设计决策：
    - 基类只包含真正共享的组件（decoder、NB参数）
    - encoder 由各子类自己定义，避免 DDP 未使用参数问题
    - 子类需要实现 _build_encoder() 方法

    核心改进（2026-01-23）：
    - 新增 use_difference 参数，控制是否使用中心-邻居差异信息
    - 解决同一 niche 中细胞 z_center 过于相似的问题
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        set_size: int = 16,
        use_difference: bool = True,  # 新增：是否使用差异信息
        **kwargs
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_residual = use_residual
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood
        self.set_size = set_size
        self.use_difference = use_difference  # 是否使用差异信息

        # 注意：encoder 由子类在 _build_encoder() 中定义
        # 这样可以避免 DDP 未使用参数问题

        # 单细胞解码器 (共享)
        self.cell_decoder = Decoder(
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_output=n_input,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
        )

        # NB 分布参数
        if dispersion == 'gene':
            self.log_theta = nn.Parameter(torch.zeros(n_input))
        else:
            # gene-cell: 从 latent z 预测 theta
            self.theta_decoder = nn.Sequential(
                nn.Linear(n_latent, n_input),
                nn.Softplus(),
            )

    def _init_weights(self):
        """初始化权重 - 子类应在构建完成后调用"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        编码方法 - 由子类实现

        Args:
            x: 输入数据

        Returns:
            z: 潜在表征
        """
        pass

    def decode_cell(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """解码单个细胞"""
        rho = self.cell_decoder(z)
        library_size = library_size.unsqueeze(-1) if library_size.dim() == 1 else library_size
        mu = library_size * rho

        outputs = {'rho': rho, 'mu': mu}

        if self.dispersion == 'gene':
            theta = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)
        else:
            # gene-cell: 使用 decoder 预测每个细胞的 theta
            theta = self.theta_decoder(z)
        outputs['theta'] = theta

        return outputs

    @abstractmethod
    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x_set: (batch, set_size, n_genes) - 细胞集合
            library_size: (batch, set_size) - 每个细胞的library size

        Returns:
            包含重建和潜在表征的字典
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        pass

    def get_latent(
        self,
        x_set: torch.Tensor,
        return_context: bool = False
    ) -> torch.Tensor:
        """获取潜在表征"""
        outputs = self.forward(x_set)
        if return_context:
            return outputs.get('z_context', outputs['z_center'])
        return outputs['z_center']


# =============================================================================
# 方案一: GraphSetAE - 基于GAT的图自编码器
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    图注意力层（支持边特征）

    改进（2026-01-23）：
    - 新增 edge_features 支持
    - 边特征 = 中心-邻居差异，让注意力权重考虑节点间关系
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        use_edge_features: bool = True,  # 新增：是否使用边特征
    ):
        super().__init__()

        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat
        self.use_edge_features = use_edge_features

        # 每个头的维度
        self.head_dim = out_features // n_heads if concat else out_features

        # 线性变换
        self.W = nn.Linear(in_features, self.head_dim * n_heads, bias=False)

        # 注意力参数
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        # 边特征投影（新增）
        if use_edge_features:
            self.a_edge = nn.Parameter(torch.zeros(n_heads, self.head_dim))
            self.edge_proj = nn.Linear(in_features, self.head_dim * n_heads, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.a_edge)
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,  # 新增：边特征
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (batch, n_nodes, n_nodes) 邻接矩阵，None表示全连接
            edge_features: (batch, n_nodes, n_nodes, in_features) 边特征（可选）

        Returns:
            (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.shape

        # 线性变换: (batch, n_nodes, n_heads * head_dim)
        h = self.W(x)
        # 重塑: (batch, n_nodes, n_heads, head_dim)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # 计算注意力分数
        # (batch, n_nodes, n_heads)
        attn_src = (h * self.a_src).sum(dim=-1)
        attn_dst = (h * self.a_dst).sum(dim=-1)

        # (batch, n_nodes, n_nodes, n_heads)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)

        # 加入边特征的贡献（新增）
        if self.use_edge_features and edge_features is not None:
            # edge_features: (batch, n_nodes, n_nodes, in_features)
            # 投影: (batch, n_nodes, n_nodes, n_heads * head_dim)
            e = self.edge_proj(edge_features)
            e = e.view(batch_size, n_nodes, n_nodes, self.n_heads, self.head_dim)
            # (batch, n_nodes, n_nodes, n_heads)
            attn_edge = (e * self.a_edge).sum(dim=-1)
            attn = attn + attn_edge

        attn = self.leaky_relu(attn)

        # 应用邻接矩阵mask
        if adj is not None:
            # adj: (batch, n_nodes, n_nodes) -> (batch, n_nodes, n_nodes, 1)
            mask = adj.unsqueeze(-1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        # 聚合: (batch, n_nodes, n_heads, head_dim)
        # h: (batch, n_nodes, n_heads, head_dim)
        # attn: (batch, n_nodes, n_nodes, n_heads)
        h = h.permute(0, 2, 1, 3)  # (batch, n_heads, n_nodes, head_dim)
        attn = attn.permute(0, 3, 1, 2)  # (batch, n_heads, n_nodes, n_nodes)
        out = torch.matmul(attn, h)  # (batch, n_heads, n_nodes, head_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, n_nodes, n_heads, head_dim)

        if self.concat:
            out = out.reshape(batch_size, n_nodes, -1)
        else:
            out = out.mean(dim=2)

        return out


class GraphSetAE(BaseSetSCAE):
    """
    方案一: 基于GAT的图自编码器（带边特征）

    通过图注意力网络在编码阶段引入微环境信息。
    双重解码器：特征重建 + 链路预测

    改进（2026-01-23）：
    - 新增边特征支持：edge_ij = h_i - h_j（节点差异）
    - 让 GAT 注意力权重考虑节点间关系，而不仅仅是节点自身特征
    - 解决同一 niche 中细胞 z 过于相似的问题

    编码器架构：input_proj + GAT layers (with edge features)
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        n_gat_layers: int = 2,
        n_heads: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        set_size: int = 16,
        link_pred_weight: float = 0.1,
        use_difference: bool = True,  # 新增：是否使用差异作为边特征
        **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            set_size=set_size,
            use_difference=use_difference,
            **kwargs
        )

        self.link_pred_weight = link_pred_weight

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # GAT 层（支持边特征）
        self.gat_layers = nn.ModuleList()
        for i in range(n_gat_layers):
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=n_hidden,
                    out_features=n_hidden,
                    n_heads=n_heads,
                    dropout=dropout_rate,
                    concat=True,
                    use_edge_features=use_difference,  # 使用差异作为边特征
                )
            )
            self.gat_layers.append(nn.LayerNorm(n_hidden))
            self.gat_layers.append(nn.GELU())

        # 潜在空间投影
        self.z_mean = nn.Linear(n_hidden, n_latent)

        self._init_weights()

    def _compute_edge_features(self, h: torch.Tensor) -> torch.Tensor:
        """
        计算边特征（节点差异）

        Args:
            h: (batch, n_nodes, n_hidden) 节点特征

        Returns:
            edge_features: (batch, n_nodes, n_nodes, n_hidden)
            edge_ij = h_i - h_j（从 j 指向 i 的边特征）
        """
        batch_size, n_nodes, n_hidden = h.shape
        # h_i: (batch, n_nodes, 1, n_hidden)
        # h_j: (batch, 1, n_nodes, n_hidden)
        # edge_ij = h_i - h_j: (batch, n_nodes, n_nodes, n_hidden)
        edge_features = h.unsqueeze(2) - h.unsqueeze(1)
        return edge_features

    def encode(
        self,
        x_set: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        GAT 编码（带边特征）

        Args:
            x_set: (batch, set_size, n_genes)
            adj: (batch, set_size, set_size) 邻接矩阵

        Returns:
            z_all: (batch, set_size, n_latent)
        """
        # 输入投影
        h = self.input_proj(x_set)  # (batch, set_size, n_hidden)

        # GAT 编码（每层更新边特征）
        for layer in self.gat_layers:
            if isinstance(layer, GraphAttentionLayer):
                # 计算边特征（如果启用）
                edge_features = None
                if self.use_difference:
                    edge_features = self._compute_edge_features(h)
                h = layer(h, adj, edge_features=edge_features)
            else:
                h = layer(h)

        # 获取潜在表征
        z_all = self.z_mean(h)  # (batch, set_size, n_latent)
        return z_all

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_set: (batch, set_size, n_genes)
            library_size: (batch, set_size)
            adj: (batch, set_size, set_size) 邻接矩阵
            center_idx: 中心细胞索引
        """
        batch_size, set_size, n_genes = x_set.shape

        # 估算 library size
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # GAT 编码（带边特征）
        z_all = self.encode(x_set, adj=adj)  # (batch, set_size, n_latent)
        z_center = z_all[:, center_idx, :]  # (batch, n_latent)

        # 解码所有细胞 (不仅仅是中心细胞)
        z_flat = z_all.view(-1, z_all.size(-1))  # (batch * set_size, n_latent)
        lib_flat = library_size.view(-1)  # (batch * set_size,)
        decoded = self.decode_cell(z_flat, lib_flat)

        # reshape 回 (batch, set_size, n_genes)
        mu_all = decoded['mu'].view(batch_size, set_size, -1)
        theta_all = decoded['theta'].view(batch_size, set_size, -1)

        # 链路预测 (内积)
        z_norm = F.normalize(z_all, dim=-1)
        adj_pred = torch.bmm(z_norm, z_norm.transpose(1, 2))

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'mu_all': mu_all,
            'theta_all': theta_all,
            'mu': mu_all[:, center_idx, :],  # 兼容旧接口
            'theta': theta_all[:, center_idx, :],
            'adj_pred': adj_pred,
        }

        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算损失 (重建所有细胞)"""
        from .losses import log_nb_positive

        # 重建损失 (所有细胞)
        if raw_counts is not None:
            target = raw_counts  # (batch, set_size, n_genes)
        else:
            target = torch.expm1(x_set)  # (batch, set_size, n_genes)

        mu_all = outputs['mu_all']  # (batch, set_size, n_genes)
        theta_all = outputs['theta_all']  # (batch, set_size, n_genes)

        # 计算所有细胞的 NB log prob
        log_prob = log_nb_positive(target, mu_all, theta_all)  # (batch, set_size, n_genes)
        recon_loss = -log_prob.mean()

        # 链路预测损失
        link_loss = torch.tensor(0.0, device=x_set.device)
        if adj is not None:
            adj_pred = outputs['adj_pred']
            link_loss = F.binary_cross_entropy_with_logits(
                adj_pred, adj.float()
            )

        total_loss = recon_loss + self.link_pred_weight * link_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'link_loss': link_loss,
        }


# =============================================================================
# 方案二: ContrastiveSetAE - 对比学习自编码器
# =============================================================================

class ContrastiveSetAE(BaseSetSCAE):
    """
    方案二: 差异感知的对比学习自编码器

    通过 InfoNCE 损失强制 Latent 空间在局部邻域内保持平滑。
    正样本：空间邻居细胞
    负样本：随机采样的非邻居细胞

    改进（2026-01-23）：
    - 新增差异感知的投影头
    - 对比学习时考虑 [z_neighbor; z_center - z_neighbor]
    - 让正样本分数不仅依赖邻居自身，还依赖与中心的关系

    编码器架构：标准 scVI-style Encoder + 差异感知投影头
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        set_size: int = 16,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
        n_negatives: int = 64,
        use_difference: bool = True,  # 新增：是否使用差异感知
        **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            set_size=set_size,
            use_difference=use_difference,
            **kwargs
        )

        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.n_negatives = n_negatives

        # 单细胞编码器 (ContrastiveSetAE 使用标准 scVI-style encoder)
        self.cell_encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
        )

        # 潜在空间投影
        self.z_mean = nn.Linear(n_hidden, n_latent)

        # 投影头 (用于对比学习)
        if use_difference:
            # 差异感知的投影头：输入 [z; diff]
            self.projection_head = nn.Sequential(
                nn.Linear(n_latent * 2, n_hidden),
                nn.GELU(),
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, n_latent),
            )
            # 中心细胞的投影（不需要差异）
            self.center_projection = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.GELU(),
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, n_latent),
            )
        else:
            # 原始投影头
            self.projection_head = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.GELU(),
                nn.Linear(n_hidden, n_latent),
            )

        self._init_weights()

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        编码单个细胞

        Args:
            x: (batch, n_genes) 或 (batch, set_size, n_genes)

        Returns:
            z: 潜在表征
        """
        # 处理 3D 输入
        if x.dim() == 3:
            batch_size, set_size, n_genes = x.shape
            x_flat = x.view(-1, n_genes)
            h_flat = self.cell_encoder(x_flat)
            z_flat = self.z_mean(h_flat)
            return z_flat.view(batch_size, set_size, -1)
        else:
            h = self.cell_encoder(x)
            return self.z_mean(h)

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_set: (batch, set_size, n_genes) - 中心细胞 + 邻居
            library_size: (batch, set_size)
            center_idx: 中心细胞索引
        """
        batch_size, set_size, n_genes = x_set.shape

        # 估算 library size
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # 编码所有细胞
        x_flat = x_set.view(-1, n_genes)
        h_flat = self.cell_encoder(x_flat)
        z_flat = self.z_mean(h_flat)
        z_all = z_flat.view(batch_size, set_size, -1)

        # 中心细胞
        z_center = z_all[:, center_idx, :]

        # 投影到对比学习空间（差异感知）
        if self.use_difference:
            # 中心细胞的投影
            proj_center = self.center_projection(z_center)  # (batch, n_latent)

            # 邻居的差异感知投影
            # z_neighbors: (batch, set_size, n_latent)
            # diff: z_neighbor - z_center
            diff_all = z_all - z_center.unsqueeze(1)  # (batch, set_size, n_latent)
            # 拼接 [z; diff]
            z_with_diff = torch.cat([z_all, diff_all], dim=-1)  # (batch, set_size, 2*n_latent)
            proj_all = self.projection_head(z_with_diff)  # (batch, set_size, n_latent)
        else:
            proj_all = self.projection_head(z_all)
            proj_center = proj_all[:, center_idx, :]

        # 解码所有细胞 (不仅仅是中心细胞)
        lib_flat = library_size.view(-1)  # (batch * set_size,)
        decoded = self.decode_cell(z_flat, lib_flat)

        # reshape 回 (batch, set_size, n_genes)
        mu_all = decoded['mu'].view(batch_size, set_size, -1)
        theta_all = decoded['theta'].view(batch_size, set_size, -1)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'proj_all': proj_all,
            'proj_center': proj_center,
            'mu_all': mu_all,
            'theta_all': theta_all,
            'mu': mu_all[:, center_idx, :],  # 兼容旧接口
            'theta': theta_all[:, center_idx, :],
        }

        return outputs

    def info_nce_loss(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        InfoNCE 损失 (修正版)

        使用 batch 内其他样本作为负样本 (in-batch negatives)

        Args:
            anchor: (batch, n_latent) - 中心细胞
            positives: (batch, n_pos, n_latent) - 正样本（邻居）
            negatives: (batch, n_neg, n_latent) - 额外负样本 (可选)
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # L2 归一化
        anchor = F.normalize(anchor, dim=-1)  # (batch, n_latent)
        positives = F.normalize(positives, dim=-1)  # (batch, n_pos, n_latent)

        # 正样本相似度: 使用所有正样本的平均
        # (batch, n_pos)
        pos_sim = torch.bmm(
            positives, anchor.unsqueeze(-1)
        ).squeeze(-1) / self.temperature
        # 取平均作为正样本得分: (batch,)
        pos_score = pos_sim.mean(dim=-1, keepdim=True)

        # 负样本: 使用 batch 内其他 anchor 作为负样本
        # anchor: (batch, n_latent)
        # 计算 anchor 与所有其他 anchor 的相似度: (batch, batch)
        neg_sim_batch = torch.mm(anchor, anchor.t()) / self.temperature
        # 移除对角线 (自己与自己)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim_batch = neg_sim_batch.masked_fill(mask, float('-inf'))

        # 如果有额外负样本
        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim_extra = torch.bmm(
                negatives, anchor.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            # 合并: (batch, batch-1 + n_neg)
            all_neg = torch.cat([neg_sim_batch, neg_sim_extra], dim=-1)
        else:
            all_neg = neg_sim_batch

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # 合并正负样本: (batch, 1 + n_neg)
        all_sim = torch.cat([pos_score, all_neg], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        loss = F.cross_entropy(all_sim, labels)

        return loss

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        negatives: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算损失 (重建所有细胞)"""
        from .losses import log_nb_positive

        # 重建损失 (所有细胞)
        if raw_counts is not None:
            target = raw_counts  # (batch, set_size, n_genes)
        else:
            target = torch.expm1(x_set)  # (batch, set_size, n_genes)

        mu_all = outputs['mu_all']  # (batch, set_size, n_genes)
        theta_all = outputs['theta_all']  # (batch, set_size, n_genes)

        log_prob = log_nb_positive(target, mu_all, theta_all)
        recon_loss = -log_prob.mean()

        # 对比损失
        proj_center = outputs['proj_center']
        proj_all = outputs['proj_all']

        # 邻居作为正样本 (排除中心细胞)
        mask = torch.ones(proj_all.size(1), dtype=torch.bool, device=proj_all.device)
        mask[center_idx] = False
        positives = proj_all[:, mask, :]

        contrastive_loss = self.info_nce_loss(proj_center, positives, negatives)

        total_loss = recon_loss + self.contrastive_weight * contrastive_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'contrastive_loss': contrastive_loss,
        }


# =============================================================================
# 方案三: MaskedSetAE - 掩码自编码器
# =============================================================================

class MaskedSetAE(BaseSetSCAE):
    """
    方案三: 差异感知的掩码自编码器

    借鉴 MAE/scGPT 的思路，通过预测被掩码的信息来学习微环境。
    - 输入：中心细胞 + 邻居细胞
    - 掩码：随机遮盖中心细胞的部分基因
    - 预测：利用邻居信息预测被遮盖的基因

    改进（2026-01-23）：
    - 新增差异感知的相对位置编码
    - 在 Transformer 输入中加入 center-neighbor 差异信息
    - 让上下文编码更好地捕捉中心细胞与邻居的关系

    编码器架构：input_proj + 差异编码 + Transformer context_encoder
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        n_context_layers: int = 2,
        n_heads: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        set_size: int = 16,
        mask_ratio: float = 0.3,
        mask_strategy: Literal['random', 'hvg', 'block'] = 'random',
        use_difference: bool = True,  # 新增：是否使用差异感知
        **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            set_size=set_size,
            use_difference=use_difference,
            **kwargs
        )

        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

        # 可学习的 MASK token
        self.mask_token = nn.Parameter(torch.randn(1, 1, n_hidden) * 0.02)

        # 位置编码
        self.pos_encoding = PositionalEncoding(n_hidden)

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 差异编码（新增）
        if use_difference:
            # 将差异信息融入表征
            self.diff_encoder = nn.Sequential(
                nn.Linear(n_hidden * 2, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )

        # 上下文 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_heads,
            dim_feedforward=n_hidden * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_context_layers
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_input),
        )

        # 潜在空间投影
        self.z_mean = nn.Linear(n_hidden, n_latent)

        self._init_weights()

    def _add_difference_encoding(
        self,
        h_set: torch.Tensor,
        center_idx: int = 0
    ) -> torch.Tensor:
        """
        添加差异编码

        Args:
            h_set: (batch, set_size, n_hidden) - 投影后的表征
            center_idx: 中心细胞索引

        Returns:
            h_set_with_diff: (batch, set_size, n_hidden) - 加入差异信息的表征
        """
        if not self.use_difference:
            return h_set

        batch_size, set_size, n_hidden = h_set.shape

        # 获取中心细胞表征
        h_center = h_set[:, center_idx, :]  # (batch, n_hidden)

        # 计算每个细胞与中心的差异
        diff = h_set - h_center.unsqueeze(1)  # (batch, set_size, n_hidden)

        # 拼接原始表征和差异
        h_with_diff = torch.cat([h_set, diff], dim=-1)  # (batch, set_size, 2*n_hidden)

        # 投影回 n_hidden
        h_set_with_diff = self.diff_encoder(h_with_diff)  # (batch, set_size, n_hidden)

        return h_set_with_diff

    def encode(
        self,
        x_set: torch.Tensor,
        center_idx: int = 0,
        gene_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        差异感知的上下文编码

        Args:
            x_set: (batch, set_size, n_genes)
            center_idx: 中心细胞索引
            gene_mask: (batch, n_genes) 基因掩码

        Returns:
            z_all: (batch, set_size, n_latent)
        """
        batch_size, set_size, n_genes = x_set.shape

        # 投影所有细胞
        h_set = self.input_proj(x_set)  # (batch, set_size, n_hidden)

        # 添加差异编码（新增）
        h_set = self._add_difference_encoding(h_set, center_idx)

        # 如果有掩码，对中心细胞应用掩码
        if gene_mask is not None:
            h_center = h_set[:, center_idx, :].clone()
            mask_ratio = gene_mask.float().mean(dim=-1, keepdim=True)
            h_center_masked = h_center * (1 - mask_ratio) + self.mask_token.squeeze(1) * mask_ratio
            h_set = h_set.clone()
            h_set[:, center_idx, :] = h_center_masked

        # 添加位置编码
        h_set = self.pos_encoding(h_set, center_idx)

        # 上下文编码
        h_context = self.context_encoder(h_set)

        # 获取潜在表征
        z_all = self.z_mean(h_context)
        return z_all

    def create_mask(
        self,
        x: torch.Tensor,
        hvg_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        创建基因掩码

        Args:
            x: (batch, n_genes)
            hvg_indices: 高变基因索引

        Returns:
            mask: (batch, n_genes) - True 表示被掩码
        """
        batch_size, n_genes = x.shape
        device = x.device

        if self.mask_strategy == 'random':
            # 随机掩码
            mask = torch.rand(batch_size, n_genes, device=device) < self.mask_ratio

        elif self.mask_strategy == 'hvg' and hvg_indices is not None:
            # 只掩码高变基因
            mask = torch.zeros(batch_size, n_genes, dtype=torch.bool, device=device)
            n_hvg = len(hvg_indices)
            n_mask = int(n_hvg * self.mask_ratio)
            for i in range(batch_size):
                perm = torch.randperm(n_hvg, device=device)[:n_mask]
                mask[i, hvg_indices[perm]] = True

        elif self.mask_strategy == 'block':
            # 块状掩码 (连续基因)
            mask = torch.zeros(batch_size, n_genes, dtype=torch.bool, device=device)
            block_size = int(n_genes * self.mask_ratio)
            for i in range(batch_size):
                start = torch.randint(0, n_genes - block_size, (1,)).item()
                mask[i, start:start + block_size] = True

        else:
            mask = torch.rand(batch_size, n_genes, device=device) < self.mask_ratio

        return mask

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        gene_mask: Optional[torch.Tensor] = None,
        hvg_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_set: (batch, set_size, n_genes)
            library_size: (batch, set_size)
            center_idx: 中心细胞索引
            gene_mask: (batch, n_genes) 预定义的掩码
            hvg_indices: 高变基因索引
        """
        batch_size, set_size, n_genes = x_set.shape

        # 估算 library size
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # 获取中心细胞
        x_center = x_set[:, center_idx, :]  # (batch, n_genes)

        # 创建或使用掩码
        if gene_mask is None:
            gene_mask = self.create_mask(x_center, hvg_indices)

        # 投影所有细胞
        h_set = self.input_proj(x_set)  # (batch, set_size, n_hidden)

        # 添加差异编码（新增）
        h_set = self._add_difference_encoding(h_set, center_idx)

        # 对中心细胞应用掩码: 用 mask_token 替换
        h_center = h_set[:, center_idx, :].clone()  # (batch, n_hidden)
        # 使用可学习的 mask_token 混合
        mask_ratio = gene_mask.float().mean(dim=-1, keepdim=True)  # (batch, 1)
        h_center_masked = h_center * (1 - mask_ratio) + self.mask_token.squeeze(1) * mask_ratio

        # 替换中心细胞的表征
        h_set_masked = h_set.clone()
        h_set_masked[:, center_idx, :] = h_center_masked

        # 添加位置编码
        h_set_masked = self.pos_encoding(h_set_masked, center_idx)

        # 上下文编码
        h_context = self.context_encoder(h_set_masked)

        # 获取中心细胞的上下文感知表征
        h_center_context = h_context[:, center_idx, :]

        # 预测被掩码的基因
        pred_genes = self.prediction_head(h_center_context)

        # 获取所有细胞的潜在表征
        z_all = self.z_mean(h_context)  # (batch, set_size, n_latent)
        z_center = z_all[:, center_idx, :]

        # 解码所有细胞 (不仅仅是中心细胞)
        z_flat = z_all.view(-1, z_all.size(-1))  # (batch * set_size, n_latent)
        lib_flat = library_size.view(-1)  # (batch * set_size,)
        decoded = self.decode_cell(z_flat, lib_flat)

        # reshape 回 (batch, set_size, n_genes)
        mu_all = decoded['mu'].view(batch_size, set_size, -1)
        theta_all = decoded['theta'].view(batch_size, set_size, -1)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'pred_genes': pred_genes,
            'gene_mask': gene_mask,
            'h_context': h_context,
            'mu_all': mu_all,
            'theta_all': theta_all,
            'mu': mu_all[:, center_idx, :],  # 兼容旧接口
            'theta': theta_all[:, center_idx, :],
        }

        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算损失 (重建所有细胞)"""
        from .losses import log_nb_positive

        # 重建损失 (所有细胞)
        if raw_counts is not None:
            target = raw_counts  # (batch, set_size, n_genes)
        else:
            target = torch.expm1(x_set)  # (batch, set_size, n_genes)

        mu_all = outputs['mu_all']  # (batch, set_size, n_genes)
        theta_all = outputs['theta_all']  # (batch, set_size, n_genes)

        log_prob = log_nb_positive(target, mu_all, theta_all)
        recon_loss = -log_prob.mean()

        # 掩码预测损失 (只计算中心细胞被掩码的基因)
        gene_mask = outputs['gene_mask']
        pred_genes = outputs['pred_genes']
        x_center = x_set[:, center_idx, :]

        # 在 log1p 空间计算 MSE
        mask_pred_loss = F.mse_loss(
            pred_genes[gene_mask],
            x_center[gene_mask]
        )

        total_loss = recon_loss + mask_pred_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'mask_pred_loss': mask_pred_loss,
        }


# =============================================================================
# 方案四: StackSetAE - 基于Stack论文的条件嵌入方法
# =============================================================================

class StackSetAE(BaseSetSCAE):
    """
    方案四: 差异感知的条件嵌入自编码器

    核心思想（参考 Stack 论文）：
    - 使用条件嵌入 (condition embedding) 来编码微环境信息
    - 通过 concatenation 或 addition 将条件信息注入 latent space
    - 简单、统一、无需为不同条件设计不同架构

    改进（2026-01-23）：
    - 使用 DifferenceAwareAggregator 聚合邻居信息
    - context = f(邻居表征, 中心-邻居差异)
    - 解决同一 niche 中细胞 z_center 过于相似的问题

    工作流程：
    1. 对邻居细胞进行**差异感知聚合**，得到微环境上下文向量
    2. 将上下文向量作为条件信息注入到中心细胞的编码过程
    3. 条件注入方式：concat, add, 或 film (Feature-wise Linear Modulation)

    优势：
    - 架构简单，易于理解和调试
    - 与标准 scVI 编码器兼容
    - 灵活的条件注入方式
    - **差异感知**：不同中心细胞得到不同的 context
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        set_size: int = 16,
        # Stack 特有参数
        context_aggregation: Literal['mean', 'attention', 'max'] = 'attention',
        condition_injection: Literal['concat', 'add', 'film'] = 'concat',
        context_weight: float = 1.0,
        n_context_heads: int = 4,
        use_difference: bool = True,  # 新增：是否使用差异感知
        **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            set_size=set_size,
            use_difference=use_difference,
            **kwargs
        )

        self.context_aggregation = context_aggregation
        self.condition_injection = condition_injection
        self.context_weight = context_weight

        # 单细胞编码器 (用于编码所有细胞)
        self.cell_encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
        )

        # 使用差异感知聚合器（新增）
        self.context_aggregator = DifferenceAwareAggregator(
            n_hidden=n_hidden,
            aggregation=context_aggregation,
            n_heads=n_context_heads,
            dropout_rate=dropout_rate,
            use_difference=use_difference,
        )

        # 条件注入模块
        if condition_injection == 'concat':
            # 拼接后投影
            self.z_mean = nn.Linear(n_hidden * 2, n_latent)
        elif condition_injection == 'add':
            # 直接相加
            self.z_mean = nn.Linear(n_hidden, n_latent)
            self.context_proj = nn.Linear(n_hidden, n_hidden)
        elif condition_injection == 'film':
            # Feature-wise Linear Modulation
            self.z_mean = nn.Linear(n_hidden, n_latent)
            self.film_gamma = nn.Linear(n_hidden, n_hidden)
            self.film_beta = nn.Linear(n_hidden, n_hidden)

        self._init_weights()

    def inject_condition(
        self,
        h_center: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        将上下文条件注入到中心细胞表征

        Args:
            h_center: (batch, n_hidden) - 中心细胞的隐藏表征
            context: (batch, n_hidden) - 上下文向量

        Returns:
            z: (batch, n_latent)
        """
        context = context * self.context_weight

        if self.condition_injection == 'concat':
            # 拼接
            h_combined = torch.cat([h_center, context], dim=-1)
            z = self.z_mean(h_combined)

        elif self.condition_injection == 'add':
            # 相加
            context_proj = self.context_proj(context)
            h_combined = h_center + context_proj
            z = self.z_mean(h_combined)

        elif self.condition_injection == 'film':
            # Feature-wise Linear Modulation: gamma * h + beta
            gamma = self.film_gamma(context)
            beta = self.film_beta(context)
            h_modulated = gamma * h_center + beta
            z = self.z_mean(h_modulated)

        return z

    def _encode_all_cells(
        self,
        x_set: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码所有细胞到隐藏空间

        Args:
            x_set: (batch, set_size, n_genes)

        Returns:
            h_all: (batch, set_size, n_hidden)
        """
        batch_size, set_size, n_genes = x_set.shape
        x_flat = x_set.view(-1, n_genes)
        h_flat = self.cell_encoder(x_flat)
        h_all = h_flat.view(batch_size, set_size, -1)
        return h_all

    def _compute_context(
        self,
        h_all: torch.Tensor,
        center_idx: int = 0,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算差异感知的上下文向量

        Args:
            h_all: (batch, set_size, n_hidden)
            center_idx: 中心细胞索引
            neighbor_mask: (batch, set_size) 邻居掩码

        Returns:
            h_center: (batch, n_hidden)
            context: (batch, n_hidden) - 差异感知的上下文
        """
        batch_size, set_size, _ = h_all.shape

        # 分离中心细胞和邻居
        h_center = h_all[:, center_idx, :]  # (batch, n_hidden)

        # 获取邻居（排除中心细胞）
        neighbor_indices = [i for i in range(set_size) if i != center_idx]
        h_neighbors = h_all[:, neighbor_indices, :]  # (batch, n_neighbors, n_hidden)

        # 处理邻居掩码
        if neighbor_mask is not None:
            neighbor_mask_filtered = neighbor_mask[:, neighbor_indices]
        else:
            neighbor_mask_filtered = None

        # 使用差异感知聚合器
        context = self.context_aggregator(h_center, h_neighbors, neighbor_mask_filtered)

        return h_center, context

    def encode(
        self,
        x_set: torch.Tensor,
        center_idx: int = 0,
        neighbor_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码细胞集合

        Args:
            x_set: (batch, set_size, n_genes)
            center_idx: 中心细胞索引
            neighbor_mask: (batch, set_size) 邻居掩码

        Returns:
            z_center: (batch, n_latent) - 条件化的中心细胞潜在表征
            context: (batch, n_hidden) - 上下文向量
        """
        # 编码所有细胞
        h_all = self._encode_all_cells(x_set)

        # 计算上下文
        h_center, context = self._compute_context(h_all, center_idx, neighbor_mask)

        # 条件注入得到 z
        z_center = self.inject_condition(h_center, context)

        return z_center, context

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        neighbor_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_set: (batch, set_size, n_genes)
            library_size: (batch, set_size)
            center_idx: 中心细胞索引
            neighbor_mask: (batch, set_size) 邻居掩码
        """
        batch_size, set_size, n_genes = x_set.shape

        # 估算 library size
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # 编码所有细胞（只调用一次 cell_encoder）
        h_all = self._encode_all_cells(x_set)

        # 计算差异感知的上下文
        h_center, context = self._compute_context(h_all, center_idx, neighbor_mask)

        # 为所有细胞生成 z（使用相同的上下文）
        z_all_list = []
        for i in range(set_size):
            z_i = self.inject_condition(h_all[:, i, :], context)
            z_all_list.append(z_i.unsqueeze(1))
        z_all = torch.cat(z_all_list, dim=1)  # (batch, set_size, n_latent)

        z_center = z_all[:, center_idx, :]

        # 解码所有细胞
        z_flat = z_all.view(-1, z_all.size(-1))
        lib_flat = library_size.view(-1)
        decoded = self.decode_cell(z_flat, lib_flat)

        # reshape 回 (batch, set_size, n_genes)
        mu_all = decoded['mu'].view(batch_size, set_size, -1)
        theta_all = decoded['theta'].view(batch_size, set_size, -1)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'context': context,
            'mu_all': mu_all,
            'theta_all': theta_all,
            'mu': mu_all[:, center_idx, :],
            'theta': theta_all[:, center_idx, :],
        }

        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        from .losses import log_nb_positive

        # 重建损失 (所有细胞)
        if raw_counts is not None:
            target = raw_counts
        else:
            target = torch.expm1(x_set)

        mu_all = outputs['mu_all']
        theta_all = outputs['theta_all']

        log_prob = log_nb_positive(target, mu_all, theta_all)
        recon_loss = -log_prob.mean()

        return {
            'loss': recon_loss,
            'recon_loss': recon_loss,
        }


# =============================================================================
# 统一接口: SetSCAE
# =============================================================================

class SetSCAE(nn.Module):
    """
    统一的 SetSCAE 接口

    通过 strategy 参数选择不同的实现方案：
    - 'graph': GraphSetAE (GAT-based)
    - 'contrastive': ContrastiveSetAE
    - 'masked': MaskedSetAE
    - 'stack': StackSetAE (条件嵌入方法)
    """

    STRATEGIES = {
        'graph': GraphSetAE,
        'contrastive': ContrastiveSetAE,
        'masked': MaskedSetAE,
        'stack': StackSetAE,
    }

    def __init__(
        self,
        strategy: Literal['graph', 'contrastive', 'masked', 'stack'] = 'contrastive',
        **kwargs
    ):
        super().__init__()

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(self.STRATEGIES.keys())}"
            )

        self.strategy = strategy
        self.model = self.STRATEGIES[strategy](**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.model.compute_loss(*args, **kwargs)

    def get_latent(self, *args, **kwargs):
        return self.model.get_latent(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def decode_cell(self, *args, **kwargs):
        return self.model.decode_cell(*args, **kwargs)
