"""
Set Single-Cell Variational Autoencoder (SetSCVAE)

SetSCAE 的变分版本。
将确定性的潜在空间 z 替换为变分分布 q(z|x) = N(mu, sigma^2)。
引入 KL 散度损失。

支持四种策略（同 SetSCAE）：
1. GraphSetVAE
2. ContrastiveSetVAE
3. MaskedSetVAE
4. StackSetVAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Tuple
from .set_scae import (
    BaseSetSCAE,
    GraphSetAE,
    ContrastiveSetAE,
    MaskedSetAE,
    StackSetAE
)

# =============================================================================
# VAE Mixin - 为 SetSCAE 添加 VAE 功能
# =============================================================================

class SetSCVAEMixin:
    """
    VAE 功能混入类
    
    1. 覆盖 z_mean 投影，增加 z_logvar 投影
    2. 添加 reparameterize 方法
    3. 添加 KL loss 计算
    """
    
    def _init_vae_heads(self, n_hidden: int, n_latent: int):
        """初始化 VAE 头"""
        # 均值 (替换原来的 self.z_mean)
        self.z_mean = nn.Linear(n_hidden, n_latent)
        # 对数方差
        self.z_logvar = nn.Linear(n_hidden, n_latent)
        
        # 初始化 logvar 为较小值
        nn.init.constant_(self.z_logvar.weight, 0.0)
        nn.init.constant_(self.z_logvar.bias, -2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化采样"""
        if self.training:
            # 限制 logvar 范围，防止数值不稳定
            logvar = logvar.clamp(min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度: KL(N(mu, var) || N(0, I))"""
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # 结果为 (batch, ) 或 (batch, set_size)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl

    def get_latent(
        self, 
        x_set: torch.Tensor, 
        return_context: bool = False, 
        center_idx: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        获取潜在表征 (VAE 特化版本)
        
        对于下游任务，应该返回后验分布的均值 mu (确定性)，而不是采样值 z (随机性)。
        """
        # 调用 encode 获取参数 (mu, logvar, [context])
        encoded = self.encode(x_set, center_idx=center_idx, **kwargs)
        
        # 提取 mu
        # encoded[0] 始终是 mu
        mu = encoded[0]
        
        # 处理维度差异
        # Graph/Masked/Contrastive: encode 返回 (batch, set_size, n_latent)
        # Stack: encode 返回 (batch, n_latent) [仅中心细胞]
        if mu.dim() == 3:
            mu = mu[:, center_idx, :]
            
        if return_context:
            # 尝试获取 context
            # StackSetVAE 返回 (mu, logvar, context)
            if len(encoded) >= 3:
                return encoded[2]
            # 其他策略需要额外计算或不支持
            # 为了简单，这里暂不处理其他策略的 context return，因为主要用 latent
            # 如果需要，可以调用 forward
            
        return mu


# =============================================================================
# GraphSetVAE
# =============================================================================

class GraphSetVAE(GraphSetAE, SetSCVAEMixin):
    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        # 重新初始化 VAE 头
        self._init_vae_heads(self.n_hidden, self.n_latent)
        self._init_weights() # Re-init weights for new heads

    def encode(self, x_set: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAT 编码 -> mu, logvar
        """
        # 输入投影
        h = self.input_proj(x_set)

        # GAT 编码
        for layer in self.gat_layers:
            if hasattr(layer, 'use_edge_features') and layer.use_edge_features:
                edge_features = None
                if self.use_difference:
                    edge_features = self._compute_edge_features(h)
                h = layer(h, adj, edge_features=edge_features)
            elif isinstance(layer, nn.Module): # LayerNorm, GELU
                h = layer(h)
            else:
                h = layer(h)

        # 获取潜在参数
        mu = self.z_mean(h)
        logvar = self.z_logvar(h)
        return mu, logvar

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, set_size, n_genes = x_set.shape
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # Encode -> mu, logvar
        mu_all, logvar_all = self.encode(x_set, adj=adj)
        
        # Reparameterize
        z_all = self.reparameterize(mu_all, logvar_all)
        z_center = z_all[:, center_idx, :]

        # Decode
        z_flat = z_all.view(-1, z_all.size(-1))
        lib_flat = library_size.view(-1)
        decoded = self.decode_cell(z_flat, lib_flat)

        mu_all_out = decoded['mu'].view(batch_size, set_size, -1)
        theta_all_out = decoded['theta'].view(batch_size, set_size, -1)

        # Link prediction using z (sampled) or mu (mean)? Usually z.
        z_norm = F.normalize(z_all, dim=-1)
        adj_pred = torch.bmm(z_norm, z_norm.transpose(1, 2))

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'mu_all': mu_all, # Posterior Mean
            'logvar_all': logvar_all,
            'mu_out': mu_all_out, # Recon Mean
            'theta_out': theta_all_out,
            'adj_pred': adj_pred,
        }
        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        from .losses import log_nb_positive

        # Recon Loss
        if raw_counts is not None:
            target = raw_counts
        else:
            target = torch.expm1(x_set)

        log_prob = log_nb_positive(target, outputs['mu_out'], outputs['theta_out'])
        recon_loss = -log_prob.mean()

        # Link Loss
        link_loss = torch.tensor(0.0, device=x_set.device)
        if adj is not None:
            link_loss = F.binary_cross_entropy_with_logits(outputs['adj_pred'], adj.float())

        # KL Loss
        # Compute KL for all cells in set
        kl = self.kl_divergence(outputs['mu_all'], outputs['logvar_all'])
        kl_loss = kl.mean()

        total_loss = recon_loss + self.link_pred_weight * link_loss + self.kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'link_loss': link_loss,
            'kl_loss': kl_loss
        }


# =============================================================================
# ContrastiveSetVAE
# =============================================================================

class ContrastiveSetVAE(ContrastiveSetAE, SetSCVAEMixin):
    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self._init_vae_heads(self.n_hidden, self.n_latent)
        self._init_weights()

    def encode(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle 3D input
        if x.dim() == 3:
            batch_size, set_size, n_genes = x.shape
            x_flat = x.view(-1, n_genes)
            h_flat = self.cell_encoder(x_flat)
            mu_flat = self.z_mean(h_flat)
            logvar_flat = self.z_logvar(h_flat)
            return mu_flat.view(batch_size, set_size, -1), logvar_flat.view(batch_size, set_size, -1)
        else:
            h = self.cell_encoder(x)
            return self.z_mean(h), self.z_logvar(h)

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, set_size, n_genes = x_set.shape
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        # Encode
        mu_all, logvar_all = self.encode(x_set)
        
        # Reparameterize
        z_all = self.reparameterize(mu_all, logvar_all)
        z_center = z_all[:, center_idx, :]

        # Contrastive Projection (using sampled z)
        if self.use_difference:
            proj_center = self.center_projection(z_center)
            diff_all = z_all - z_center.unsqueeze(1)
            z_with_diff = torch.cat([z_all, diff_all], dim=-1)
            proj_all = self.projection_head(z_with_diff)
        else:
            proj_all = self.projection_head(z_all)
            proj_center = proj_all[:, center_idx, :]

        # Decode
        z_flat = z_all.view(-1, z_all.size(-1))
        lib_flat = library_size.view(-1)
        decoded = self.decode_cell(z_flat, lib_flat)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'mu_all': mu_all, # posterior mean
            'logvar_all': logvar_all,
            'proj_all': proj_all,
            'proj_center': proj_center,
            'mu_out': decoded['mu'].view(batch_size, set_size, -1),
            'theta_out': decoded['theta'].view(batch_size, set_size, -1),
        }
        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        negatives: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        from .losses import log_nb_positive

        if raw_counts is not None:
            target = raw_counts
        else:
            target = torch.expm1(x_set)

        log_prob = log_nb_positive(target, outputs['mu_out'], outputs['theta_out'])
        recon_loss = -log_prob.mean()

        # Contrastive Loss
        proj_center = outputs['proj_center']
        proj_all = outputs['proj_all']
        mask = torch.ones(proj_all.size(1), dtype=torch.bool, device=proj_all.device)
        mask[center_idx] = False
        positives = proj_all[:, mask, :]
        contrastive_loss = self.info_nce_loss(proj_center, positives, negatives)

        # KL Loss
        kl = self.kl_divergence(outputs['mu_all'], outputs['logvar_all'])
        kl_loss = kl.mean()

        total_loss = recon_loss + self.contrastive_weight * contrastive_loss + self.kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'contrastive_loss': contrastive_loss,
            'kl_loss': kl_loss
        }


# =============================================================================
# MaskedSetVAE
# =============================================================================

class MaskedSetVAE(MaskedSetAE, SetSCVAEMixin):
    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self._init_vae_heads(self.n_hidden, self.n_latent)
        self._init_weights()

    def encode(
        self,
        x_set: torch.Tensor,
        center_idx: int = 0,
        gene_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, set_size, n_genes = x_set.shape
        h_set = self.input_proj(x_set)
        h_set = self._add_difference_encoding(h_set, center_idx)

        if gene_mask is not None:
            h_center = h_set[:, center_idx, :].clone()
            mask_ratio = gene_mask.float().mean(dim=-1, keepdim=True)
            h_center_masked = h_center * (1 - mask_ratio) + self.mask_token.squeeze(1) * mask_ratio
            h_set = h_set.clone()
            h_set[:, center_idx, :] = h_center_masked

        h_set = self.pos_encoding(h_set, center_idx)
        h_context = self.context_encoder(h_set)

        mu_all = self.z_mean(h_context)
        logvar_all = self.z_logvar(h_context)
        return mu_all, logvar_all

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        gene_mask: Optional[torch.Tensor] = None,
        hvg_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, set_size, n_genes = x_set.shape
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)
        
        x_center = x_set[:, center_idx, :]
        if gene_mask is None:
            gene_mask = self.create_mask(x_center, hvg_indices)

        # Same projection logic as AE
        h_set = self.input_proj(x_set)
        h_set = self._add_difference_encoding(h_set, center_idx)

        h_center = h_set[:, center_idx, :].clone()
        mask_ratio = gene_mask.float().mean(dim=-1, keepdim=True)
        h_center_masked = h_center * (1 - mask_ratio) + self.mask_token.squeeze(1) * mask_ratio
        h_set_masked = h_set.clone()
        h_set_masked[:, center_idx, :] = h_center_masked
        h_set_masked = self.pos_encoding(h_set_masked, center_idx)
        h_context = self.context_encoder(h_set_masked)

        # Prediction Head (Mask prediction)
        h_center_context = h_context[:, center_idx, :]
        pred_genes = self.prediction_head(h_center_context)

        # Latent Space (VAE)
        mu_all = self.z_mean(h_context)
        logvar_all = self.z_logvar(h_context)
        z_all = self.reparameterize(mu_all, logvar_all)
        z_center = z_all[:, center_idx, :]

        # Decode
        z_flat = z_all.view(-1, z_all.size(-1))
        lib_flat = library_size.view(-1)
        decoded = self.decode_cell(z_flat, lib_flat)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'mu_all': mu_all,
            'logvar_all': logvar_all,
            'pred_genes': pred_genes,
            'gene_mask': gene_mask,
            'mu_out': decoded['mu'].view(batch_size, set_size, -1),
            'theta_out': decoded['theta'].view(batch_size, set_size, -1),
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
        from .losses import log_nb_positive

        if raw_counts is not None:
            target = raw_counts
        else:
            target = torch.expm1(x_set)

        log_prob = log_nb_positive(target, outputs['mu_out'], outputs['theta_out'])
        recon_loss = -log_prob.mean()

        # Mask loss
        gene_mask = outputs['gene_mask']
        pred_genes = outputs['pred_genes']
        x_center = x_set[:, center_idx, :]
        mask_pred_loss = F.mse_loss(pred_genes[gene_mask], x_center[gene_mask])

        # KL Loss
        kl = self.kl_divergence(outputs['mu_all'], outputs['logvar_all'])
        kl_loss = kl.mean()

        total_loss = recon_loss + mask_pred_loss + self.kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'mask_pred_loss': mask_pred_loss,
            'kl_loss': kl_loss
        }


# =============================================================================
# StackSetVAE
# =============================================================================

class StackSetVAE(StackSetAE, SetSCVAEMixin):
    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        
        # Override VAE heads. StackSetAE has specific logic for z_mean
        if self.condition_injection == 'concat':
            dim = self.n_hidden * 2
        elif self.condition_injection in ['add', 'film']:
            dim = self.n_hidden
            
        self.z_mean = nn.Linear(dim, self.n_latent)
        self.z_logvar = nn.Linear(dim, self.n_latent)
        
        nn.init.constant_(self.z_logvar.weight, 0.0)
        nn.init.constant_(self.z_logvar.bias, -2.0)
        self._init_weights()

    def inject_condition(
        self,
        h_center: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject condition and return mu, logvar
        """
        context = context * self.context_weight

        if self.condition_injection == 'concat':
            h_combined = torch.cat([h_center, context], dim=-1)
            mu = self.z_mean(h_combined)
            logvar = self.z_logvar(h_combined)

        elif self.condition_injection == 'add':
            context_proj = self.context_proj(context)
            h_combined = h_center + context_proj
            mu = self.z_mean(h_combined)
            logvar = self.z_logvar(h_combined)

        elif self.condition_injection == 'film':
            gamma = self.film_gamma(context)
            beta = self.film_beta(context)
            h_modulated = gamma * h_center + beta
            mu = self.z_mean(h_modulated)
            logvar = self.z_logvar(h_modulated)

        return mu, logvar

    def encode(
        self,
        x_set: torch.Tensor,
        center_idx: int = 0,
        neighbor_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return mu, logvar, context
        """
        h_all = self._encode_all_cells(x_set)
        h_center, context = self._compute_context(h_all, center_idx, neighbor_mask)
        mu, logvar = self.inject_condition(h_center, context)
        return mu, logvar, context

    def forward(
        self,
        x_set: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        center_idx: int = 0,
        neighbor_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, set_size, n_genes = x_set.shape
        if library_size is None:
            library_size = torch.expm1(x_set).sum(dim=-1).clamp(min=1.0)

        h_all = self._encode_all_cells(x_set)

        # Loop for all cells (using fixed logic from SetSCAE)
        mu_all_list = []
        logvar_all_list = []
        z_all_list = []
        
        for i in range(set_size):
            _, context_i = self._compute_context(h_all, center_idx=i, neighbor_mask=neighbor_mask)
            mu_i, logvar_i = self.inject_condition(h_all[:, i, :], context_i)
            z_i = self.reparameterize(mu_i, logvar_i)
            
            mu_all_list.append(mu_i.unsqueeze(1))
            logvar_all_list.append(logvar_i.unsqueeze(1))
            z_all_list.append(z_i.unsqueeze(1))
            
        mu_all = torch.cat(mu_all_list, dim=1)
        logvar_all = torch.cat(logvar_all_list, dim=1)
        z_all = torch.cat(z_all_list, dim=1)

        z_center = z_all[:, center_idx, :]
        _, context_center = self._compute_context(h_all, center_idx, neighbor_mask)

        # Decode
        z_flat = z_all.view(-1, z_all.size(-1))
        lib_flat = library_size.view(-1)
        decoded = self.decode_cell(z_flat, lib_flat)

        outputs = {
            'z_all': z_all,
            'z_center': z_center,
            'mu_all': mu_all,
            'logvar_all': logvar_all,
            'context': context_center,
            'mu_out': decoded['mu'].view(batch_size, set_size, -1),
            'theta_out': decoded['theta'].view(batch_size, set_size, -1),
        }
        return outputs

    def compute_loss(
        self,
        x_set: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        raw_counts: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        from .losses import log_nb_positive

        if raw_counts is not None:
            target = raw_counts
        else:
            target = torch.expm1(x_set)

        log_prob = log_nb_positive(target, outputs['mu_out'], outputs['theta_out'])
        recon_loss = -log_prob.mean()

        kl = self.kl_divergence(outputs['mu_all'], outputs['logvar_all'])
        kl_loss = kl.mean()

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


# =============================================================================
# 统一接口: SetSCVAE
# =============================================================================

class SetSCVAE(nn.Module):
    """
    统一的 SetSCVAE 接口
    """

    STRATEGIES = {
        'graph': GraphSetVAE,
        'contrastive': ContrastiveSetVAE,
        'masked': MaskedSetVAE,
        'stack': StackSetVAE,
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
