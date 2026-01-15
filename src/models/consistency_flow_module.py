"""
时间链一致性 Flow Module

Loss 设计：
- L_step: 逐步监督（相邻对 Flow loss）
- L_skip: 跨步监督（跨越多步的 Flow loss，学习任意时间跨度）
- L_consistency: 真正的一致性约束（直接预测 ≈ 链式预测）

L_total = L_step + λ_skip * L_skip + λ_cons * L_consistency
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.components.flow.rectified_flow import RectifiedFlow


class ConsistencyFlowLitModule(LightningModule):
    """
    支持时间链一致性训练的 Flow Module。

    核心特性：
    1. 支持序列输入（而非仅细胞对）
    2. 混合一致性 Loss
    3. 支持不同时间跨度的预测
    """

    def __init__(
        self,
        net: nn.Module,
        flow_type: str = "rectified_flow",
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        # Loss 权重
        lambda_skip: float = 0.5,      # 跨步监督权重
        lambda_cons: float = 0.1,      # 一致性约束权重
        # 采样参数
        sample_steps: int = 10,        # 采样步数
        cons_every_n_steps: int = 10,  # 每 N 步计算一次一致性 loss（加速训练）
        # 以下参数由 train.py 使用
        mode: Optional[str] = None,
        ae_ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        # 创建 Flow 模型
        if flow_type == "rectified_flow":
            self.flow = RectifiedFlow(backbone=net)
        else:
            raise ValueError(f"不支持的 Flow 类型: {flow_type}")

        self.lambda_skip = lambda_skip
        self.lambda_cons = lambda_cons
        self.sample_steps = sample_steps
        self.cons_every_n_steps = cons_every_n_steps

        if compile and hasattr(torch, "compile"):
            self.flow = torch.compile(self.flow)

    def forward(self, x_seq: torch.Tensor, time_seq: torch.Tensor, seq_len: torch.Tensor):
        """前向传播，计算混合一致性 Loss"""
        return self.model_step({'x_seq': x_seq, 'time_seq': time_seq, 'seq_len': seq_len})

    def model_step(
        self, batch: Dict[str, Any], compute_cons: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        计算混合 Loss。

        输入 batch:
        - x_seq: [B, L, D] 细胞序列
        - time_seq: [B, L] 时间序列
        - seq_len: [B] 实际序列长度
        - stage: [B] 发育阶段（可选）

        Args:
            compute_cons: 是否计算一致性 loss（可跳过以加速）
        """
        x_seq = batch['x_seq']
        time_seq = batch['time_seq']
        seq_len = batch['seq_len']
        stage = batch.get('stage', None)

        device = x_seq.device

        # 1. L_step: 逐步监督
        loss_step = self._compute_step_loss(x_seq, time_seq, seq_len, stage)

        # 2. L_skip: 跨步监督
        loss_skip = self._compute_skip_loss(x_seq, time_seq, seq_len, stage)

        # 3. L_consistency: 真正的一致性约束（可选，每 N 步才计算）
        if compute_cons and self.lambda_cons > 0:
            loss_cons = self._compute_consistency_loss(x_seq, time_seq, seq_len, stage)
        else:
            loss_cons = torch.tensor(0.0, device=device)

        # 混合 Loss
        loss_total = loss_step + self.lambda_skip * loss_skip + self.lambda_cons * loss_cons

        return {
            'loss': loss_total,
            'loss_step': loss_step,
            'loss_skip': loss_skip,
            'loss_cons': loss_cons,
        }

    def _compute_step_loss(
        self,
        x_seq: torch.Tensor,
        time_seq: torch.Tensor,
        seq_len: torch.Tensor,
        stage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算逐步 Flow Loss（批量化版本）。
        """
        B, L, D = x_seq.shape
        device = x_seq.device

        # 收集所有有效的相邻对
        x_curr_list, x_next_list = [], []
        t_curr_list, delta_t_list = [], []
        stage_list = []

        for b in range(B):
            slen = seq_len[b].item()
            for i in range(int(slen) - 1):
                x_curr_list.append(x_seq[b, i])
                x_next_list.append(x_seq[b, i + 1])
                t_curr_list.append(time_seq[b, i])
                delta_t_list.append(time_seq[b, i + 1] - time_seq[b, i])
                if stage is not None:
                    stage_list.append(stage[b])

        if len(x_curr_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 批量化
        x_curr_batch = torch.stack(x_curr_list)  # [N, D]
        x_next_batch = torch.stack(x_next_list)  # [N, D]
        t_curr_batch = torch.stack(t_curr_list)  # [N]
        delta_t_batch = torch.stack(delta_t_list)  # [N]

        cond_data = {
            'x_curr': x_curr_batch,
            'time_curr': t_curr_batch,
            'delta_t': delta_t_batch,
        }
        if stage is not None:
            cond_data['stage'] = torch.stack(stage_list)

        # 批量计算 Flow loss
        loss = self.flow(x_next_batch, cond_data)
        return loss

    def _compute_skip_loss(
        self,
        x_seq: torch.Tensor,
        time_seq: torch.Tensor,
        seq_len: torch.Tensor,
        stage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算跨步监督 Loss。

        让模型学习任意时间跨度的预测（从起点直接预测终点）。
        """
        B, L, D = x_seq.shape
        device = x_seq.device

        # 收集所有有效的 (start, end) 对
        x_start_list, x_end_list = [], []
        t_start_list, delta_t_list = [], []
        stage_list = []

        for b in range(B):
            slen = int(seq_len[b].item())
            if slen < 3:
                continue

            x_start_list.append(x_seq[b, 0])
            x_end_list.append(x_seq[b, slen - 1])
            t_start_list.append(time_seq[b, 0])
            delta_t_list.append(time_seq[b, slen - 1] - time_seq[b, 0])
            if stage is not None:
                stage_list.append(stage[b])

        if len(x_start_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 批量化
        x_start = torch.stack(x_start_list)
        x_end = torch.stack(x_end_list)
        t_start = torch.stack(t_start_list)
        delta_t = torch.stack(delta_t_list)

        # 计算直接跨越的 Flow loss（预测 x_end）
        cond_data = {
            'x_curr': x_start,
            'time_curr': t_start,
            'delta_t': delta_t,
        }
        if stage is not None:
            cond_data['stage'] = torch.stack(stage_list)

        # 使用 Flow loss 监督直接预测
        loss = self.flow(x_end, cond_data)
        return loss

    def _compute_consistency_loss(
        self,
        x_seq: torch.Tensor,
        time_seq: torch.Tensor,
        seq_len: torch.Tensor,
        stage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算 Velocity Consistency Loss（速度场一致性）。

        核心思想：
        - 直接跨越 (start → end) 的速度场预测
        - 应该与分步跨越 (start → mid, mid → end) 的组合速度场一致

        这样避免了多步采样，既有梯度又高效。

        Loss = ||v_direct - v_composed||²
        其中 v_composed = (v1 * dt1 + v2 * dt2) / (dt1 + dt2)
        """
        B, L, D = x_seq.shape
        device = x_seq.device

        # 收集所有有效样本的数据
        x_start_list, x_mid_list, x_end_list = [], [], []
        t_start_list, t_mid_list, t_end_list = [], [], []
        stage_list = []

        for b in range(B):
            slen = int(seq_len[b].item())
            if slen < 3:
                continue

            mid_idx = slen // 2
            x_start_list.append(x_seq[b, 0])
            x_mid_list.append(x_seq[b, mid_idx])
            x_end_list.append(x_seq[b, slen - 1])
            t_start_list.append(time_seq[b, 0])
            t_mid_list.append(time_seq[b, mid_idx])
            t_end_list.append(time_seq[b, slen - 1])
            if stage is not None:
                stage_list.append(stage[b])

        if len(x_start_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 批量化处理
        x_start = torch.stack(x_start_list)  # [N, D]
        x_mid = torch.stack(x_mid_list)      # [N, D]
        x_end = torch.stack(x_end_list)      # [N, D]
        t_start = torch.stack(t_start_list)  # [N]
        t_mid = torch.stack(t_mid_list)      # [N]
        t_end = torch.stack(t_end_list)      # [N]
        stg = torch.stack(stage_list) if stage_list else None

        N = x_start.shape[0]

        # 计算时间差
        dt_total = t_end - t_start  # [N]
        dt1 = t_mid - t_start       # [N]
        dt2 = t_end - t_mid         # [N]

        # 生成中间点的插值状态（用于查询速度场）
        # 在 flow 时间 t=0.5 处查询
        flow_t = torch.full((N,), 0.5, device=device)

        # 1. 直接路径的速度场：start → end
        z_direct = 0.5 * (x_start + x_end)  # 中间插值点
        cond_direct = {
            'x_curr': x_start,
            'time_curr': t_start,
            'delta_t': dt_total,
        }
        if stg is not None:
            cond_direct['stage'] = stg
        v_direct = self.flow.backbone(z_direct, flow_t, cond_direct)

        # 2. 第一段路径的速度场：start → mid
        z1 = 0.5 * (x_start + x_mid)
        cond1 = {
            'x_curr': x_start,
            'time_curr': t_start,
            'delta_t': dt1,
        }
        if stg is not None:
            cond1['stage'] = stg
        v1 = self.flow.backbone(z1, flow_t, cond1)

        # 3. 第二段路径的速度场：mid → end
        z2 = 0.5 * (x_mid + x_end)
        cond2 = {
            'x_curr': x_mid,
            'time_curr': t_mid,
            'delta_t': dt2,
        }
        if stg is not None:
            cond2['stage'] = stg
        v2 = self.flow.backbone(z2, flow_t, cond2)

        # 组合速度场：加权平均（按时间比例）
        # v_composed = (v1 * dt1 + v2 * dt2) / (dt1 + dt2)
        dt1_expanded = dt1.unsqueeze(-1)  # [N, 1]
        dt2_expanded = dt2.unsqueeze(-1)  # [N, 1]
        dt_total_expanded = dt_total.unsqueeze(-1)  # [N, 1]

        # 避免除零
        dt_total_safe = dt_total_expanded.clamp(min=1e-6)
        v_composed = (v1 * dt1_expanded + v2 * dt2_expanded) / dt_total_safe

        # 一致性 Loss：直接速度场应该与组合速度场一致
        loss = F.mse_loss(v_direct, v_composed)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        # 每 N 步才计算一致性 loss（加速训练）
        compute_cons = (batch_idx % self.cons_every_n_steps == 0)
        losses = self.model_step(batch, compute_cons=compute_cons)

        self.log("train/loss", losses['loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_step", losses['loss_step'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_skip", losses['loss_skip'], on_step=False, on_epoch=True, sync_dist=True)
        if compute_cons:
            self.log("train/loss_cons", losses['loss_cons'], on_step=False, on_epoch=True, sync_dist=True)
        return losses['loss']

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
