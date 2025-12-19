"""
RTF 模型插值推理脚本

功能：
1. 加载训练好的 RTF 模型
2. 对输入的 h5ad 数据进行时间插值
3. 支持多步插值（生成中间时间点）
4. 输出插值结果为新的 h5ad 文件

使用方式：
    # 单步预测：从 t_curr 预测 t_next
    python scripts/interpolate_h5ad.py \
        --ckpt logs/rtf_train_cfg/runs/2025-12-18_09-04-14/checkpoints/last.ckpt \
        --input data/my_cells.h5ad \
        --output results/predicted.h5ad \
        --t_curr 30 \
        --t_next 36

    # 多步插值：在 t_curr 和 t_next 之间插入中间点
    python scripts/interpolate_h5ad.py \
        --ckpt logs/rtf_train_cfg/runs/xxx/checkpoints/last.ckpt \
        --input data/my_cells.h5ad \
        --output results/trajectory.h5ad \
        --t_curr 30 \
        --t_next 60 \
        --n_interp 5 \
        --cfg_scale 2.0

    # 使用已有的 time 列
    python scripts/interpolate_h5ad.py \
        --ckpt logs/rtf_train_cfg/runs/xxx/checkpoints/last.ckpt \
        --input data/my_cells.h5ad \
        --output results/predicted.h5ad \
        --delta_t 6
"""

import argparse
import os
import sys
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.environ["PROJECT_ROOT"] = str(project_root)

from src.models.flow_module import FlowLitModule


def load_model(ckpt_path: str, device: torch.device):
    """加载 RTF 模型"""
    ckpt_path = Path(ckpt_path)

    # 找到 hydra config
    run_dir = ckpt_path.parent.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    # 如果是 latent 模式，需要推断 input_dim
    mode = cfg.model.get("mode", "raw")
    ae_ckpt_path = cfg.model.get("ae_ckpt_path")

    if mode == "latent" and ae_ckpt_path:
        ae_ckpt_dir = Path(ae_ckpt_path).parent.parent
        ae_hydra_config = ae_ckpt_dir / ".hydra" / "config.yaml"

        if ae_hydra_config.exists():
            ae_cfg = OmegaConf.load(ae_hydra_config)
            latent_dim = ae_cfg.get('model', {}).get('net', {}).get('latent_dim')
            if latent_dim:
                OmegaConf.set_struct(cfg, False)
                cfg.model.net.input_dim = latent_dim
                OmegaConf.set_struct(cfg, True)
                print(f"Inferred input_dim={latent_dim} from AE checkpoint")

    # 实例化 backbone
    net = instantiate(cfg.model.net)

    # 加载模型
    model = FlowLitModule.load_from_checkpoint(
        str(ckpt_path),
        net=net,
        optimizer=None,
        scheduler=None,
        map_location=device
    )
    model.to(device)
    model.eval()

    return model, cfg


def load_ae_encoder(ae_ckpt_path: str, device: torch.device):
    """加载 AE encoder（用于将原始数据编码到 latent space）"""
    from src.models.ae_module import AELitModule

    ae_ckpt_path = Path(ae_ckpt_path)
    ae_run_dir = ae_ckpt_path.parent.parent
    ae_cfg_path = ae_run_dir / ".hydra" / "config.yaml"

    if not ae_cfg_path.exists():
        raise FileNotFoundError(f"AE Hydra config not found: {ae_cfg_path}")

    ae_cfg = OmegaConf.load(ae_cfg_path)
    ae_net = instantiate(ae_cfg.model.net)

    ae_model = AELitModule.load_from_checkpoint(
        str(ae_ckpt_path),
        net=ae_net,
        optimizer=None,
        scheduler=None,
        map_location=device
    )
    ae_model.to(device)
    ae_model.eval()

    return ae_model


def encode_tissue(tissue_name: str) -> int:
    """编码 tissue 名称为整数 ID"""
    tissue_mapping = {
        'Brain': 0, 'Heart': 1, 'Liver': 2, 'Lung': 3,
        'Kidney': 4, 'Intestine': 5, 'Skin': 6, 'Blood': 7,
        'Bone': 8, 'Muscle': 9, 'Pancreas': 10, 'Unknown': 11
    }
    return tissue_mapping.get(tissue_name, 11)


def encode_celltype(celltype_name: str) -> int:
    """编码 celltype 名称为整数 ID（简化版，实际应从训练数据中获取）"""
    # 这里返回默认值，实际使用时应该加载完整的 celltype mapping
    return 0


def predict_next_state(
    model,
    x_curr: torch.Tensor,
    time_curr: float,
    delta_t: float,
    tissue: int = 11,
    celltype: int = 0,
    steps: int = 50,
    cfg_scale: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    预测下一时刻的细胞状态

    Args:
        model: RTF 模型
        x_curr: 当前细胞状态 [B, D]
        time_curr: 当前时间
        delta_t: 时间步长
        tissue: tissue ID
        celltype: celltype ID
        steps: ODE 求解步数
        cfg_scale: CFG 强度
        device: 计算设备

    Returns:
        x_next: 预测的下一时刻状态 [B, D]
    """
    batch_size = x_curr.shape[0]

    # 构建条件数据
    cond_data = {
        'x_curr': x_curr,
        'time_curr': torch.full((batch_size,), time_curr, device=device),
        'time_next': torch.full((batch_size,), time_curr + delta_t, device=device),
        'delta_t': torch.full((batch_size,), delta_t, device=device),
        'tissue': torch.full((batch_size,), tissue, dtype=torch.long, device=device),
        'celltype': torch.full((batch_size,), celltype, dtype=torch.long, device=device),
    }

    # 从噪声开始采样
    x0 = torch.randn_like(x_curr)

    # 调用 RectifiedFlow.sample
    with torch.no_grad():
        x_next = model.flow.sample(x0, cond_data, steps=steps, method='euler', cfg_scale=cfg_scale)

    return x_next


def interpolate_trajectory(
    model,
    x_start: torch.Tensor,
    t_start: float,
    t_end: float,
    n_steps: int,
    tissue: int = 11,
    celltype: int = 0,
    ode_steps: int = 50,
    cfg_scale: float = 1.0,
    device: torch.device = None,
) -> tuple:
    """
    生成从 t_start 到 t_end 的完整轨迹

    Args:
        model: RTF 模型
        x_start: 起始细胞状态 [B, D]
        t_start: 起始时间
        t_end: 结束时间
        n_steps: 插值步数（生成 n_steps+1 个时间点）
        tissue: tissue ID
        celltype: celltype ID
        ode_steps: 每步 ODE 求解步数
        cfg_scale: CFG 强度
        device: 计算设备

    Returns:
        trajectory: 轨迹数据 [n_steps+1, B, D]
        time_points: 时间点 [n_steps+1]
    """
    batch_size = x_start.shape[0]

    # 生成时间网格
    time_points = np.linspace(t_start, t_end, n_steps + 1)
    delta_t = (t_end - t_start) / n_steps

    # 存储轨迹
    trajectory = [x_start.cpu().numpy()]

    x_curr = x_start
    for i in tqdm(range(n_steps), desc="Interpolating"):
        t_curr = time_points[i]

        x_next = predict_next_state(
            model=model,
            x_curr=x_curr,
            time_curr=t_curr,
            delta_t=delta_t,
            tissue=tissue,
            celltype=celltype,
            steps=ode_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        trajectory.append(x_next.cpu().numpy())
        x_curr = x_next

    trajectory = np.stack(trajectory, axis=0)  # [n_steps+1, B, D]

    return trajectory, time_points


def process_h5ad(
    adata: sc.AnnData,
    model,
    ae_model,
    t_curr: float,
    t_next: float = None,
    delta_t: float = None,
    n_interp: int = 1,
    tissue_col: str = 'Tissue',
    celltype_col: str = 'Celltype',
    time_col: str = 'time',
    batch_size: int = 256,
    ode_steps: int = 50,
    cfg_scale: float = 1.0,
    device: torch.device = None,
    mode: str = 'latent',
) -> sc.AnnData:
    """
    处理 h5ad 文件，进行插值预测

    Args:
        adata: 输入的 AnnData 对象
        model: RTF 模型
        ae_model: AE 模型（仅 latent 模式需要）
        t_curr: 当前时间（如果为 None，则从 time_col 读取）
        t_next: 目标时间
        delta_t: 时间步长（与 t_next 二选一）
        n_interp: 插值步数
        tissue_col: tissue 列名
        celltype_col: celltype 列名
        time_col: time 列名
        batch_size: 批量大小
        ode_steps: ODE 求解步数
        cfg_scale: CFG 强度
        device: 计算设备
        mode: 'latent' 或 'raw'

    Returns:
        result_adata: 包含插值结果的 AnnData
    """
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    # 确定 delta_t
    if t_next is not None:
        total_delta = t_next - t_curr
    elif delta_t is not None:
        total_delta = delta_t
        t_next = t_curr + total_delta
    else:
        raise ValueError("Must specify either t_next or delta_t")

    step_delta = total_delta / n_interp

    print(f"Interpolating from t={t_curr} to t={t_next} in {n_interp} steps")
    print(f"Step delta_t: {step_delta}")

    # 获取数据
    if mode == 'latent':
        # 先编码到 latent space
        print("Encoding to latent space...")
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32, device=device)

        latents = []
        for i in tqdm(range(0, n_cells, batch_size), desc="Encoding"):
            batch = X[i:i+batch_size]
            with torch.no_grad():
                z = ae_model.net.encode(batch)
            latents.append(z.cpu())

        x_data = torch.cat(latents, dim=0).to(device)
        latent_dim = x_data.shape[1]
    else:
        # 直接使用原始数据
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        x_data = torch.tensor(X, dtype=torch.float32, device=device)

    # 获取 tissue 和 celltype
    if tissue_col in adata.obs.columns:
        tissues = adata.obs[tissue_col].apply(encode_tissue).values
    else:
        tissues = np.full(n_cells, 11)  # Unknown

    if celltype_col in adata.obs.columns:
        celltypes = adata.obs[celltype_col].apply(encode_celltype).values
    else:
        celltypes = np.zeros(n_cells, dtype=int)

    # 存储所有时间点的结果
    all_results = []
    all_times = []
    all_obs = []

    # 添加原始数据（t_curr）
    all_results.append(x_data.cpu().numpy())
    all_times.extend([t_curr] * n_cells)
    obs_t0 = adata.obs.copy()
    obs_t0['interp_time'] = t_curr
    obs_t0['interp_step'] = 0
    all_obs.append(obs_t0)

    # 逐步插值
    x_curr = x_data
    current_time = t_curr

    for step in range(n_interp):
        print(f"\nStep {step+1}/{n_interp}: t={current_time:.2f} -> {current_time + step_delta:.2f}")

        step_results = []

        for i in tqdm(range(0, n_cells, batch_size), desc=f"Predicting step {step+1}"):
            batch_x = x_curr[i:i+batch_size]
            batch_tissue = tissues[i:i+batch_size]
            batch_celltype = celltypes[i:i+batch_size]

            # 使用最常见的 tissue/celltype（简化处理）
            tissue_id = int(np.median(batch_tissue))
            celltype_id = int(np.median(batch_celltype))

            x_next = predict_next_state(
                model=model,
                x_curr=batch_x,
                time_curr=current_time,
                delta_t=step_delta,
                tissue=tissue_id,
                celltype=celltype_id,
                steps=ode_steps,
                cfg_scale=cfg_scale,
                device=device,
            )

            step_results.append(x_next.cpu())

        x_next_all = torch.cat(step_results, dim=0)
        all_results.append(x_next_all.numpy())

        next_time = current_time + step_delta
        all_times.extend([next_time] * n_cells)

        obs_step = adata.obs.copy()
        obs_step['interp_time'] = next_time
        obs_step['interp_step'] = step + 1
        all_obs.append(obs_step)

        x_curr = x_next_all.to(device)
        current_time = next_time

    # 合并结果
    X_combined = np.vstack(all_results)  # [n_cells * (n_interp+1), D]
    obs_combined = pd.concat(all_obs, ignore_index=True)

    # 如果是 latent 模式，需要解码回原始空间
    if mode == 'latent':
        print("\nDecoding from latent space...")
        X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32, device=device)

        decoded = []
        for i in tqdm(range(0, len(X_combined_tensor), batch_size), desc="Decoding"):
            batch = X_combined_tensor[i:i+batch_size]
            with torch.no_grad():
                x_decoded = ae_model.net.decode(batch)
            decoded.append(x_decoded.cpu().numpy())

        X_combined = np.vstack(decoded)

    # 创建结果 AnnData
    result_adata = sc.AnnData(
        X=X_combined,
        obs=obs_combined,
        var=adata.var.copy() if X_combined.shape[1] == n_genes else None,
    )

    # 添加元信息
    result_adata.uns['interpolation'] = {
        't_start': t_curr,
        't_end': t_next,
        'n_steps': n_interp,
        'step_delta': step_delta,
        'cfg_scale': cfg_scale,
        'ode_steps': ode_steps,
    }

    return result_adata


import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="RTF 模型插值推理")
    parser.add_argument("--ckpt", type=str, required=True, help="RTF 模型 checkpoint 路径")
    parser.add_argument("--input", type=str, required=True, help="输入 h5ad 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 h5ad 文件路径")

    # 时间参数
    parser.add_argument("--t_curr", type=float, default=None, help="当前时间点")
    parser.add_argument("--t_next", type=float, default=None, help="目标时间点")
    parser.add_argument("--delta_t", type=float, default=None, help="时间步长（与 t_next 二选一）")
    parser.add_argument("--n_interp", type=int, default=1, help="插值步数")

    # 模型参数
    parser.add_argument("--ode_steps", type=int, default=50, help="ODE 求解步数")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG 强度")
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")

    # 数据参数
    parser.add_argument("--tissue_col", type=str, default="Tissue", help="Tissue 列名")
    parser.add_argument("--celltype_col", type=str, default="Celltype", help="Celltype 列名")
    parser.add_argument("--time_col", type=str, default="time", help="Time 列名")

    args = parser.parse_args()

    # 检查参数
    if args.t_next is None and args.delta_t is None:
        raise ValueError("Must specify either --t_next or --delta_t")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 RTF 模型
    print(f"\nLoading RTF model from {args.ckpt}...")
    model, cfg = load_model(args.ckpt, device)
    mode = cfg.model.get("mode", "raw")
    print(f"Model mode: {mode}")

    # 加载 AE 模型（如果是 latent 模式）
    ae_model = None
    if mode == "latent":
        ae_ckpt_path = cfg.model.get("ae_ckpt_path")
        if ae_ckpt_path:
            print(f"\nLoading AE model from {ae_ckpt_path}...")
            ae_model = load_ae_encoder(ae_ckpt_path, device)
        else:
            raise ValueError("Latent mode requires ae_ckpt_path in config")

    # 加载输入数据
    print(f"\nLoading input data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # 确定 t_curr
    if args.t_curr is not None:
        t_curr = args.t_curr
    elif args.time_col in adata.obs.columns:
        t_curr = float(adata.obs[args.time_col].median())
        print(f"Using median time from '{args.time_col}' column: {t_curr}")
    else:
        raise ValueError("Must specify --t_curr or have time column in data")

    # 运行插值
    result_adata = process_h5ad(
        adata=adata,
        model=model,
        ae_model=ae_model,
        t_curr=t_curr,
        t_next=args.t_next,
        delta_t=args.delta_t,
        n_interp=args.n_interp,
        tissue_col=args.tissue_col,
        celltype_col=args.celltype_col,
        time_col=args.time_col,
        batch_size=args.batch_size,
        ode_steps=args.ode_steps,
        cfg_scale=args.cfg_scale,
        device=device,
        mode=mode,
    )

    # 保存结果
    print(f"\nSaving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    result_adata.write_h5ad(args.output)

    print(f"\nDone! Output shape: {result_adata.shape}")
    print(f"Time points: {sorted(result_adata.obs['interp_time'].unique())}")


if __name__ == "__main__":
    main()
