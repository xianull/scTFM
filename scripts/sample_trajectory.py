#!/usr/bin/env python
"""
细胞轨迹采样脚本

功能：
1. 加载训练好的 RTF 模型
2. 从指定起始细胞生成未来轨迹
3. 可视化生成的轨迹（UMAP/t-SNE）
4. 保存生成的细胞状态

使用方法：
# 从指定细胞生成轨迹
python scripts/sample_trajectory.py \
    --ckpt_path logs/rtf/checkpoints/best.ckpt \
    --data_dir /fast/data/scTFM/rtf/TEDD/latents \
    --start_cell_id "SRX21870170_AAACCCAAGGAGAGTA-1" \
    --time_points 0.0,0.5,1.0,1.5,2.0 \
    --output_dir outputs/trajectories

# 批量生成多条轨迹
python scripts/sample_trajectory.py \
    --ckpt_path logs/rtf/checkpoints/best.ckpt \
    --data_dir /fast/data/scTFM/rtf/TEDD/latents \
    --n_samples 100 \
    --time_points 0.0,0.5,1.0,1.5,2.0 \
    --output_dir outputs/trajectories
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import tiledbsoma
import scanpy as sc
import anndata as ad

from src.models.flow_module import FlowLitModule


def load_cell_from_tiledb(data_dir: str, cell_id: str, measurement_name: str = "RNA"):
    """
    从 TileDB-SOMA 加载指定细胞的 latent 表示。
    
    Args:
        data_dir: TileDB 根目录
        cell_id: 细胞 ID
        measurement_name: Measurement 名称
    
    Returns:
        x: 细胞的 latent 向量 (latent_dim,)
        obs_data: 细胞的元数据（dict）
    """
    # 扫描所有 shards
    shard_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    ctx = tiledbsoma.SOMATileDBContext()
    
    for shard_uri in shard_dirs:
        try:
            with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
                obs = exp.obs.read().concat().to_pandas()
                
                if cell_id in obs.index:
                    # 找到了！
                    X = exp.ms[measurement_name].X[measurement_name].read(
                        coords=([cell_id], slice(None))
                    ).tables().concat().to_pandas().to_numpy()
                    
                    obs_data = obs.loc[cell_id].to_dict()
                    
                    return X[0], obs_data
        except:
            continue
    
    raise ValueError(f"❌ Cell ID '{cell_id}' not found in {data_dir}")


def sample_random_cells(data_dir: str, n_samples: int, split_label: int = 0, measurement_name: str = "RNA"):
    """
    从数据集中随机采样起始细胞。
    
    Args:
        data_dir: TileDB 根目录
        n_samples: 采样数量
        split_label: 数据集划分 (0=Train, 1=Val)
        measurement_name: Measurement 名称
    
    Returns:
        cells: 列表，每个元素是 (cell_id, x, obs_data)
    """
    ctx = tiledbsoma.SOMATileDBContext()
    
    # 收集所有符合条件的细胞
    all_cells = []
    
    shard_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    print(f"🔍 扫描 {len(shard_dirs)} 个 shards...")
    
    for shard_uri in tqdm(shard_dirs, desc="收集细胞"):
        try:
            with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
                obs = exp.obs.read().concat().to_pandas()
                
                # 过滤 split_label
                obs_filtered = obs[obs['split_label'] == split_label]
                
                if len(obs_filtered) == 0:
                    continue
                
                # 读取 X 数据
                X = exp.ms[measurement_name].X[measurement_name].read().tables().concat().to_pandas().to_numpy()
                
                # 收集细胞
                for cell_id in obs_filtered.index:
                    idx = list(obs.index).index(cell_id)
                    all_cells.append((cell_id, X[idx], obs_filtered.loc[cell_id].to_dict()))
        except:
            continue
    
    # 随机采样
    if len(all_cells) < n_samples:
        print(f"⚠️  只找到 {len(all_cells)} 个细胞，小于请求的 {n_samples}")
        n_samples = len(all_cells)
    
    indices = np.random.choice(len(all_cells), n_samples, replace=False)
    sampled_cells = [all_cells[i] for i in indices]
    
    return sampled_cells


def generate_trajectory(model, x_start, time_points, obs_data, device='cuda'):
    """
    从起始细胞生成完整的时间轨迹。
    
    Args:
        model: 训练好的 Flow 模型
        x_start: 起始细胞 (latent_dim,)
        time_points: 时间点列表 [0.0, 0.5, 1.0, 1.5, 2.0]
        obs_data: 起始细胞的元数据
        device: 计算设备
    
    Returns:
        trajectory: numpy array (n_time_points, latent_dim)
        metadata: 每个时间点的元数据列表
    """
    trajectory = []
    metadata = []
    
    x_curr = torch.from_numpy(x_start).float().unsqueeze(0).to(device)
    trajectory.append(x_start)
    metadata.append({'time': time_points[0], **obs_data})
    
    for i in range(len(time_points) - 1):
        t_curr = time_points[i]
        t_next = time_points[i + 1]
        delta_t = t_next - t_curr
        
        cond_data = {
            'x_curr': x_curr,
            'time_curr': torch.tensor([t_curr]).to(device),
            'time_next': torch.tensor([t_next]).to(device),
            'delta_t': torch.tensor([delta_t]).to(device),
        }
        
        with torch.no_grad():
            x_next = model.flow.sample(x_curr, cond_data, steps=50, method='rk4')
        
        trajectory.append(x_next.cpu().numpy()[0])
        metadata.append({'time': t_next, **obs_data})
        
        x_curr = x_next
    
    return np.array(trajectory), metadata


def visualize_trajectories(trajectories_list, metadata_list, output_path):
    """
    可视化生成的轨迹。
    
    Args:
        trajectories_list: 列表，每个元素是 (n_time_points, latent_dim)
        metadata_list: 列表，每个元素是 metadata
        output_path: 输出路径
    """
    # 合并所有轨迹
    all_traj = np.concatenate(trajectories_list, axis=0)
    
    # 创建 AnnData
    adata = ad.AnnData(X=all_traj)
    
    # 添加元数据
    obs_data = []
    for traj_idx, metadata in enumerate(metadata_list):
        for time_idx, meta in enumerate(metadata):
            obs_data.append({
                'trajectory_id': traj_idx,
                'time': meta['time'],
                'time_step': time_idx,
            })
    
    adata.obs = pd.DataFrame(obs_data)
    
    # 降维
    print("🎨 计算 PCA...")
    sc.tl.pca(adata, n_comps=30)
    
    print("🎨 计算 UMAP...")
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 按轨迹 ID 着色
    sc.pl.umap(adata, color='trajectory_id', ax=axes[0], show=False, title="Trajectory ID")
    
    # 2. 按时间着色
    sc.pl.umap(adata, color='time', ax=axes[1], show=False, title="Time", cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="细胞轨迹采样")
    parser.add_argument("--ckpt_path", type=str, required=True, help="RTF 模型权重路径")
    parser.add_argument("--data_dir", type=str, required=True, help="Latent TileDB 根目录")
    parser.add_argument("--output_dir", type=str, default="outputs/trajectories", help="输出目录")
    
    # 采样模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_cell_id", type=str, help="起始细胞 ID")
    group.add_argument("--n_samples", type=int, help="随机采样细胞数量")
    
    # 时间点
    parser.add_argument("--time_points", type=str, required=True, help="时间点列表（逗号分隔），例如 '0.0,0.5,1.0,1.5,2.0'")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--measurement_name", type=str, default="RNA", help="Measurement 名称")
    
    args = parser.parse_args()
    
    # 解析时间点
    time_points = [float(t) for t in args.time_points.split(',')]
    print(f"⏰ 时间点: {time_points}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    print(f"🔧 加载 RTF 模型: {args.ckpt_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FlowLitModule.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    print(f"✅ 模型已加载到 {device}")
    
    # 2. 加载起始细胞
    if args.start_cell_id:
        print(f"📦 加载起始细胞: {args.start_cell_id}")
        x_start, obs_data = load_cell_from_tiledb(args.data_dir, args.start_cell_id, args.measurement_name)
        start_cells = [(args.start_cell_id, x_start, obs_data)]
    else:
        print(f"🎲 随机采样 {args.n_samples} 个起始细胞...")
        start_cells = sample_random_cells(args.data_dir, args.n_samples, split_label=0, measurement_name=args.measurement_name)
        print(f"✅ 采样完成！共 {len(start_cells)} 个细胞")
    
    # 3. 生成轨迹
    trajectories_list = []
    metadata_list = []
    
    print(f"🚀 生成轨迹...")
    for cell_id, x_start, obs_data in tqdm(start_cells, desc="生成轨迹"):
        trajectory, metadata = generate_trajectory(
            model, x_start, time_points, obs_data, device=device
        )
        trajectories_list.append(trajectory)
        metadata_list.append(metadata)
        
        # 保存单条轨迹
        traj_output = os.path.join(args.output_dir, f"trajectory_{cell_id}.npy")
        np.save(traj_output, trajectory)
    
    print(f"✅ 生成完成！共 {len(trajectories_list)} 条轨迹")
    
    # 4. 可视化
    print("🎨 可视化轨迹...")
    vis_output = os.path.join(args.output_dir, "trajectories_visualization.png")
    visualize_trajectories(trajectories_list, metadata_list, vis_output)
    
    # 5. 保存汇总
    summary_output = os.path.join(args.output_dir, "all_trajectories.npz")
    np.savez(
        summary_output,
        trajectories=np.array(trajectories_list),
        time_points=np.array(time_points)
    )
    print(f"✅ 轨迹汇总保存到: {summary_output}")
    
    print("\n🎉 完成！")


if __name__ == "__main__":
    main()

