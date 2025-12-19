"""
插值结果可视化脚本

功能：
1. UMAP 轨迹可视化（带时间颜色编码）
2. 基因表达动态变化热图
3. 轨迹动画（可选）

使用方式：
    python scripts/visualize_interpolation.py \
        --input results/trajectory.h5ad \
        --output_dir results/plots \
        --n_genes 50
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import umap

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_umap_trajectory(adata, output_path=None, n_neighbors=30, min_dist=0.3):
    """
    绘制 UMAP 轨迹图，展示时间演化

    不同时间点用不同颜色表示，同一细胞的轨迹用线连接
    """
    print("Computing UMAP...")

    # 获取数据
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # 计算 UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_coords = reducer.fit_transform(X)

    adata.obsm['X_umap'] = umap_coords

    # 获取时间信息
    times = adata.obs['interp_time'].values
    steps = adata.obs['interp_step'].values
    unique_steps = sorted(adata.obs['interp_step'].unique())
    n_steps = len(unique_steps)

    # 计算每个原始细胞的数量
    n_cells_per_step = len(adata) // n_steps

    fig, ax = plt.subplots(figsize=(12, 10))

    # 颜色映射
    cmap = plt.cm.viridis
    norm = Normalize(vmin=times.min(), vmax=times.max())

    # 绘制所有点
    scatter = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=times,
        cmap=cmap,
        s=10,
        alpha=0.6,
    )

    # 绘制轨迹线（连接同一细胞的不同时间点）
    print("Drawing trajectories...")
    n_trajectories_to_draw = min(200, n_cells_per_step)  # 限制绘制数量
    trajectory_indices = np.random.choice(n_cells_per_step, n_trajectories_to_draw, replace=False)

    for idx in trajectory_indices:
        # 获取这个细胞在所有时间点的坐标
        cell_coords = []
        cell_times = []

        for step in unique_steps:
            cell_idx = step * n_cells_per_step + idx
            if cell_idx < len(umap_coords):
                cell_coords.append(umap_coords[cell_idx])
                cell_times.append(times[cell_idx])

        if len(cell_coords) > 1:
            cell_coords = np.array(cell_coords)
            # 绘制带颜色的线段
            points = cell_coords.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.3, linewidth=0.8)
            lc.set_array(np.array(cell_times[:-1]))
            ax.add_collection(lc)

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Time', fontsize=12)

    # 标记时间点
    for step in unique_steps:
        step_mask = steps == step
        step_time = times[step_mask][0]
        step_coords = umap_coords[step_mask]
        centroid = step_coords.mean(axis=0)
        ax.annotate(
            f't={step_time:.1f}',
            xy=centroid,
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title('Cell Trajectory Interpolation (UMAP)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_umap_by_step(adata, output_path=None):
    """
    分面图：每个时间步一个子图
    """
    steps = sorted(adata.obs['interp_step'].unique())
    n_steps = len(steps)

    # 计算全局 UMAP（如果还没有）
    if 'X_umap' not in adata.obsm:
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        adata.obsm['X_umap'] = reducer.fit_transform(X)

    umap_coords = adata.obsm['X_umap']
    times = adata.obs['interp_time'].values

    # 创建子图
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_steps == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # 全局坐标范围
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)

    for i, step in enumerate(steps):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        step_mask = adata.obs['interp_step'] == step
        step_coords = umap_coords[step_mask]
        step_time = times[step_mask][0]

        # 绘制背景（其他时间点，半透明）
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c='lightgray',
            s=5,
            alpha=0.2,
        )

        # 绘制当前时间点
        ax.scatter(
            step_coords[:, 0],
            step_coords[:, 1],
            c=plt.cm.viridis(i / (n_steps - 1) if n_steps > 1 else 0.5),
            s=15,
            alpha=0.7,
        )

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f't = {step_time:.1f}', fontsize=12)
        ax.set_xlabel('UMAP1', fontsize=10)
        ax.set_ylabel('UMAP2', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.2)

    # 隐藏多余的子图
    for i in range(n_steps, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.suptitle('Cell State Evolution Over Time', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_gene_dynamics(adata, output_path=None, n_genes=30, method='variance'):
    """
    绘制基因表达动态变化热图

    Args:
        adata: AnnData 对象
        output_path: 输出路径
        n_genes: 展示的基因数量
        method: 选择基因的方法 ('variance', 'change')
    """
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    steps = sorted(adata.obs['interp_step'].unique())
    times = [adata.obs[adata.obs['interp_step'] == s]['interp_time'].iloc[0] for s in steps]

    # 计算每个时间点的平均表达
    mean_expr_by_step = []
    for step in steps:
        step_mask = adata.obs['interp_step'] == step
        mean_expr = X[step_mask].mean(axis=0)
        mean_expr_by_step.append(mean_expr)

    mean_expr_matrix = np.array(mean_expr_by_step)  # [n_steps, n_genes]

    # 选择基因
    if method == 'variance':
        # 选择跨时间变化最大的基因
        gene_var = np.var(mean_expr_matrix, axis=0)
        top_genes = np.argsort(gene_var)[-n_genes:]
    else:  # change
        # 选择从开始到结束变化最大的基因
        change = np.abs(mean_expr_matrix[-1] - mean_expr_matrix[0])
        top_genes = np.argsort(change)[-n_genes:]

    # 提取子矩阵
    expr_subset = mean_expr_matrix[:, top_genes].T  # [n_genes, n_steps]

    # 标准化（z-score）
    expr_normalized = (expr_subset - expr_subset.mean(axis=1, keepdims=True)) / (expr_subset.std(axis=1, keepdims=True) + 1e-8)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, max(8, n_genes * 0.3)))

    # 热图
    im = ax.imshow(expr_normalized, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

    # 设置坐标轴
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f't={t:.1f}' for t in times], rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Gene Index', fontsize=12)
    ax.set_title('Gene Expression Dynamics Over Time (Z-score)', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Z-score', fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_expression_trends(adata, output_path=None, n_genes=9):
    """
    绘制单个基因的表达趋势线图
    """
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    steps = sorted(adata.obs['interp_step'].unique())
    times = [adata.obs[adata.obs['interp_step'] == s]['interp_time'].iloc[0] for s in steps]

    # 计算每个时间点的平均表达和标准差
    mean_expr_by_step = []
    std_expr_by_step = []
    for step in steps:
        step_mask = adata.obs['interp_step'] == step
        mean_expr = X[step_mask].mean(axis=0)
        std_expr = X[step_mask].std(axis=0)
        mean_expr_by_step.append(mean_expr)
        std_expr_by_step.append(std_expr)

    mean_expr_matrix = np.array(mean_expr_by_step)
    std_expr_matrix = np.array(std_expr_by_step)

    # 选择变化最大的基因
    gene_var = np.var(mean_expr_matrix, axis=0)
    top_genes = np.argsort(gene_var)[-n_genes:]

    # 绘图
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, gene_idx in enumerate(top_genes):
        ax = axes[i]

        mean_vals = mean_expr_matrix[:, gene_idx]
        std_vals = std_expr_matrix[:, gene_idx]

        ax.plot(times, mean_vals, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.fill_between(
            times,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.2,
            color='steelblue'
        )

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Expression', fontsize=10)
        ax.set_title(f'Gene {gene_idx}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 隐藏多余子图
    for i in range(n_genes, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Gene Expression Trends Over Time', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_distribution_shift(adata, output_path=None):
    """
    绘制表达分布随时间的变化（小提琴图）
    """
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    steps = sorted(adata.obs['interp_step'].unique())
    times = [adata.obs[adata.obs['interp_step'] == s]['interp_time'].iloc[0] for s in steps]

    # 计算每个细胞的统计量
    cell_means = X.mean(axis=1)
    cell_vars = X.var(axis=1)

    # 创建 DataFrame
    df = pd.DataFrame({
        'Time': [f't={t:.1f}' for t in adata.obs['interp_time']],
        'Mean Expression': cell_means,
        'Expression Variance': cell_vars,
        'Step': adata.obs['interp_step'].values,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 平均表达分布
    ax = axes[0]
    sns.violinplot(data=df, x='Time', y='Mean Expression', ax=ax, palette='viridis')
    ax.set_title('Mean Expression Distribution Over Time', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 表达方差分布
    ax = axes[1]
    sns.violinplot(data=df, x='Time', y='Expression Variance', ax=ax, palette='viridis')
    ax.set_title('Expression Variance Distribution Over Time', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('Cell Expression Distribution Shift', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="插值结果可视化")
    parser.add_argument("--input", type=str, required=True, help="输入 h5ad 文件路径")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="输出目录")
    parser.add_argument("--n_genes", type=int, default=30, help="热图中展示的基因数量")
    parser.add_argument("--prefix", type=str, default="interp", help="输出文件前缀")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    print(f"Loaded {adata.n_obs} cells")
    print(f"Time points: {sorted(adata.obs['interp_time'].unique())}")

    # 1. UMAP 轨迹图
    print("\n1. Generating UMAP trajectory plot...")
    plot_umap_trajectory(
        adata,
        output_path=os.path.join(args.output_dir, f"{args.prefix}_umap_trajectory.png")
    )

    # 2. UMAP 分面图
    print("\n2. Generating UMAP facet plot...")
    plot_umap_by_step(
        adata,
        output_path=os.path.join(args.output_dir, f"{args.prefix}_umap_facet.png")
    )

    # 3. 基因表达热图
    print("\n3. Generating gene dynamics heatmap...")
    plot_gene_dynamics(
        adata,
        output_path=os.path.join(args.output_dir, f"{args.prefix}_gene_heatmap.png"),
        n_genes=args.n_genes
    )

    # 4. 基因表达趋势
    print("\n4. Generating expression trend plots...")
    plot_expression_trends(
        adata,
        output_path=os.path.join(args.output_dir, f"{args.prefix}_gene_trends.png")
    )

    # 5. 分布变化
    print("\n5. Generating distribution shift plot...")
    plot_distribution_shift(
        adata,
        output_path=os.path.join(args.output_dir, f"{args.prefix}_distribution.png")
    )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
