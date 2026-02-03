#!/usr/bin/env python3
"""
可视化 finetuned 与 zeroshot 的评估指标对比。

用法:
    python bench/visualize_metrics.py
    python bench/visualize_metrics.py --finetuned path/to/finetuned/metrics.csv --zeroshot path/to/zeroshot/metrics.csv

指标说明 (参见 downstreams/time_ipsc.py):
    - Mean_Correlation: 预测与真实群体在「基因平均表达」上的 Pearson 相关。
      衡量整体表达谱的匹配程度，越高越好。
    - LFC_Correlation: 预测与真实在「对数倍数变化 LFC = mean_pred - mean_start」上的 Pearson 相关。
      衡量时间演化方向是否一致，越高越好。
    - Mean_Wasserstein: 各基因表达分布上 Wasserstein 距离的均值（EMD）。
      衡量分布形状的差异，越低越好。
    - Mean_Max_Cell_Correlation: 对每个预测细胞，在真实群体中找最相似细胞（余弦相似度），再取平均。
      衡量生成细胞是否「看起来像」真实细胞，越高越好。
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# 默认路径
DEFAULT_FINETUNED = os.path.join(
    PROJECT_ROOT,
    "bench/output/eval/ipsc/2026-01-26_03-25-56/finetuned/metrics.csv",
)
DEFAULT_ZEROSHOT = os.path.join(
    PROJECT_ROOT,
    "bench/output/eval/ipsc/2026-01-26_03-25-56/zeroshot/metrics.csv",
)

# 绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


def load_and_merge(finetuned_path: str, zeroshot_path: str) -> pd.DataFrame:
    """加载两个 CSV 并合并为长格式，便于绘图。"""
    df_ft = pd.read_csv(finetuned_path)
    df_zs = pd.read_csv(zeroshot_path)

    df_ft["Method"] = "Finetuned"
    df_zs["Method"] = "Zeroshot"

    df = pd.concat([df_ft, df_zs], ignore_index=True)

    # 创建时间区间标签
    df["Time_Range"] = df.apply(
        lambda r: f"{int(r['Day_From'])}→{int(r['Day_To'])}", axis=1
    )

    return df


def plot_grouped_bars(df: pd.DataFrame, output_dir: str) -> None:
    """绘制各指标的 Finetuned vs Zeroshot 分组柱状图。"""
    metrics = [
        ("Mean_Correlation", "Mean Correlation", "higher"),
        ("LFC_Correlation", "LFC Correlation", "higher"),
        ("Mean_Wasserstein", "Mean Wasserstein", "lower"),
        ("Mean_Max_Cell_Correlation", "Mean Max Cell Correlation", "higher"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (col, title, _) in zip(axes, metrics):
        sns.barplot(
            data=df,
            x="Time_Range",
            y=col,
            hue="Method",
            palette={"Finetuned": "#2ecc71", "Zeroshot": "#3498db"},
            ax=ax,
            errorbar=None,
        )
        ax.set_title(title)
        ax.set_xlabel("Day Range")
        ax.set_ylabel(col)
        ax.legend(loc="lower right", fontsize=9)
        ax.tick_params(axis="x", rotation=0)

    plt.suptitle("Finetuned vs Zeroshot: Evaluation Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_grouped_bars.png")
    plt.savefig(out_path)
    plt.close()
    print(f"已保存: {out_path}")


def plot_line_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """绘制各指标随时间区间的折线对比图。"""
    metrics = [
        ("Mean_Correlation", "Mean Correlation"),
        ("LFC_Correlation", "LFC Correlation"),
        ("Mean_Wasserstein", "Mean Wasserstein"),
        ("Mean_Max_Cell_Correlation", "Mean Max Cell Correlation"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for method, color in [("Finetuned", "#2ecc71"), ("Zeroshot", "#3498db")]:
            subset = df[df["Method"] == method].sort_values("Day_From")
            ax.plot(
                subset["Time_Range"],
                subset[col],
                marker="o",
                label=method,
                color=color,
                linewidth=2,
                markersize=8,
            )
        ax.set_title(title)
        ax.set_xlabel("Day Range")
        ax.set_ylabel(col)
        ax.legend()
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Finetuned vs Zeroshot: Metrics Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_line_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"已保存: {out_path}")


def plot_radar_summary(df: pd.DataFrame, output_dir: str) -> None:
    """绘制雷达图汇总（取各指标均值，使用绝对参考尺度归一化到 0-1）。

    注意：不使用 min-max 跨方法归一化，否则较差方法会坍缩到中心点。
    改用各指标的自然尺度，使两种方法都有可读的数值。
    """
    # 按 Method 聚合（取均值）
    agg = df.groupby("Method").agg(
        {
            "Mean_Correlation": "mean",
            "LFC_Correlation": "mean",
            "Mean_Wasserstein": "mean",
            "Mean_Max_Cell_Correlation": "mean",
        }
    ).T

    # 使用绝对参考尺度归一化（避免较差方法坍缩到 0）
    # Mean_Correlation, LFC_Correlation: 相关性 [-1,1]，映射到 [0,1]
    for col in ["Mean_Correlation", "LFC_Correlation"]:
        agg.loc[col] = (agg.loc[col].clip(-1, 1) + 1) / 2

    # Mean_Wasserstein: 越小越好，用 1/(1+k*x) 映射到 [0,1]，k=15 使 0.03→0.67, 0.06→0.53
    ws = agg.loc["Mean_Wasserstein"]
    agg.loc["Mean_Wasserstein"] = 1 / (1 + 15 * ws)

    # Mean_Max_Cell_Correlation: 余弦相似度 [0,1]，直接使用
    agg.loc["Mean_Max_Cell_Correlation"] = agg.loc["Mean_Max_Cell_Correlation"].clip(
        0, 1
    )

    labels = list(agg.index)
    angles = [n / len(labels) * 2 * 3.14159 for n in range(len(labels))]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))

    for method, color in [("Finetuned", "#2ecc71"), ("Zeroshot", "#3498db")]:
        values = agg[method].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Normalized Metrics Summary (Finetuned vs Zeroshot)", pad=20)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_radar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 finetuned 与 zeroshot 评估指标")
    parser.add_argument(
        "--finetuned",
        default=DEFAULT_FINETUNED,
        help="Finetuned metrics CSV 路径",
    )
    parser.add_argument(
        "--zeroshot",
        default=DEFAULT_ZEROSHOT,
        help="Zeroshot metrics CSV 路径",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出目录，默认与 finetuned 同目录",
    )
    args = parser.parse_args()

    if not os.path.exists(args.finetuned):
        print(f"错误: 找不到 finetuned 文件: {args.finetuned}")
        sys.exit(1)
    if not os.path.exists(args.zeroshot):
        print(f"错误: 找不到 zeroshot 文件: {args.zeroshot}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(args.finetuned)
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_merge(args.finetuned, args.zeroshot)
    print("数据预览:")
    print(df.to_string(index=False))

    plot_grouped_bars(df, output_dir)
    plot_line_comparison(df, output_dir)
    plot_radar_summary(df, output_dir)

    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
