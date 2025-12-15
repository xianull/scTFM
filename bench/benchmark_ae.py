import argparse
import os
import sys
import glob
import json
import yaml
import torch
import tiledbsoma
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
os.environ["PROJECT_ROOT"] = project_root 

from src.models.ae_module import AELitModule
from src.data.components.ae_dataset import SomaCollectionDataset

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register dummy hydra resolver
try:
    OmegaConf.register_new_resolver("hydra", lambda *args: "hydra_placeholder")
except Exception:
    pass

def find_runs(base_dir):
    """
    扫描目录寻找包含 .hydra/config.yaml 的运行目录。
    """
    runs = []
    if os.path.exists(os.path.join(base_dir, ".hydra", "config.yaml")):
        runs.append(base_dir)
    
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            run_path = os.path.join(root, d)
            if os.path.exists(os.path.join(run_path, ".hydra", "config.yaml")):
                runs.append(run_path)
    return sorted(list(set(runs)))

def get_best_checkpoint(run_dir):
    """
    寻找最佳 checkpoint。
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def setup_dataloaders(data_dir, batch_size=1024, io_chunk_size=4096):
    """
    使用 SomaCollectionDataset 构建 DataLoader。
    
    Split Labels:
    0: Train (ID)
    1: Val (ID)
    2: Test (ID)
    3: Test (OOD)
    """
    logger.info(f"Setting up dataloaders from {data_dir}...")
    
    # Test ID (Split 2)
    # 如果没有 Test set，SomaCollectionDataset 会处理为空的情况，但我们需要 catch
    try:
        ds_test_id = SomaCollectionDataset(
            root_dir=data_dir, 
            split_label=2, # Test ID
            batch_size=batch_size,
            io_chunk_size=io_chunk_size
        )
        # 检查是否为空 (通过尝试获取一个 iterator，或者直接相信 dataset 内部逻辑)
        # SomaCollectionDataset 是 IterableDataset，len() 不可用
        loader_test_id = DataLoader(
            ds_test_id, 
            batch_size=None, # Dataset handles batching
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )
    except Exception as e:
        logger.warning(f"Failed to load Test ID (Split 2): {e}. Falling back to Validation (Split 1).")
        ds_test_id = SomaCollectionDataset(
            root_dir=data_dir, 
            split_label=1, # Fallback to Val
            batch_size=batch_size,
            io_chunk_size=io_chunk_size
        )
        loader_test_id = DataLoader(
            ds_test_id, 
            batch_size=None, 
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )

    # Test OOD (Split 3)
    loader_ood = None
    try:
        # 先简单检查一下是否有 OOD 数据 (非强制)
        ds_ood = SomaCollectionDataset(
            root_dir=data_dir, 
            split_label=3, # Test OOD
            batch_size=batch_size,
            io_chunk_size=io_chunk_size
        )
        loader_ood = DataLoader(
            ds_ood, 
            batch_size=None, 
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True
        )
    except Exception:
        logger.info("No OOD data found or load failed (Split 3). Skipping OOD evaluation.")

    return loader_test_id, loader_ood

def evaluate_model(model, dataloader, device, desc="ID"):
    """
    在给定 DataLoader 上评估模型。
    """
    model.eval()
    mse_sum = 0
    n_elements = 0
    n_total_cells = 0
    
    cell_corr_sum = 0.0
    
    sum_orig = None
    sum_recon = None
    sum_sq_orig = None
    sum_sq_recon = None
    sum_gt0_orig = None
    sum_gt0_recon = None
    
    latents = []
    
    n_keep = 4096
    kept_recon = []
    kept_orig = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}"):
            # SomaCollectionDataset returns (x, labels)
            x, _ = batch
            x = x.to(device)
            
            # Forward pass
            # model is AELitModule
            # outputs = (recon_x, z, ...) depending on AE type
            res = model(x)
            
            if isinstance(res, tuple):
                recon_x = res[0]
                z = res[1]
            else:
                recon_x = res
                z = torch.zeros(x.shape[0], 64).to(device) # Fallback
            
            # 1. MSE
            loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            mse_sum += loss.item()
            n_elements += x.numel()
            n_total_cells += x.size(0)
            
            # 2. Per-Cell Pearson Correlation
            vx = x - x.mean(dim=1, keepdim=True)
            vy = recon_x - recon_x.mean(dim=1, keepdim=True)
            cost = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8)
            cell_corr_sum += cost.sum().item()
            
            # 3. Global Stats Accumulation
            if sum_orig is None:
                sum_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_sq_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_sq_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_gt0_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_gt0_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
            
            x_64 = x.to(torch.float64)
            recon_64 = recon_x.to(torch.float64)
            
            sum_orig += x_64.sum(dim=0)
            sum_recon += recon_64.sum(dim=0)
            sum_sq_orig += (x_64 ** 2).sum(dim=0)
            sum_sq_recon += (recon_64 ** 2).sum(dim=0)
            sum_gt0_orig += (x_64 > 0).float().sum(dim=0)
            sum_gt0_recon += (recon_64 > 0).float().sum(dim=0)
            
            latents.append(z.cpu().numpy())
            
            if len(kept_recon) * dataloader.dataset.batch_size < n_keep:
                kept_recon.append(recon_x.cpu().numpy())
                kept_orig.append(x.cpu().numpy())
                
    if n_total_cells == 0:
        logger.warning(f"No data found for {desc}")
        return None

    mse = mse_sum / n_elements
    mean_cell_corr = cell_corr_sum / n_total_cells
    
    # Global Gene Statistics
    mean_orig = (sum_orig / n_total_cells).float()
    mean_recon = (sum_recon / n_total_cells).float()
    
    var_orig = ((sum_sq_orig / n_total_cells) - (mean_orig.double() ** 2)).float()
    var_recon = ((sum_sq_recon / n_total_cells) - (mean_recon.double() ** 2)).float()
    
    dropout_orig = 1.0 - (sum_gt0_orig / n_total_cells).float()
    dropout_recon = 1.0 - (sum_gt0_recon / n_total_cells).float()
    
    vm_orig = mean_orig - mean_orig.mean()
    vm_recon = mean_recon - mean_recon.mean()
    gene_mean_corr = (vm_orig * vm_recon).sum() / (torch.sqrt((vm_orig**2).sum()) * torch.sqrt((vm_recon**2).sum()) + 1e-8)
    gene_mean_corr = gene_mean_corr.item()
    
    latents = np.concatenate(latents, axis=0)
    
    if kept_recon:
        kept_recon = np.concatenate(kept_recon, axis=0)
        kept_orig = np.concatenate(kept_orig, axis=0)
        if kept_recon.shape[0] > n_keep:
            kept_recon = kept_recon[:n_keep]
            kept_orig = kept_orig[:n_keep]
            
    return {
        "mse": mse,
        "mean_cell_corr": mean_cell_corr,
        "gene_mean_corr": gene_mean_corr,
        "latents": latents,
        "recon_sample": kept_recon,
        "orig_sample": kept_orig,
        "stats": {
            "mean_orig": mean_orig.cpu().numpy(),
            "mean_recon": mean_recon.cpu().numpy(),
            "var_orig": var_orig.cpu().numpy(),
            "var_recon": var_recon.cpu().numpy(),
            "dropout_orig": dropout_orig.cpu().numpy(),
            "dropout_recon": dropout_recon.cpu().numpy()
        }
    }

def plot_reconstruction(orig, recon, title_suffix="", save_path=None):
    mean_orig = np.mean(orig, axis=0)
    mean_recon = np.mean(recon, axis=0)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_orig, mean_recon, s=5, alpha=0.5, c='b')
    max_val = max(mean_orig.max(), mean_recon.max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    plt.title(f"Reconstruction: Mean Expression ({title_suffix})")
    plt.xlabel("Original Mean Expression")
    plt.ylabel("Reconstructed Mean Expression")
    corr = np.corrcoef(mean_orig, mean_recon)[0, 1]
    plt.text(0.05, 0.95, f"R = {corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_metric_scatter(val_orig, val_recon, metric_name, title_suffix="", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(val_orig, val_recon, s=5, alpha=0.5, c='purple')
    max_val = max(val_orig.max(), val_recon.max())
    min_val = min(val_orig.min(), val_recon.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    plt.title(f"Gene {metric_name} ({title_suffix})")
    plt.xlabel(f"Original {metric_name}")
    plt.ylabel(f"Reconstructed {metric_name}")
    
    mask = np.isfinite(val_orig) & np.isfinite(val_recon)
    if mask.sum() > 1:
        corr = np.corrcoef(val_orig[mask], val_recon[mask])[0, 1]
        plt.text(0.05, 0.95, f"R = {corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_gene_distributions(orig_data, recon_data, title_suffix="", save_path=None, n_genes=3):
    vars = np.var(orig_data, axis=0)
    top_indices = np.argsort(vars)[-n_genes:][::-1]
    
    fig, axes = plt.subplots(1, n_genes, figsize=(4 * n_genes, 4))
    if n_genes == 1: axes = [axes]
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        sns.kdeplot(orig_data[:, idx], label='Original', fill=True, alpha=0.3, ax=ax, clip=(0, None))
        sns.kdeplot(recon_data[:, idx], label='Recon', fill=True, alpha=0.3, ax=ax, clip=(0, None))
        ax.set_title(f"Gene {idx} Distribution")
        ax.set_xlabel("Expression")
        if i == 0:
            ax.legend()
    plt.suptitle(f"Top Variable Genes Distributions ({title_suffix})")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_gene_corr_scatter(orig_data, recon_data, title_suffix="", save_path=None):
    vars = np.var(orig_data, axis=0)
    n_top = min(500, orig_data.shape[1])
    top_indices = np.argsort(vars)[-n_top:]
    
    sub_orig = orig_data[:, top_indices]
    sub_recon = recon_data[:, top_indices]
    
    corr_orig = np.corrcoef(sub_orig, rowvar=False)
    corr_recon = np.corrcoef(sub_recon, rowvar=False)
    
    triu_idx = np.triu_indices_from(corr_orig, k=1)
    flat_orig = corr_orig[triu_idx]
    flat_recon = corr_recon[triu_idx]
    
    max_points = 10000
    if len(flat_orig) > max_points:
        idx = np.random.choice(len(flat_orig), max_points, replace=False)
        flat_orig = flat_orig[idx]
        flat_recon = flat_recon[idx]
        
    plt.figure(figsize=(6, 6))
    plt.scatter(flat_orig, flat_recon, s=5, alpha=0.3, c='green')
    plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.7)
    
    plt.title(f"Gene-Gene Correlation ({title_suffix})\n(Top {n_top} HVGs)")
    plt.xlabel("Original Correlation")
    plt.ylabel("Reconstructed Correlation")
    
    corr_of_corr = np.corrcoef(flat_orig, flat_recon)[0, 1]
    plt.text(0.05, 0.95, f"R = {corr_of_corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def run_benchmark(run_dir, wandb_base_cfg):
    logger.info(f"Processing run: {run_dir}")
    
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    if "paths" in cfg:
        project_root = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
        cfg.paths.root_dir = project_root
        cfg.paths.data_dir = os.path.join(project_root, "data")
        cfg.paths.log_dir = os.path.join(project_root, "logs")
        cfg.paths.output_dir = run_dir
        cfg.paths.work_dir = run_dir
            
    ckpt_path = get_best_checkpoint(run_dir)
    if not ckpt_path:
        logger.warning(f"No checkpoint found in {run_dir}, skipping.")
        return
    
    logger.info(f"Found checkpoint: {ckpt_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_cfg = cfg.model.net
    net = instantiate(net_cfg)
    model = AELitModule.load_from_checkpoint(ckpt_path, net=net, optimizer=None, scheduler=None, map_location=device)
    model.to(device)
    model.eval()
    
    # 4. 准备数据
    # 从 cfg.data.data_dir 获取路径，或者使用默认的
    data_dir = cfg.data.get("data_dir", "/fast/data/scTFM/ae/tileDB/all_data") # 默认 fallback
    batch_size = 2048
    io_chunk_size = cfg.data.get("io_chunk_size", 4096)
    
    loader_id, loader_ood = setup_dataloaders(data_dir, batch_size=batch_size, io_chunk_size=io_chunk_size)
    
    # 5. WandB Init
    wandb_cfg = OmegaConf.load(wandb_base_cfg)
    run_name = f"bench_{os.path.basename(run_dir)}_{cfg.model.net.get('_target_', 'ae').split('.')[-1]}"
    
    # Clean config
    for key in ["callbacks", "trainer", "logger", "hydra"]:
        if key in cfg:
            with open(os.devnull, "w") as f: 
                pass
            try:
                del cfg[key]
            except Exception:
                pass

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    run = wandb.init(
        project=wandb_cfg.get("project", "scTime-AE-bench"),
        name=run_name,
        config=config_dict,
        job_type="benchmark",
        tags=["benchmark"],
        reinit=True
    )
    
    # 6. Eval ID
    res_id = evaluate_model(model, loader_id, device, desc="ID Test")
    if res_id:
        wandb.log({
            "eval/id_mse": res_id['mse'],
            "eval/id_cell_corr": res_id['mean_cell_corr'],
            "eval/id_gene_corr": res_id['gene_mean_corr']
        })
        logger.info(f"ID MSE: {res_id['mse']:.6f}, Cell Corr: {res_id['mean_cell_corr']:.4f}, Gene Mean Corr: {res_id['gene_mean_corr']:.4f}")
        generate_plots(res_id, "id")
    
    # 7. Eval OOD
    if loader_ood:
        res_ood = evaluate_model(model, loader_ood, device, desc="OOD Test")
        if res_ood:
            wandb.log({
                "eval/ood_mse": res_ood['mse'],
                "eval/ood_cell_corr": res_ood['mean_cell_corr'],
                "eval/ood_gene_corr": res_ood['gene_mean_corr']
            })
            logger.info(f"OOD MSE: {res_ood['mse']:.6f}, Cell Corr: {res_ood['mean_cell_corr']:.4f}, Gene Mean Corr: {res_ood['gene_mean_corr']:.4f}")
            generate_plots(res_ood, "ood")
    
    # 8. Latent Vis
    if res_id:
        latents_id = res_id['latents']
        labels = ["ID"] * len(latents_id)
        all_latents = latents_id
        
        if loader_ood and res_ood:
            latents_ood = res_ood['latents']
            all_latents = np.concatenate([latents_id, latents_ood], axis=0)
            labels += ["OOD"] * len(latents_ood)
        
        # Subsample for plot
        max_plot = 10000
        if len(all_latents) > max_plot:
            idx = np.random.choice(len(all_latents), max_plot, replace=False)
            all_latents = all_latents[idx]
            labels = np.array(labels)[idx]
        
        adata_vis = sc.AnnData(X=all_latents)
        adata_vis.obs['condition'] = labels
        
        logger.info("Computing UMAP...")
        sc.tl.pca(adata_vis)
        sc.pp.neighbors(adata_vis)
        sc.tl.umap(adata_vis)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sc.pl.umap(adata_vis, color='condition', ax=ax, show=False, title="Latent Space: ID vs OOD", frameon=False)
        fig_path_umap = "latent_umap.png"
        plt.savefig(fig_path_umap, bbox_inches='tight')
        plt.close()
        
        wandb.log({"plots/latent_umap": wandb.Image(fig_path_umap)})
        os.remove(fig_path_umap)
    
    wandb.finish()

def generate_plots(res, suffix):
    stats = res['stats']
    
    fig_mean = f"recon_mean_{suffix}.png"
    plot_reconstruction(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_mean)
    wandb.log({f"plots/recon_mean_{suffix}": wandb.Image(fig_mean)})
    os.remove(fig_mean)
    
    fig_var = f"recon_var_{suffix}.png"
    plot_metric_scatter(stats['var_orig'], stats['var_recon'], "Variance", title_suffix=suffix, save_path=fig_var)
    wandb.log({f"plots/recon_var_{suffix}": wandb.Image(fig_var)})
    os.remove(fig_var)
    
    fig_drop = f"recon_dropout_{suffix}.png"
    plot_metric_scatter(stats['dropout_orig'], stats['dropout_recon'], "Dropout Rate", title_suffix=suffix, save_path=fig_drop)
    wandb.log({f"plots/recon_dropout_{suffix}": wandb.Image(fig_drop)})
    os.remove(fig_drop)
    
    fig_dist = f"gene_dist_{suffix}.png"
    plot_gene_distributions(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_dist)
    wandb.log({f"plots/gene_dist_{suffix}": wandb.Image(fig_dist)})
    os.remove(fig_dist)
    
    fig_corr = f"gene_corr_{suffix}.png"
    plot_gene_corr_scatter(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_corr)
    wandb.log({f"plots/gene_corr_{suffix}": wandb.Image(fig_corr)})
    os.remove(fig_corr)

def main():
    parser = argparse.ArgumentParser(description="AE 模型批量测评脚本")
    parser.add_argument("--dir", type=str, required=True, help="包含运行日志的目录")
    parser.add_argument("--wandb_config", type=str, default="configs/logger/wandb.yaml", help="WandB 配置文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        sys.exit(1)
        
    runs = find_runs(args.dir)
    logger.info(f"Found {len(runs)} runs to benchmark.")
    
    for run in runs:
        try:
            run_benchmark(run, args.wandb_config)
        except Exception as e:
            logger.error(f"Failed to benchmark run {run}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
