import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path

from src.models.flow_module import FlowLitModule
from src.models.components.backbone.dit_cross_attn import DiTCrossAttn

# ============================================================================ 
# Constants & Utils
# ============================================================================ 
MAX_TIME_DAYS = 36500.0
MAX_DELTA_DAYS = 36500.0

def normalize_time_vec(time_days: np.ndarray, use_log: bool = True) -> np.ndarray:
    if use_log:
        return np.log1p(time_days) / np.log1p(MAX_TIME_DAYS)
    else:
        return np.minimum(time_days / MAX_TIME_DAYS, 1.0)

def normalize_delta_t_vec(delta_days: np.ndarray, use_log: bool = True) -> np.ndarray:
    if use_log:
        sign = np.sign(delta_days)
        sign[sign == 0] = 1.0
        abs_delta = np.abs(delta_days)
        return sign * np.log1p(abs_delta) / np.log1p(MAX_DELTA_DAYS)
    else:
        return np.clip(delta_days / MAX_DELTA_DAYS, -1.0, 1.0)

def set_nature_style():
    """Configure Matplotlib for Nature-style publication plots."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['figure.titlesize'] = 8
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

# ============================================================================ 
# Data Alignment
# ============================================================================ 
def align_to_model_genes(adata, target_gene_list_path):
    print(f"Loading target gene list from {target_gene_list_path}...")
    with open(target_gene_list_path, 'r') as f:
        target_genes = [line.strip() for line in f if line.strip()]
    
    n_target = len(target_genes)
    print(f"Target genes: {n_target}")
    print(f"Input genes: {adata.n_vars}")
    
    # Create mapping
    input_genes = adata.var_names.tolist()
    input_gene_to_idx = {g: i for i, g in enumerate(input_genes)}
    
    # Build index mapping array
    # -1 indicates missing in input
    mapping = np.full(n_target, -1, dtype=int)
    
    hits = 0
    for i, gene in enumerate(target_genes):
        if gene in input_gene_to_idx:
            mapping[i] = input_gene_to_idx[gene]
            hits += 1
            
    print(f"Matched {hits}/{n_target} genes ({hits/n_target:.1%})")
    
    # Function to transform batch
    def transform(X_input):
        # X_input: (B, n_input)
        B = X_input.shape[0]
        X_out = np.zeros((B, n_target), dtype=np.float32)
        
        # Mask for valid genes
        valid_mask = mapping != -1
        valid_indices_target = np.where(valid_mask)[0]
        valid_indices_input = mapping[valid_indices_target]
        
        # Copy
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        
        # Handle sparse
        if hasattr(X_input, "toarray"):
            X_input = X_input.toarray()
            
        X_out[:, valid_indices_target] = X_input[:, valid_indices_input]
        return X_out

    return transform, target_genes

# ============================================================================ 
# Dataset
# ============================================================================ 
class IPSCFineTuneDataset(Dataset):
    def __init__(self, adata, transform_fn, time_key='day'):
        self.adata = adata
        self.transform_fn = transform_fn
        self.time_key = time_key
        
        # Group cells by day
        self.cells_by_day = {}
        days = sorted(adata.obs[time_key].unique())
        print(f"Found days: {days}")
        
        for d in days:
            self.cells_by_day[d] = np.where(adata.obs[time_key] == d)[0]
            
        self.pairs = []
        
        print("Constructing OT-like pairs based on similarity...")
        # Use simple Nearest Neighbor / Similarity pairing
        # Ideally we use POT (Python Optimal Transport), but let's implement a cosine-similarity matching
        # to avoid extra dependencies and keep it fast.
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        for i in range(len(days) - 1):
            d_curr = days[i]
            d_next = days[i+1]
            
            indices_curr = self.cells_by_day[d_curr]
            indices_next = self.cells_by_day[d_next]
            
            # Get data for matching (use PCA if available for speed/robustness, else raw)
            if 'X_pca' in adata.obsm:
                X_curr = adata.obsm['X_pca'][indices_curr]
                X_next = adata.obsm['X_pca'][indices_next]
            else:
                # Fallback to subset of genes or just raw
                # Using raw might be slow if large, but here N=2400 is small.
                try:
                    X_curr = adata.layers["log_normalised"][indices_curr]
                    X_next = adata.layers["log_normalised"][indices_next]
                except:
                    X_curr = adata.X[indices_curr]
                    X_next = adata.X[indices_next]
                
                # If sparse, densify
                if hasattr(X_curr, "toarray"): X_curr = X_curr.toarray()
                if hasattr(X_next, "toarray"): X_next = X_next.toarray()

            # Compute Similarity Matrix
            # We want to match each cell in curr to its most likely descendant in next
            # Simple strategy: For each cell in curr, pick top k matches in next (probabilistic)
            # Or simplified OT: Softmax over similarity
            
            print(f"  Matching Day {d_curr} ({len(X_curr)} cells) -> Day {d_next} ({len(X_next)} cells)...")
            
            sim_matrix = cosine_similarity(X_curr, X_next) # (N_curr, N_next)
            
            # Strategy: "Soft" Nearest Neighbors
            # For each cell in curr, sample a target in next based on similarity
            # To ensure coverage of next, we can also do the reverse, or just standard pairs.
            # Let's do: For each cell in CURR, pick the single Best Match in NEXT.
            # AND: For each cell in NEXT, pick the single Best Match in CURR.
            # This ensures we don't drop cells and we respect local structure.
            
            # 1. Forward Best Matches
            best_matches_fwd = np.argmax(sim_matrix, axis=1)
            for idx_c_local, idx_n_local in enumerate(best_matches_fwd):
                idx_c_global = indices_curr[idx_c_local]
                idx_n_global = indices_next[idx_n_local]
                self.pairs.append((idx_c_global, idx_n_global))
                
            # 2. Backward Best Matches (to ensure targets are covered)
            best_matches_bwd = np.argmax(sim_matrix, axis=0)
            for idx_n_local, idx_c_local in enumerate(best_matches_bwd):
                idx_c_global = indices_curr[idx_c_local]
                idx_n_global = indices_next[idx_n_local]
                self.pairs.append((idx_c_global, idx_n_global))
                
        # Remove duplicates
        self.pairs = list(set(self.pairs))
        print(f"Generated {len(self.pairs)} training pairs (Similarity-based).")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx_curr, idx_next = self.pairs[idx]
        
        # Load raw data
        try:
            x_curr_raw = self.adata.layers["log_normalised"][idx_curr]
            x_next_raw = self.adata.layers["log_normalised"][idx_next]
        except:
            x_curr_raw = self.adata.X[idx_curr]
            x_next_raw = self.adata.X[idx_next]

        # Reshape to (1, D) for transform then back
        x_curr_raw = x_curr_raw.reshape(1, -1)
        x_next_raw = x_next_raw.reshape(1, -1)
        
        x_curr = self.transform_fn(x_curr_raw)[0]
        x_next = self.transform_fn(x_next_raw)[0]
        
        # Meta
        t_curr = self.adata.obs[self.time_key].iloc[idx_curr]
        t_next = self.adata.obs[self.time_key].iloc[idx_next]
        dt = t_next - t_curr
        
        # Normalize
        t_curr_norm = normalize_time_vec(np.array([t_curr]))[0]
        t_next_norm = normalize_time_vec(np.array([t_next]))[0]
        dt_norm = normalize_delta_t_vec(np.array([dt]))[0]
        
        return {
            'x_curr': torch.from_numpy(x_curr),
            'x_next': torch.from_numpy(x_next),
            'cond_meta': {
                'time_curr': torch.tensor(t_curr_norm, dtype=torch.float32),
                'time_next': torch.tensor(t_next_norm, dtype=torch.float32),
                'delta_t': torch.tensor(dt_norm, dtype=torch.float32),
                'tissue': torch.tensor(0, dtype=torch.long), # Unknown
                'celltype': torch.tensor(0, dtype=torch.long), # Unknown
                'stage': torch.tensor(0, dtype=torch.long), # Unknown
            }
        }

# ============================================================================ 
# Analysis Functions
# ============================================================================ 
def calculate_metrics(pred_data, true_data, output_dir, label_suffix=""):
    """
    Calculate metrics between predicted and true cell populations.
    Generates plots and returns a dictionary of metrics.
    """
    metrics = {}
    
    # 1. Mean Gene Expression Correlation
    mean_pred = np.mean(pred_data, axis=0)
    mean_true = np.mean(true_data, axis=0)
    
    corr_mean, _ = pearsonr(mean_pred, mean_true)
    metrics['Mean_Correlation'] = corr_mean
    
    # Scatter Plot: Mean Expression
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(mean_true, mean_pred, alpha=0.5, s=5, c='navy', edgecolor='none')
    
    # Add diagonal line
    min_val = min(mean_true.min(), mean_pred.min())
    max_val = max(mean_true.max(), mean_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    
    ax.set_title(f"Mean Expression Correlation: {corr_mean:.4f}")
    ax.set_xlabel("True Mean Expression")
    ax.set_ylabel("Predicted Mean Expression")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_mean_expr_{label_suffix}.png"))
    plt.close()

    return metrics

def run_inference(model, adata, transform_fn, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    days = sorted(adata.obs['day'].unique())
    print(f"Timeline: {days}")
    
    metrics_list = []
    preds_all = []
    
    # We only predict for days that have a 'next' day
    for i in range(len(days) - 1):
        d_curr = days[i]
        d_next = days[i+1]
        
        print(f"Predicting Day {d_curr} -> Day {d_next}...")
        
        # Get Current cells
        idx_curr = np.where(adata.obs['day'] == d_curr)[0]
        
        # Get True Next cells (for evaluation)
        idx_next_true = np.where(adata.obs['day'] == d_next)[0]
        try:
             x_next_true_raw = adata.layers["log_normalised"][idx_next_true]
        except:
             x_next_true_raw = adata.X[idx_next_true]
        x_next_true = transform_fn(x_next_true_raw)
        
        # --- Prediction Loop ---
        batch_size = 128
        preds_for_step = []
        
        for b_start in range(0, len(idx_curr), batch_size):
            b_idx = idx_curr[b_start : b_start + batch_size]
            
            try:
                x_raw = adata.layers["log_normalised"][b_idx]
            except:
                x_raw = adata.X[b_idx]
                
            x_curr_np = transform_fn(x_raw)
            x_curr = torch.from_numpy(x_curr_np).to(device)
            
            B = x_curr.shape[0]
            
            # Meta
            cond_data = {
                'x_curr': x_curr,
                'time_curr': torch.full((B,), normalize_time_vec(np.array([d_curr]))[0], device=device, dtype=torch.float32),
                'time_next': torch.full((B,), normalize_time_vec(np.array([d_next]))[0], device=device, dtype=torch.float32),
                'delta_t': torch.full((B,), normalize_delta_t_vec(np.array([d_next - d_curr]))[0], device=device, dtype=torch.float32),
                'tissue': torch.zeros(B, device=device, dtype=torch.long),
                'celltype': torch.zeros(B, device=device, dtype=torch.long),
                'stage': torch.zeros(B, device=device, dtype=torch.long),
            }
            
            # Sample (Prediction)
            # [Critical Fix]: For the new Data-to-Data training logic,
            # x_pred is generated starting from x_curr, not from random noise.
            with torch.no_grad():
                x_pred = model.flow.sample(x_curr, cond_data, steps=50)
                
            preds_list.append(x_pred.cpu().numpy())
            
        x_pred_all = np.concatenate(preds_for_step, axis=0)
        
        # --- Evaluation for this step ---
        step_metrics = calculate_metrics(
            x_pred_all, 
            x_next_true, 
            out_dir, 
            label_suffix=f"day{d_curr}_to_{d_next}"
        )
        step_metrics['Day_From'] = d_curr
        step_metrics['Day_To'] = d_next
        metrics_list.append(step_metrics)
        
        # Store for Combined Plotting
        preds_all.append({
            'data': x_pred_all,
            'day': d_next,
            'from': d_curr
        })

    # --- Save Metrics ---
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print("Metrics saved.")

    # --- Combined Visualization ---
    print("Generating combined visualizations...")
    
    # 1. Prepare data for UMAP/PCA
    # We combine: 
    #   - All Ground Truth Data (for reference structure)
    #   - All Predicted Data
    
    # Transform full dataset
    try:
        X_full_raw = adata.layers["log_normalised"]
    except:
        X_full_raw = adata.X
    X_full_aligned = transform_fn(X_full_raw)
    
    # Build Obs for Full Data
    obs_full = adata.obs.copy()
    obs_full['type'] = 'Truth'
    obs_full['predicted_from'] = np.nan
    
    # Build Obs for Pred Data
    X_pred_list = []
    obs_pred_list = []
    
    for item in preds_all:
        X_pred_list.append(item['data'])
        n_cells = item['data'].shape[0]
        obs_df = pd.DataFrame({
            'day': [item['day']] * n_cells,
            'type': ['Predicted'] * n_cells,
            'predicted_from': [item['from']] * n_cells
        })
        obs_pred_list.append(obs_df)
        
    X_pred_combined = np.concatenate(X_pred_list, axis=0)
    obs_pred_combined = pd.concat(obs_pred_list, ignore_index=True)
    
    # Combine
    X_final = np.concatenate([X_full_aligned, X_pred_combined], axis=0)
    obs_final = pd.concat([obs_full, obs_pred_combined], ignore_index=True)
    
    adata_vis = sc.AnnData(X=X_final, obs=obs_final)
    
    # PCA
    print("Computing PCA...")
    sc.tl.pca(adata_vis, svd_solver='arpack')
    
    # UMAP
    print("Computing UMAP...")
    sc.pp.neighbors(adata_vis, n_neighbors=30, n_pcs=40)
    sc.tl.umap(adata_vis)
    
    # --- Plotting UMAP ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 1: Type (Truth vs Predicted)
    # Shuffle indices to avoid occlusion
    indices = np.arange(adata_vis.n_obs)
    np.random.shuffle(indices)
    
    sc.pl.umap(adata_vis[indices], color='type', ax=axes[0], show=False, 
               title="Truth vs Predicted", frameon=False, s=10)
    
    # Plot 2: Day
    sc.pl.umap(adata_vis[indices], color='day', ax=axes[1], show=False, 
               title="Development Day", frameon=False, s=10, cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_summary.png"))
    plt.close()
    
    # --- Plotting PCA ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sc.pl.pca(adata_vis[indices], color='type', ax=axes[0], show=False, 
               title="PCA: Truth vs Predicted", frameon=False, s=10)
    sc.pl.pca(adata_vis[indices], color='day', ax=axes[1], show=False, 
               title="PCA: Development Day", frameon=False, s=10, cmap='viridis')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_summary.png"))
    plt.close()

    print(f"All results saved to {out_dir}")

# ============================================================================ 
# Main
# ============================================================================ 
def main():
    set_nature_style()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/fast/data/scTFM/downstream/lineage/ipsc/test.h5ad")
    parser.add_argument("--gene_list", type=str, default="downstreams/rtf_gene_list.txt")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="bench/output/eval/ipsc")
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    args = parser.parse_args()
    
    # Parse timestamp from ckpt path for folder naming
    # Expected format: .../runs/TIMESTAMP/checkpoints/last.ckpt
    try:
        timestamp = Path(args.ckpt_path).parents[1].name
    except:
        timestamp = "unknown_timestamp"
        
    base_out_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(base_out_dir, exist_ok=True)
    
    print(f"Output Directory: {base_out_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    
    print("Parsing 'day' column...")
    try:
        if adata.obs['day'].dtype == 'object' or isinstance(adata.obs['day'].dtype, pd.CategoricalDtype):
             adata.obs['day'] = adata.obs['day'].astype(str).str.replace('day', '', regex=False).astype(float)
        else:
             adata.obs['day'] = adata.obs['day'].astype(float)
    except Exception as e:
        print(f"Warning: Failed to parse 'day' column: {e}")
    
    # 2. Setup Transform
    transform_fn, target_genes = align_to_model_genes(adata, args.gene_list)
    
    # 3. Load Model
    print(f"Loading model from {args.ckpt_path}...")
    
    net = DiTCrossAttn(
        input_dim=28231,
        hidden_size=768,
        depth=16,
        num_heads=12,
        n_tissues=12,
        n_celltypes=62,
        n_stages=6,
        cond_dropout=0.15,
        mlp_ratio=4.0,
        dropout=0.0
    )
    
    model = FlowLitModule.load_from_checkpoint(args.ckpt_path, net=net, map_location=device)
    model.to(device)
    
    # --------------------------------------------------------- 
    # Zero-Shot Inference
    # --------------------------------------------------------- 
    print("\n--- Running Zero-Shot Inference ---")
    run_inference(model, adata, transform_fn, device, os.path.join(base_out_dir, "zeroshot"))
    
    # --------------------------------------------------------- 
    # Fine-Tuning
    # --------------------------------------------------------- 
    print("\n--- Running Fine-Tuning ---")
    model.train()
    dataset = IPSCFineTuneDataset(adata, transform_fn)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(args.fine_tune_epochs):
        total_loss = 0
        steps = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.fine_tune_epochs}"):
            # Move to device
            batch['x_next'] = batch['x_next'].to(device)
            batch['x_curr'] = batch['x_curr'].to(device)
            for k, v in batch['cond_meta'].items():
                batch['cond_meta'][k] = v.to(device)
            
            loss = model.model_step(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        print(f"Epoch {epoch+1} Loss: {total_loss/steps:.4f}")
        
    # --------------------------------------------------------- 
    # Fine-Tuned Inference
    # --------------------------------------------------------- 
    print("\n--- Running Fine-Tuned Inference ---")
    run_inference(model, adata, transform_fn, device, os.path.join(base_out_dir, "finetuned"))

if __name__ == "__main__":
    main()