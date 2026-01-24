
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
import math

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
    # Assuming adata.var_names are symbols. If not, check var['Gene Symbol']
    input_genes = adata.var_names.tolist()
    if 'Gene Symbol' in adata.var.columns:
         # Use Gene Symbol if available and looks like symbols
         # But usually var_names is the index. Let's stick to var_names for now or check
         pass
         
    # We will use var_names.
    # Create a mapping from gene name to index in input
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
        # Random pairing between adjacent days
        for i in range(len(days) - 1):
            d_curr = days[i]
            d_next = days[i+1]
            
            curr_indices = self.cells_by_day[d_curr]
            next_indices = self.cells_by_day[d_next]
            
            # Create random pairs (min length)
            n_pairs = min(len(curr_indices), len(next_indices)) * 2 # Oversample a bit? Or just use min
            # To be more robust, let's just randomly sample `n_pairs` times
            n_pairs = max(len(curr_indices), len(next_indices))
            
            idx1 = np.random.choice(curr_indices, n_pairs, replace=True)
            idx2 = np.random.choice(next_indices, n_pairs, replace=True)
            
            for c, n in zip(idx1, idx2):
                self.pairs.append((c, n))
                
        print(f"Generated {len(self.pairs)} training pairs.")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx_curr, idx_next = self.pairs[idx]
        
        # Load raw data
        # Assuming layers['log_normalised'] is what we want, or X
        # User said: Use adata.layers["log_normalised"]
        try:
            x_curr_raw = self.adata.layers["log_normalised"][idx_curr]
            x_next_raw = self.adata.layers["log_normalised"][idx_next]
        except:
             # Fallback to X if layer not found (though user said it exists)
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
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/fast/data/scTFM/downstream/lineage/ipsc/test.h5ad")
    parser.add_argument("--gene_list", type=str, default="downstreams/rtf_gene_list.txt")
    parser.add_argument("--ckpt_path", type=str, default="logs/rtf_stage2_cfg_day100/runs/2026-01-22_15-48-16/checkpoints/last.ckpt")
    parser.add_argument("--output_dir", type=str, default="bench/output/eval/ipsc")
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    
    # Convert 'day' to float (e.g., "day0" -> 0.0)
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
    
    # Manually instantiate backbone with parameters from stage2_rtf.yaml
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
    run_inference(model, adata, transform_fn, device, os.path.join(args.output_dir, "zeroshot"))
    
    # ---------------------------------------------------------
    # Fine-Tuning
    # ---------------------------------------------------------
    print("\n--- Running Fine-Tuning ---")
    model.train()
    dataset = IPSCFineTuneDataset(adata, transform_fn)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Low LR for fine-tuning
    
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
    run_inference(model, adata, transform_fn, device, os.path.join(args.output_dir, "finetuned"))


def run_inference(model, adata, transform_fn, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    # We want to predict trajectories.
    # Logic: Take cells at Day T, predict Day T+1.
    # Collect all predictions and Ground Truth for visualization.
    
    preds_list = []
    truth_list = []
    
    days = sorted(adata.obs['day'].unique())
    
    # We only predict for days that have a 'next' day
    for i in range(len(days) - 1):
        d_curr = days[i]
        d_next = days[i+1]
        
        print(f"Predicting Day {d_curr} -> Day {d_next}...")
        
        idx_curr = np.where(adata.obs['day'] == d_curr)[0]
        
        # Prepare batches
        batch_size = 128
        for b_start in range(0, len(idx_curr), batch_size):
            b_idx = idx_curr[b_start : b_start + batch_size]
            
            # Get data
            try:
                x_raw = adata.layers["log_normalised"][b_idx]
            except:
                x_raw = adata.X[b_idx]
                
            x_curr_np = transform_fn(x_raw)
            x_curr = torch.from_numpy(x_curr_np).to(device)
            
            # Meta
            t_curr = d_curr
            t_next = d_next
            dt = t_next - t_curr
            
            # Cond Data
            B = x_curr.shape[0]
            cond_data = {
                'x_curr': x_curr,
                'time_curr': torch.full((B,), normalize_time_vec(np.array([t_curr]))[0], device=device, dtype=torch.float32),
                'time_next': torch.full((B,), normalize_time_vec(np.array([t_next]))[0], device=device, dtype=torch.float32),
                'delta_t': torch.full((B,), normalize_delta_t_vec(np.array([dt]))[0], device=device, dtype=torch.float32),
                'tissue': torch.zeros(B, device=device, dtype=torch.long),
                'celltype': torch.zeros(B, device=device, dtype=torch.long),
                'stage': torch.zeros(B, device=device, dtype=torch.long),
            }
            
            # Sample (Prediction)
            # IMPORTANT: Start from Noise!
            x0_noise = torch.randn_like(x_curr)
            with torch.no_grad():
                x_pred = model.flow.sample(x0_noise, cond_data, steps=50)
                
            preds_list.append(x_pred.cpu().numpy())
            
            # Add metadata for plotting
            for _ in range(B):
                truth_list.append({
                    'type': 'Predicted',
                    'day': d_next, # It represents the next day
                    'from_day': d_curr
                })
                
    # Also add ALL Ground Truth cells to the visualization set
    print("Adding Ground Truth data...")
    all_data = []
    
    # 1. Add Predictions
    X_preds = np.concatenate(preds_list, axis=0)
    
    # 2. Add Truth (aligned)
    # We transform all data to model space for consistent PCA/UMAP
    try:
        X_full_raw = adata.layers["log_normalised"]
    except:
        X_full_raw = adata.X
        
    # Processing full dataset in chunks to save memory if needed, but 2400 is small.
    X_full_aligned = transform_fn(X_full_raw)
    
    # Construct combined AnnData
    # Structure: [Predictions; Ground Truth]
    
    X_combined = np.concatenate([X_preds, X_full_aligned], axis=0)
    
    # Obs
    obs_pred = pd.DataFrame(truth_list)
    
    obs_truth = adata.obs.copy()
    obs_truth['type'] = 'Truth'
    obs_truth['from_day'] = np.nan
    
    # Ensure columns match
    cols = ['type', 'day', 'from_day']
    obs_combined = pd.concat([obs_pred, obs_truth], ignore_index=True)
    # Fill missing columns in obs_pred with defaults if needed
    
    adata_vis = sc.AnnData(X=X_combined, obs=obs_combined)
    
    # UMAP
    print("Computing UMAP...")
    sc.pp.neighbors(adata_vis, n_neighbors=30, use_rep='X')
    sc.tl.umap(adata_vis)
    
    # Plot 1: Truth vs Predict
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata_vis, color='type', ax=ax, show=False, title="Truth vs Predicted")
    plt.savefig(os.path.join(out_dir, "umap_truth_vs_pred.png"))
    plt.close()
    
    # Plot 2: Day
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata_vis, color='day', ax=ax, show=False, title="Day")
    plt.savefig(os.path.join(out_dir, "umap_day.png"))
    plt.close()
    
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
