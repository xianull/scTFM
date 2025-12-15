import os
import torch
import tiledbsoma
import numpy as np
import math
import argparse
import yaml
from tqdm import tqdm
import sys
import pyarrow as pa
import shutil

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ae_module import AELitModule

def extract_latents(
    data_dir: str,
    output_dir: str,
    ckpt_path: str,
    batch_size: int = 4096,
    device: str = "cuda"
):
    """
    提取 AE Latent Codes 并保存为新的 TileDB 数据集。
    同时保留所有必要的元数据用于 Flow Training (next_cell_idx, tissue, etc.)
    """
    print(f"🚀 [Extract] Start extracting latents...")
    print(f"   - Data: {data_dir}")
    print(f"   - Output: {output_dir}")
    print(f"   - Model: {ckpt_path}")

    # 1. Load Model
    print("📦 Loading AE model...")
    model = AELitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)
    
    latent_dim = None
    
    # 2. Prepare Output
    if os.path.exists(output_dir):
        print(f"⚠️ Output directory exists. Cleaning up: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 3. Scan Shards
    sub_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"🔍 Found {len(sub_dirs)} shards.")
    
    ctx = tiledbsoma.SOMATileDBContext()
    
    for shard_name in tqdm(sub_dirs, desc="Processing Shards"):
        shard_uri = os.path.join(data_dir, shard_name)
        output_shard_uri = os.path.join(output_dir, shard_name)
        
        try:
            with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
                # A. Read Metadata (obs)
                obs_df = exp.obs.read().concat().to_pandas()
                n_cells = len(obs_df)
                if n_cells == 0: continue
                
                # B. Read Raw X
                # Assuming 'RNA' measurement
                if 'RNA' not in exp.ms:
                    print(f"⚠️ No RNA measurement in {shard_name}, skipping.")
                    continue
                    
                x_array = exp.ms['RNA'].X['data']
                
                # Determine input dim dynamically from first shard
                n_genes = x_array.shape[1]
                
                # Initialize Latent Storage
                # We need to run one batch to know latent dim if not known
                latents_buffer = []
                
                # Batch Processing
                n_batches = math.ceil(n_cells / batch_size)
                
                # Indices (assuming dense 0..N-1 for now, or match obs index)
                # SOMA read via coords. row_indices match obs index if not sparse-reindexed.
                # Usually soma_joinid is consistent.
                joinids = obs_df.index.values
                
                for i in range(n_batches):
                    batch_joinids = joinids[i*batch_size : (i+1)*batch_size]
                    
                    # Read Sparse/Dense Batch
                    # Read into table
                    tbl = x_array.read(coords=(batch_joinids, slice(None))).tables().concat()
                    
                    # Convert to Dense Tensor for Model
                    # We need to construct a dense matrix of shape (len(batch_joinids), n_genes)
                    rows = tbl['soma_dim_0'].to_numpy()
                    cols = tbl['soma_dim_1'].to_numpy()
                    data = tbl['soma_data'].to_numpy()
                    
                    # Re-map rows to 0..batch_len
                    # Create a map for this batch
                    id_map = {jid: idx for idx, jid in enumerate(batch_joinids)}
                    mapped_rows = np.array([id_map[r] for r in rows])
                    
                    dense_x = torch.zeros((len(batch_joinids), n_genes), dtype=torch.float32, device=device)
                    # Use index_put or sparse tensor logic
                    # To avoid CPU<->GPU sync overhead for large indexing, build on CPU then move? 
                    # Actually constructing on CPU is safer for memory.
                    dense_x_cpu = torch.zeros((len(batch_joinids), n_genes), dtype=torch.float32)
                    dense_x_cpu[mapped_rows, cols] = torch.from_numpy(data).float()
                    dense_x = dense_x_cpu.to(device)
                    
                    # Inference
                    with torch.no_grad():
                        # Check for VAE or AE interface
                        if hasattr(model.net, 'encode'):
                            res = model.net.encode(dense_x)
                            if isinstance(res, tuple):
                                z = res[0] # VAE: use mu
                            else:
                                z = res
                        else:
                            # Fallback
                            _, z = model(dense_x)
                    
                    latents_buffer.append(z.cpu().numpy())
                    
                    if latent_dim is None:
                        latent_dim = z.shape[1]
                        print(f"ℹ️ Detected Latent Dim: {latent_dim}")

                # Concatenate all latents
                full_latents = np.concatenate(latents_buffer, axis=0)
                
                # C. Write New Shard
                # Create Experiment
                tiledbsoma.Experiment.create(output_shard_uri)
                with tiledbsoma.Experiment.open(output_shard_uri, mode='w') as new_exp:
                    # Write obs (Copy metadata)
                    tiledbsoma.DataFrame.create(
                        os.path.join(output_shard_uri, "obs"),
                        schema=obs_df.to_arrow().schema,
                        domain=[(0, n_cells-1)]
                    ).write(obs_df.to_arrow())
                    new_exp.obs = tiledbsoma.open(os.path.join(output_shard_uri, "obs"))
                    
                    # Write Latent X
                    tiledbsoma.DenseNDArray.create(
                        os.path.join(output_shard_uri, "X_latent"),
                        type=pa.float32(),
                        shape=(n_cells, latent_dim)
                    ).write((slice(0, n_cells-1), slice(0, latent_dim-1)), pa.Tensor.from_numpy(full_latents))
                    
                    new_exp.set("X_latent", tiledbsoma.open(os.path.join(output_shard_uri, "X_latent")))

        except Exception as e:
            print(f"❌ Error processing {shard_name}: {e}")
            import traceback
            traceback.print_exc()

    print("✅ Extraction Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw TileDB data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save latent TileDB data")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    extract_latents(args.data_dir, args.output_dir, args.ckpt_path, device=args.device)

