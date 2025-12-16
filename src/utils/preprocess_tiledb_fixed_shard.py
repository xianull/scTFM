"""
重构版 TileDB 预处理脚本：按固定细胞数切分 Shard
解决原版脚本中 Shard 大小不均导致的训练负载不均衡问题

核心改进：
1. 不再 1 文件 = 1 Shard，而是 N 个细胞 = 1 Shard
2. 确保每个 Shard 大小一致（最后一个可能略小）
3. 减少 Shard 总数，提升 I/O 效率
4. 更好的 DDP 负载均衡
"""

import multiprocessing
import os
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import shutil
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse
import tiledbsoma
import tiledbsoma.io
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CSV_PATH = '/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/ae_data_info.csv'
GENE_ORDER_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/gene_order.tsv"
OUTPUT_BASE_URI = "/fast/data/scTFM/ae/tile_4000_fix"

CELLS_PER_SHARD = 4000  # 每个 Shard 的目标细胞数
MAX_WORKERS = 16        # 并行读取文件的进程数

# ==================== 全局变量 ====================
global_target_genes = None
global_target_gene_map = None

def worker_init(gene_list):
    """子进程初始化：加载目标基因列表"""
    global global_target_genes, global_target_gene_map
    global_target_genes = gene_list
    global_target_gene_map = {gene: i for i, gene in enumerate(gene_list)}

def load_and_process_one_file(file_path, is_full_val):
    """
    加载并预处理单个 h5ad 文件
    返回处理后的 AnnData（已对齐、归一化、log1p）
    """
    try:
        if not os.path.exists(file_path):
            return None, f"Missing: {file_path}"
        
        # 1. 读取数据
        adata = sc.read_h5ad(file_path)
        
        # 2. 准备变量名
        adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names_make_unique()
        
        # 3. 过滤低质量细胞
        sc.pp.filter_cells(adata, min_genes=200)
        
        if adata.n_obs == 0:
            return None, "Skipped (Low quality)"
        
        # 4. 基因对齐
        target_genes = global_target_genes
        target_n_vars = len(target_genes)
        target_gene_map = global_target_gene_map
        
        common_genes = [g for g in adata.var_names if g in target_gene_map]
        
        if len(common_genes) == 0:
            new_X = scipy.sparse.csr_matrix((adata.n_obs, target_n_vars), dtype=np.float32)
            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes
        else:
            adata = adata[:, common_genes].copy()
            
            if not scipy.sparse.isspmatrix_csr(adata.X):
                adata.X = adata.X.tocsr()
            
            current_col_to_target_col = np.array(
                [target_gene_map[g] for g in adata.var_names], 
                dtype=np.int32
            )
            new_indices = current_col_to_target_col[adata.X.indices]
            
            new_X = scipy.sparse.csr_matrix(
                (adata.X.data, new_indices, adata.X.indptr),
                shape=(adata.n_obs, target_n_vars)
            )
            new_X.sort_indices()
            
            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes
        
        # 5. 归一化和 log1p
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # 6. 打标签
        if is_full_val == 1:
            adata.obs['split_label'] = 3  # Test OOD
        else:
            n_cells = adata.n_obs
            split_labels = np.random.choice(
                [0, 1, 2], 
                size=n_cells, 
                p=[0.9, 0.05, 0.05]
            )
            adata.obs['split_label'] = split_labels
        
        adata.obs['split_label'] = adata.obs['split_label'].astype(np.int32)
        
        # 7. 确保 float32
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)
        
        return adata, "Success"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def batch_load_files(df, max_workers=16):
    """
    并行加载所有文件，返回 AnnData 列表
    """
    print(f"📂 Loading {len(df)} files with {max_workers} workers...")
    
    tasks = [(row['file_path'], row['full_validation_dataset']) 
             for _, row in df.iterrows()]
    
    all_adatas = []
    stats = {"Success": 0, "Skipped": 0, "Errors": 0}
    
    with ProcessPoolExecutor(max_workers=max_workers, 
                             initializer=worker_init, 
                             initargs=(global_target_genes,)) as executor:
        
        futures = {executor.submit(load_and_process_one_file, path, is_val): path 
                   for path, is_val in tasks}
        
        pbar = tqdm(total=len(futures), desc="Loading H5ADs")
        
        for future in as_completed(futures):
            try:
                adata, status = future.result()
                
                if status == "Success":
                    all_adatas.append(adata)
                    stats["Success"] += 1
                elif status.startswith("Skipped"):
                    stats["Skipped"] += 1
                else:
                    stats["Errors"] += 1
                    
            except Exception as exc:
                tqdm.write(f"Critical exception: {exc}")
                stats["Errors"] += 1
            
            pbar.update(1)
        
        pbar.close()
    
    print(f"✅ Loaded: {stats['Success']} | ⏭️  Skipped: {stats['Skipped']} | ❌ Errors: {stats['Errors']}")
    return all_adatas

def write_shards(all_adatas, output_base, cells_per_shard=4000):
    """
    将所有 AnnData 按固定细胞数切分成多个 Shard 并写入 TileDB
    """
    if os.path.exists(output_base):
        print(f"🗑️  Removing existing output directory: {output_base}")
        shutil.rmtree(output_base)
    
    os.makedirs(output_base)
    print(f"📁 Created output directory: {output_base}")
    
    # 1. 合并所有数据（注意：如果内存不足，可改为流式处理）
    print("🔗 Concatenating all AnnData objects...")
    combined = ad.concat(all_adatas, join='outer', merge='same')
    
    total_cells = combined.n_obs
    n_shards = (total_cells + cells_per_shard - 1) // cells_per_shard
    
    print(f"📊 Total cells: {total_cells:,}")
    print(f"📦 Target shard size: {cells_per_shard} cells/shard")
    print(f"📦 Will create {n_shards} shards")
    
    # 2. 打乱顺序（确保每个 Shard 的数据分布均匀）
    print("🔀 Shuffling cells for balanced shards...")
    perm = np.random.permutation(total_cells)
    combined = combined[perm, :].copy()
    
    # 3. 切分并写入
    print(f"💾 Writing shards to {output_base}/...")
    
    for shard_idx in tqdm(range(n_shards), desc="Writing Shards"):
        start = shard_idx * cells_per_shard
        end = min(start + cells_per_shard, total_cells)
        
        shard_data = combined[start:end, :].copy()
        
        # Shard 命名：shard_0000, shard_0001, ...
        shard_name = f"shard_{shard_idx:04d}"
        shard_uri = os.path.join(output_base, shard_name)
        
        tiledbsoma.io.from_anndata(
            experiment_uri=shard_uri,
            anndata=shard_data,
            measurement_name="RNA"
        )
    
    print(f"✅ Successfully created {n_shards} shards with ~{cells_per_shard} cells each")
    return n_shards

if __name__ == "__main__":
    print("="*60)
    print("TileDB 预处理脚本 - 固定 Shard 大小版本")
    print("="*60)
    
    # 1. 加载基因顺序
    print("\n📖 Loading gene order...")
    target_genes = pd.read_csv(GENE_ORDER_PATH, sep='\t', header=None)[0].values
    global_target_genes = target_genes
    global_target_gene_map = {gene: i for i, gene in enumerate(target_genes)}
    print(f"   Target genes: {len(target_genes)}")
    
    # 2. 加载文件列表
    print("\n📋 Loading file list...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total files: {len(df)}")
    
    # 3. 并行加载所有文件
    all_adatas = batch_load_files(df, max_workers=MAX_WORKERS)
    
    if len(all_adatas) == 0:
        print("❌ No valid data loaded. Exiting.")
        exit(1)
    
    # 4. 切分并写入 Shard
    n_shards = write_shards(all_adatas, OUTPUT_BASE_URI, cells_per_shard=CELLS_PER_SHARD)
    
    print("\n" + "="*60)
    print("✅ Processing completed!")
    print(f"📦 Created {n_shards} shards")
    print(f"💾 Data saved to: {OUTPUT_BASE_URI}/")
    print("="*60)

