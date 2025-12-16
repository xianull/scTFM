"""
流式版 TileDB 预处理脚本：适配超大规模数据集（1亿+ 细胞）
内存占用优化：不加载全部数据，使用固定大小的 Buffer 流式写入

关键设计：
1. Cell Buffer：只在内存中保持 BUFFER_SIZE 个细胞（如 20 万）
2. 流式写入：Buffer 满了立即写入 Shard，释放内存
3. 全局打乱：通过文件级随机读取 + Shard 内打乱实现数据均匀分布
4. 内存峰值：~10-20GB（取决于 BUFFER_SIZE）

适用场景：
- 数据量：1亿+ 细胞
- 可用内存：<2TB
- 目标：生成均匀大小的 Shards
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
import gc

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CSV_PATH = '/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/ae_data_info.csv'
GENE_ORDER_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/gene_order.tsv"
OUTPUT_BASE_URI = "/fast/data/scTFM/ae/tile_4000_stream"

CELLS_PER_SHARD = 8192      # 每个 Shard 的目标细胞数
BUFFER_SIZE = 409600        # 内存中最多缓存的细胞数（40万细胞 ≈ 10-20GB 内存）
MAX_WORKERS = 16            # 并行读取文件的进程数
SHUFFLE_FILES = True        # 是否随机打乱文件读取顺序（确保数据分布均匀）

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
            return None, f"Missing"
        
        # 1. 读取数据
        adata = sc.read_h5ad(file_path)
        
        # 2. 准备变量名
        adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names_make_unique()
        
        # 3. 过滤低质量细胞
        sc.pp.filter_cells(adata, min_genes=200)
        
        if adata.n_obs == 0:
            return None, "Skipped"
        
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
        return None, f"Error: {str(e)[:50]}"


class StreamingShardWriter:
    """
    流式 Shard 写入器
    维护一个固定大小的 Cell Buffer，满了就写入磁盘
    """
    def __init__(self, output_base, cells_per_shard=4000, buffer_size=200000):
        self.output_base = output_base
        self.cells_per_shard = cells_per_shard
        self.buffer_size = buffer_size
        
        # 清空输出目录
        if os.path.exists(output_base):
            print(f"🗑️  Removing existing output: {output_base}")
            shutil.rmtree(output_base)
        os.makedirs(output_base)
        print(f"📁 Created output directory: {output_base}")
        
        # 状态
        self.buffer = []           # 当前 Buffer 中的 AnnData 列表
        self.buffer_cells = 0      # Buffer 中的细胞总数
        self.shard_idx = 0         # 当前 Shard 索引
        self.total_cells = 0       # 已处理的总细胞数
        
    def add_adata(self, adata):
        """
        添加一个 AnnData 到 Buffer
        如果 Buffer 满了，自动 flush
        """
        self.buffer.append(adata)
        self.buffer_cells += adata.n_obs
        self.total_cells += adata.n_obs
        
        # Buffer 满了，写入磁盘
        if self.buffer_cells >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """
        将 Buffer 中的所有细胞合并、打乱、切分成多个 Shards 并写入
        """
        if len(self.buffer) == 0:
            return
        
        n_shards_in_buffer = (self.buffer_cells + self.cells_per_shard - 1) // self.cells_per_shard
        print(f"\n💾 Flushing buffer: {self.buffer_cells:,} cells → {n_shards_in_buffer} shards...")
        
        # 1. 合并 Buffer
        print(f"   1/4 Concatenating {len(self.buffer)} AnnData objects...")
        combined = ad.concat(self.buffer, join='outer', merge='same')
        n_cells = combined.n_obs
        print(f"       ✓ Merged: {n_cells:,} cells")
        
        # 2. 打乱（确保 Shard 内数据分布均匀）
        print(f"   2/4 Shuffling cells...")
        perm = np.random.permutation(n_cells)
        combined = combined[perm, :].copy()
        print(f"       ✓ Shuffled")
        
        # 3. 切分成多个 Shards
        print(f"   3/4 Splitting into {n_shards_in_buffer} shards...")
        
        # 4. 写入（带进度条）
        print(f"   4/4 Writing to TileDB...")
        for i in tqdm(range(n_shards_in_buffer), desc="       Writing", leave=False):
            start = i * self.cells_per_shard
            end = min(start + self.cells_per_shard, n_cells)
            
            shard_data = combined[start:end, :].copy()
            
            shard_name = f"shard_{self.shard_idx:05d}"
            shard_uri = os.path.join(self.output_base, shard_name)
            
            tiledbsoma.io.from_anndata(
                experiment_uri=shard_uri,
                anndata=shard_data,
                measurement_name="RNA"
            )
            
            self.shard_idx += 1
        
        print(f"       ✓ Wrote shards {self.shard_idx - n_shards_in_buffer} to {self.shard_idx - 1}")
        
        # 5. 清空 Buffer，释放内存
        self.buffer.clear()
        self.buffer_cells = 0
        del combined
        gc.collect()
    
    def finalize(self):
        """
        处理完所有文件后，写入剩余的 Buffer
        """
        if len(self.buffer) > 0:
            self._flush_buffer()
        
        print(f"\n✅ Streaming write completed:")
        print(f"   Total shards: {self.shard_idx}")
        print(f"   Total cells: {self.total_cells:,}")
        print(f"   Avg cells/shard: {self.total_cells / max(1, self.shard_idx):.1f}")


def streaming_process(df, writer, max_workers=16, shuffle=True):
    """
    流式处理所有文件，边读边写
    """
    print(f"\n🚀 Starting streaming processing...")
    print(f"   Files: {len(df)}")
    print(f"   Workers: {max_workers}")
    print(f"   Buffer size: {writer.buffer_size:,} cells")
    print(f"   Shard size: {writer.cells_per_shard} cells")
    
    # 随机打乱文件顺序（确保数据分布均匀）
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"   🔀 File order shuffled")
    
    tasks = [(row['file_path'], row['full_validation_dataset']) 
             for _, row in df.iterrows()]
    
    stats = {"Success": 0, "Skipped": 0, "Errors": 0}
    
    with ProcessPoolExecutor(max_workers=max_workers, 
                             initializer=worker_init, 
                             initargs=(global_target_genes,)) as executor:
        
        # 提交所有任务
        futures = {executor.submit(load_and_process_one_file, path, is_val): (idx, path) 
                   for idx, (path, is_val) in enumerate(tasks)}
        
        pbar = tqdm(total=len(futures), desc="Processing & Writing")
        
        for future in as_completed(futures):
            idx, file_path = futures[future]
            
            try:
                adata, status = future.result()
                
                if status == "Success":
                    # 立即写入 Writer（可能触发 flush）
                    writer.add_adata(adata)
                    stats["Success"] += 1
                    
                    # 更新进度条信息
                    pbar.set_postfix({
                        "Cells": f"{writer.total_cells:,}",
                        "Shards": writer.shard_idx,
                        "Buffer": f"{writer.buffer_cells:,}"
                    })
                    
                elif status.startswith("Skipped"):
                    stats["Skipped"] += 1
                else:
                    stats["Errors"] += 1
                    
            except Exception as exc:
                tqdm.write(f"❌ Critical exception: {exc}")
                stats["Errors"] += 1
            
            pbar.update(1)
        
        pbar.close()
    
    print(f"\n📊 Processing stats:")
    print(f"   ✅ Success: {stats['Success']}")
    print(f"   ⏭️  Skipped: {stats['Skipped']}")
    print(f"   ❌ Errors: {stats['Errors']}")


if __name__ == "__main__":
    print("="*70)
    print("TileDB 预处理脚本 - 流式处理版本（适配超大规模数据集）")
    print("="*70)
    
    # 1. 加载基因顺序
    print("\n📖 Loading gene order...")
    target_genes = pd.read_csv(GENE_ORDER_PATH, sep='\t', header=None)[0].values
    global_target_genes = target_genes
    global_target_gene_map = {gene: i for i, gene in enumerate(target_genes)}
    print(f"   Target genes: {len(target_genes):,}")
    
    # 2. 加载文件列表
    print("\n📋 Loading file list...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total files: {len(df):,}")
    
    # 3. 创建流式写入器
    writer = StreamingShardWriter(
        output_base=OUTPUT_BASE_URI,
        cells_per_shard=CELLS_PER_SHARD,
        buffer_size=BUFFER_SIZE
    )
    
    # 4. 流式处理
    streaming_process(
        df=df,
        writer=writer,
        max_workers=MAX_WORKERS,
        shuffle=SHUFFLE_FILES
    )
    
    # 5. 完成写入
    writer.finalize()
    
    print("\n" + "="*70)
    print("✅ All done!")
    print(f"💾 Data saved to: {OUTPUT_BASE_URI}/")
    print("="*70)

