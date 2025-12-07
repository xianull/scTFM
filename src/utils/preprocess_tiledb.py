import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import tiledb
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import gc
import scipy.sparse as sp
from typing import List, Dict, Optional, Tuple
import threading
import queue
import time

import shutil
import subprocess

# 忽略 Scanpy 的一些 FutureWarning
warnings.filterwarnings("ignore")

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gene_vocab(vocab_path: str) -> List[str]:
    """加载基因词表"""
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with open(path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes

def process_h5ad_vectorized(args) -> Optional[Dict]:
    """
    Worker 函数：处理单个 h5ad 文件
    """
    # 【优化】接收预构建好的 gene_map，而不是 list
    file_path, target_gene_map, target_genes_list, min_genes, target_sum, is_ood_flag = args
    
    try:
        # 1. 读取数据
        adata = sc.read_h5ad(file_path)

        # 2. 统一基因名为索引
        if "gene_symbols" in adata.var.columns:
            adata.var_names = adata.var["gene_symbols"].astype(str)
        adata.var_names_make_unique()

        # 3. 过滤细胞
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)

        if adata.shape[0] == 0:
            return None

        # 4. 归一化 (注意：这里会改变数据为 log1p)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # --- 核心向量化逻辑 ---
        
        # A. 找到交集基因 (利用 set 转换 list 加速 isin)
        mask = np.isin(adata.var_names, target_genes_list)
        
        # B. 切片
        adata_sub = adata[:, mask]
        
        if adata_sub.shape[1] == 0:
            return None 
            
        # C. 转 COO
        X_coo = adata_sub.X.tocoo()
        
        # D. 索引重映射
        sub_gene_names = adata_sub.var_names
        
        # 使用传入的 map 进行映射
        local_to_global = np.array([target_gene_map[g] for g in sub_gene_names], dtype=np.int64)
        new_gene_indices = local_to_global[X_coo.col]
        
        # 构建结果
        res = {
            'n_cells': adata.shape[0],
            'row_indices': X_coo.row.astype(np.int64),
            'col_indices': new_gene_indices,
            'values': X_coo.data.astype(np.float32),
            'is_ood': is_ood_flag,
            'file_path': str(file_path)
        }
        
        # 【安全】主动释放大对象内存
        del adata, adata_sub, X_coo
        gc.collect()
        
        return res

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def init_tiledb_array(output_dir: Path, n_genes: int):
    tiledb_path = output_dir / "all_data"
    
    if tiledb_path.exists():
        import shutil
        logger.warning(f"Output path exists, cleaning up: {tiledb_path}")
        shutil.rmtree(tiledb_path)
    tiledb_path.mkdir(parents=True)
    
    # 瓦片大小 4096 (scimilarity 推荐: 2048-4096)
    # 避免 GPFS 上的读放大
    tile_extent = 4096
    # 防止 int64 溢出
    max_domain = np.iinfo(np.int64).max - tile_extent - 1000
    
    # 默认压缩过滤器
    filters = [tiledb.ZstdFilter(level=4)]
    
    # 1. Counts Schema (Sparse)
    counts_uri = str(tiledb_path / "counts")
    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max_domain), tile=tile_extent, dtype=np.int64),
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1), tile=n_genes, dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom, sparse=True, 
        attrs=[tiledb.Attr(name="data", dtype=np.float32, filters=filters)], 
        allows_duplicates=False,
        coords_filters=filters,
        offsets_filters=filters
    )
    tiledb.Array.create(counts_uri, schema)
    
    # 2. Metadata Schema (Dense!)
    # 优化：使用 Dense Array 存储 Metadata，支持 O(1) 随机访问
    meta_uri = str(tiledb_path / "cell_metadata")
    dom_meta = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max_domain), tile=tile_extent, dtype=np.int64)
    )
    schema_meta = tiledb.ArraySchema(
        domain=dom_meta, sparse=False, # Changed to Dense
        attrs=[
            tiledb.Attr(name="is_ood", dtype=np.int8, filters=filters),
            tiledb.Attr(name="file_source", dtype='ascii', var=True, filters=filters) 
        ]
    )
    tiledb.Array.create(meta_uri, schema_meta)
    
    return tiledb_path

class AsyncBatchWriter:
    def __init__(self, tiledb_path: Path, batch_size: int = 500000):
        self.counts_uri = str(tiledb_path / "counts")
        self.meta_uri = str(tiledb_path / "cell_metadata")
        self.batch_size = batch_size
        self.global_cell_offset = 0
        
        self.current_buffer = self._init_buffer()
        self.current_count = 0
        
        # 【重要修复】设置 maxsize 防止内存爆炸 (背压机制)
        # 允许队列里存 3 个 batch，如果满了，主进程的 add() 会阻塞等待
        self.write_queue = queue.Queue(maxsize=3)
        
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.write_error = None

    def _init_buffer(self):
        return {
            'rows': [], 'cols': [], 'vals': [],
            'meta_indices': [], 'meta_ood': [], 'meta_src': []
        }

    def add(self, result: Dict):
        if self.write_error: raise self.write_error
        if not result: return
            
        n_cells = result['n_cells']
        global_rows = result['row_indices'] + self.global_cell_offset
        
        self.current_buffer['rows'].append(global_rows)
        self.current_buffer['cols'].append(result['col_indices'])
        self.current_buffer['vals'].append(result['values'])
        self.current_buffer['meta_indices'].append(np.arange(self.global_cell_offset, self.global_cell_offset + n_cells))
        self.current_buffer['meta_ood'].append(np.full(n_cells, result['is_ood'], dtype=np.int8))
        self.current_buffer['meta_src'].extend([result['file_path']] * n_cells)
        
        self.global_cell_offset += n_cells
        self.current_count += n_cells
        
        if self.current_count >= self.batch_size:
            self._push_to_queue()

    def _push_to_queue(self):
        if self.current_count == 0: return
        logger.info(f"  [Main] Batch full ({self.current_count} cells). Pushing to queue (Size: {self.write_queue.qsize()})...")
        
        task = (self.current_buffer, self.current_count)
        # put 默认是阻塞的，如果队列满了，这里会停下等待，保护内存
        self.write_queue.put(task)
        
        self.current_buffer = self._init_buffer()
        self.current_count = 0

    def _writer_loop(self):
        while self.is_running or not self.write_queue.empty():
            try:
                try:
                    buffer, count = self.write_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # --- [新增] 详细分段计时 ---
                logger.info(f"  [Debug] Start writing batch of {count} cells...")
                t_start = time.time()
                
                # 1. 内存拼接 (检测是否爆内存/Swap)
                all_rows = np.concatenate(buffer['rows'])
                all_cols = np.concatenate(buffer['cols'])
                all_vals = np.concatenate(buffer['vals'])
                
                t_concat = time.time()
                logger.info(f"  [Debug] Step 1: Numpy Concat used {t_concat - t_start:.2f}s") # 如果这里慢，说明内存爆了

                # 2. TileDB 上下文配置
                cfg = tiledb.Config({
                    "sm.compute_concurrency_level": "100",
                    "sm.io_concurrency_level": "16",
                    "vfs.file.enable_filelocks": "false", # 确保锁已关
                })
                ctx = tiledb.Ctx(cfg)
                
                # 3. 写 Counts (检测硬盘速度)
                with tiledb.open(self.counts_uri, 'w', ctx=ctx) as arr:
                    arr[all_rows, all_cols] = all_vals
                
                t_write_counts = time.time()
                logger.info(f"  [Debug] Step 2: Write Counts Disk IO used {t_write_counts - t_concat:.2f}s") # 如果这里慢，说明是 GPFS 的问题

                # 4. 写 Metadata (Dense Optimized)
                all_meta_indices = np.concatenate(buffer['meta_indices'])
                all_meta_ood = np.concatenate(buffer['meta_ood'])
                
                # Dense Array 写入优化：使用切片赋值
                start_idx = int(all_meta_indices[0])
                end_idx = int(all_meta_indices[-1]) + 1
                
                with tiledb.open(self.meta_uri, 'w', ctx=ctx) as arr:
                    arr[start_idx:end_idx] = {
                        'is_ood': all_meta_ood,
                        'file_source': np.array(buffer['meta_src'], dtype=object)
                    }
                
                # 5. 清理
                del buffer, all_rows, all_cols, all_vals
                gc.collect()
                
                logger.info(f"  [Async] Total Batch Time: {time.time()-t_start:.1f}s.")
                self.write_queue.task_done()
                
            except Exception as e:
                logger.error(f"Async Writer Crashed: {e}")
                self.write_error = e
                break

    def finish(self):
        self._push_to_queue()
        self.is_running = False
        self.writer_thread.join()
        if self.write_error: raise self.write_error

def consolidate_arrays(tiledb_path: Path):
    """合并 TileDB 碎片以优化读取性能"""
    logger.info("Starting consolidation...")
    
    # 1. Consolidate Counts
    counts_uri = str(tiledb_path / "counts")
    tiledb.consolidate(counts_uri)
    tiledb.vacuum(counts_uri)
    
    # 2. Consolidate Metadata
    meta_uri = str(tiledb_path / "cell_metadata")
    tiledb.consolidate(meta_uri)
    tiledb.vacuum(meta_uri)
    
    logger.info("Consolidation complete.")

def main():
    parser = argparse.ArgumentParser(description="Efficient TileDB Converter (In-Memory Fast Track)")
    parser.add_argument("--csv_path", type=str, default="data/assets/ae_data_info_1000.csv")
    parser.add_argument("--vocab_path", type=str, default="data/assets/gene_order.tsv")
    
    # 【最终目的地】GPFS 路径
    parser.add_argument("--final_output_dir", type=str, 
                        default="/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/tiledb_10m")
    
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--num_workers", type=int, default=64) # 保持 64 以防 OOM
    # 优化：设置为 Tile Extent (4096) 的整数倍，确保 Dense Array 写入对齐
    # 4096 * 128 = 524288
    parser.add_argument("--batch_size", type=int, default=524288)
    parser.add_argument("--max_files", type=int, default=1000)
    
    args = parser.parse_args()
    
    # --- 1. 设置极速内存路径 ---
    # 利用 Linux 的 /dev/shm (Shared Memory)，速度比 NVMe 快 10 倍，且没有文件锁延迟
    shm_path = Path("/dev/shm/tiledb_fast_buffer")
    final_path = Path(args.final_output_dir)
    
    logger.info("="*60)
    logger.info(f"🚀 SPEED MODE ACTIVATED")
    logger.info(f"1. Working Directory (RAM): {shm_path}")
    logger.info(f"2. Final Destination (SSD): {final_path}")
    logger.info("="*60)
    
    # 清理旧的内存缓存（如果存在）
    if shm_path.exists():
        logger.warning(f"Cleaning up previous buffer at {shm_path}...")
        shutil.rmtree(shm_path)
    
    # 2. 准备工作
    target_genes = load_gene_vocab(args.vocab_path)
    target_gene_map = {g: i for i, g in enumerate(target_genes)}
    
    info_df = pd.read_csv(args.csv_path)
    if args.max_files > 0:
        info_df = info_df.head(args.max_files)

    # 3. 初始化 TileDB (在内存盘上！)
    tiledb_path = init_tiledb_array(shm_path, len(target_genes))
    
    # 写入基因注释
    gene_annot_uri = str(tiledb_path / "gene_annotation")
    tiledb.Array.create(gene_annot_uri, tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="gene_index", domain=(0, len(target_genes)-1), tile=len(target_genes), dtype=np.int64)),
        sparse=False,
        attrs=[tiledb.Attr(name="gene_symbol", dtype='ascii', var=True)]
    ))
    with tiledb.open(gene_annot_uri, 'w') as arr:
        arr[:] = {'gene_symbol': np.array(target_genes, dtype=object)}

    # 4. 准备任务
    tasks = []
    for _, row in info_df.iterrows():
        is_ood = int(row.get('full_validation_dataset', 0))
        tasks.append((
            row['file_path'], target_gene_map, target_genes,
            args.min_genes, args.target_sum, is_ood
        ))

    # 5. 执行 (写入内存盘)
    writer = AsyncBatchWriter(tiledb_path, batch_size=args.batch_size)
    logger.info(f"Starting processing {len(tasks)} files...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_h5ad_vectorized, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing & Writing (RAM)"):
            try:
                result = future.result()
                if result: writer.add(result)
            except Exception as e:
                logger.error(f"Failed: {e}")

    writer.finish()
    
    # --- [新增] 合并碎片 (Consolidation) ---
    # 这一步对于 GPFS 性能至关重要，避免数千个小文件
    consolidate_arrays(tiledb_path)
    
    # 保存 Metadata
    metadata = {
        'total_cells': writer.global_cell_offset,
        'n_genes': len(target_genes),
        'storage_path': str(final_path) # 这里记录最终路径
    }
    with open(tiledb_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # --- 6. 最终搬运 (RAM -> GPFS) ---
    logger.info("="*60)
    logger.info("Processing Complete. Moving data from RAM to GPFS...")
    logger.info("This may take a while, but it's much faster than direct writing.")
    
    # 确保目标父目录存在
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        shutil.rmtree(final_path) # 如果目标已存在，先删除（慎用，或者改名）
    
    # 使用 rsync 进行搬运 (比 shutil 更稳健，显示进度)
    try:
        cmd = ["rsync", "-avP", str(shm_path) + "/", str(final_path) + "/"]
        subprocess.run(cmd, check=True)
        logger.info(f"✅ SUCCESS! Data moved to: {final_path}")
        
        # 搬运成功后，清理内存
        shutil.rmtree(shm_path)
        logger.info("RAM buffer cleaned.")
        
    except Exception as e:
        logger.error(f"❌ Error moving data: {e}")
        logger.error(f"⚠️ YOUR DATA IS STILL IN: {shm_path}. PLEASE MOVE IT MANUALLY!")

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()