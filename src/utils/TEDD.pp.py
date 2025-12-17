#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import os
# tiledb底层是c，用fork会导致每个进程都初始化实例
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import shutil
import random
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

warnings.filterwarnings('ignore')


# In[2]:


# ============ 时间解析部分 ============
from format_time_h5ad import process_anndata_timepoints

# In[3]:


import cellrank as cr
from cellrank.kernels import RealTimeKernel
from cellrank.estimators import GPCCA
from moscot.problems.time import TemporalProblem


# In[4]:


CSV_PATH = '/fast/data/scTFM/rtf/TEDD/RTF_Data.TEDD.csv'
GENE_ORDER_PATH = "/gpfs/hybrid/data/public/SCimilarity/model/model_v1.1/gene_order.tsv"
OUTPUT_BASE_URI = "/fast/data/scTFM/rtf/TEDD/tile_4000_fix"
NUM_SAMPLES = 98
MAX_WORKERS = 4


# In[5]:


# 定义全局变量
# 这样每个线程读取的时候共享同一个变量
global_target_genes = None
global_target_gene_map = None

def worker_init(gene_list):
    """子进程初始化函数：加载目标基因列表"""
    global global_target_genes, global_target_gene_map
    global_target_genes = gene_list
    global_target_gene_map = {gene: i for i, gene in enumerate(gene_list)}


# In[6]:


def get_most_probable_next_cell_v2(adata, transition_matrix, time_key='time'):
    """
    只保留概率最大的后一个细胞的细胞ID - 从前往后推理版本
    默认值: None (表示无下一个细胞)
    
    参数:
    - adata: AnnData对象
    - transition_matrix: 转移矩阵 (稀疏矩阵)
    - time_key: 时间列的名称
    
    返回:
    - 更新后的adata，每个细胞只包含一个最可能的下一个细胞ID
    """
    import numpy as np
    import pandas as pd
    
    # 转换时间数据为数值型
    times = pd.to_numeric(adata.obs[time_key], errors='coerce').values
    
    # 检查NaN值
    valid_mask = ~np.isnan(times)
    if not np.all(valid_mask):
        print("警告：时间数据中存在NaN值，将被忽略")
        adata_valid = adata[valid_mask].copy()
        times = times[valid_mask]
        matrix = transition_matrix[valid_mask][:, valid_mask]
    else:
        adata_valid = adata
        matrix = transition_matrix
    
    n_cells = adata_valid.n_obs
    
    # 初始化结果数组，默认值为None
    next_cell_ids = np.full(n_cells, None, dtype=object)  # 使用None而不是空字符串
    next_cell_probs = np.zeros(n_cells, dtype=float)
    
    # 为每个细胞找到最可能的下一个细胞
    for i in range(n_cells):
        current_time = times[i]
        
        # 找到后一个时间点的细胞
        future_mask = times > current_time
        if not np.any(future_mask):
            continue
            
        # 获取转移概率
        future_probs = matrix[i, future_mask].toarray().flatten()
        future_indices = np.where(future_mask)[0]
        
        # 找到最大概率
        if len(future_probs) > 0 and np.any(future_probs > 0):
            max_idx = np.argmax(future_probs)
            max_prob = future_probs[max_idx]
            
            if max_prob > 0:
                # 获取对应的细胞ID（不是行号索引）
                next_cell_ids[i] = adata_valid.obs_names[future_indices[max_idx]]
                next_cell_probs[i] = max_prob
    
    # 将结果添加到原始adata
    adata.obs['next_cell_id'] = None  # 默认值None
    adata.obs['next_cell_probability'] = 0.0
    
    if not np.all(valid_mask):
        # 处理有NaN值的情况
        valid_obs_names = adata_valid.obs_names
        for i, obs_name in enumerate(valid_obs_names):
            if obs_name in adata.obs_names:
                orig_idx = adata.obs_names.get_loc(obs_name)
                adata.obs['next_cell_id'].iloc[orig_idx] = next_cell_ids[i]
                adata.obs['next_cell_probability'].iloc[orig_idx] = next_cell_probs[i]
    else:
        adata.obs['next_cell_id'] = next_cell_ids
        adata.obs['next_cell_probability'] = next_cell_probs
    
    return adata


def get_most_probable_prev_cell_v2(adata, transition_matrix, time_key='time'):
    """
    只保留概率最大的前一个细胞的细胞ID - 从后往前推理版本
    默认值: None (表示无前一个细胞)
    
    参数:
    - adata: AnnData对象
    - transition_matrix: 转移矩阵 (稀疏矩阵)，T[i,j]表示从细胞i到细胞j的转移概率
    - time_key: 时间列的名称
    
    返回:
    - 更新后的adata，每个细胞只包含一个最可能的前一个细胞ID
    """
    import numpy as np
    import pandas as pd
    
    # 转换时间数据为数值型
    times = pd.to_numeric(adata.obs[time_key], errors='coerce').values
    
    # 检查NaN值
    valid_mask = ~np.isnan(times)
    if not np.all(valid_mask):
        print("警告：时间数据中存在NaN值，将被忽略")
        adata_valid = adata[valid_mask].copy()
        times = times[valid_mask]
        matrix = transition_matrix[valid_mask][:, valid_mask]
    else:
        adata_valid = adata
        matrix = transition_matrix
    
    n_cells = adata_valid.n_obs
    
    # 初始化结果数组，默认值为None
    prev_cell_ids = np.full(n_cells, None, dtype=object)  # 使用None而不是空字符串
    prev_cell_probs = np.zeros(n_cells, dtype=float)
    
    # 关键：我们需要处理传入概率（矩阵的列）
    matrix_T = matrix.T.tocsr()  # 转置以便行访问
    
    # 为每个细胞找到最可能的前一个细胞
    for j in range(n_cells):  # j是当前细胞（接收转移）
        current_time = times[j]
        
        # 找到时间更早的细胞（可能的来源）
        past_mask = times < current_time
        if not np.any(past_mask):
            continue
            
        # 获取传入概率（矩阵的第j列）
        incoming_probs = matrix_T[j].toarray().flatten()
        
        # 只考虑时间更早的细胞的传入概率
        incoming_probs[~past_mask] = 0
        
        # 找到最大概率
        if np.any(incoming_probs > 0):
            max_idx = np.argmax(incoming_probs)
            max_prob = incoming_probs[max_idx]
            
            if max_prob > 0:
                # 获取对应的细胞ID（不是行号索引）
                prev_cell_ids[j] = adata_valid.obs_names[max_idx]
                prev_cell_probs[j] = max_prob
    
    # 将结果添加到原始adata
    adata.obs['prev_cell_id'] = None  # 默认值None
    adata.obs['prev_cell_probability'] = 0.0
    
    if not np.all(valid_mask):
        # 处理有NaN值的情况
        valid_obs_names = adata_valid.obs_names
        for j, obs_name in enumerate(valid_obs_names):
            if obs_name in adata.obs_names:
                orig_idx = adata.obs_names.get_loc(obs_name)
                adata.obs['prev_cell_id'].iloc[orig_idx] = prev_cell_ids[j]
                adata.obs['prev_cell_probability'].iloc[orig_idx] = prev_cell_probs[j]
    else:
        adata.obs['prev_cell_id'] = prev_cell_ids
        adata.obs['prev_cell_probability'] = prev_cell_probs
    
    return adata


# In[7]:


def cellrank_link_cell(adata):
    # 确保有PCA和邻居图
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=30)
    
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    
    # 在运行函数之前，先确保时间数据是数值型
    print("原始时间数据类型:", adata.obs['time'].dtype)
    
    # 如果是分类变量，转换为数值
    if adata.obs['time'].dtype.name == 'category':
        # 方法1: 如果分类是有序的，直接使用codes
        adata.obs['time'] = adata.obs['time'].astype(str).astype(float)
    else:
        # 确保是浮点数
        adata.obs['time'] = adata.obs['time'].astype(float)
    
    print("转换后的时间数据类型:", adata.obs['time'].dtype)
    
    # 2. 创建和计算RealTimeKernel
    
    # 创建TemporalProblem
    tp = TemporalProblem(adata)
    
    # 准备数据
    tp = tp.prepare(
            time_key='time',
            joint_attr='X_pca'  # 或使用其他降维结果
            )
    
    # 求解最优传输问题
    tp = tp.solve(
            epsilon=1e-3,
            tau_a=0.95,
            tau_b=0.95,
            scale_cost='mean'
            )
    
    # 从Moscot结果创建RealTimeKernel
    rtk = RealTimeKernel.from_moscot(tp)
    
    # 计算转移矩阵
    rtk.compute_transition_matrix(
            self_transitions='all',
            conn_weight=0.2,
            threshold='auto'
            )
    # 3. 提取细胞关系
    adata = get_most_probable_next_cell_v2(adata, rtk.transition_matrix, 'time')
    adata = get_most_probable_prev_cell_v2(adata, rtk.transition_matrix, 'time')

    return adata


# In[8]:


def process_single_file(row_data):
    """
    处理单个文件的核心逻辑
    """
    idx, row = row_data
    file_path = row['file_path']
    is_full_val = row['full_validation_dataset']
    
    # 用文件名作为 ID，而不是随机 UUID
    try:
        file_name = os.path.basename(file_path)       # 获取文件名 "SRX21870170.h5ad"
        sample_id = os.path.splitext(file_name)[0]    # 去掉后缀 "SRX21870170"
        soma_uri = os.path.join(OUTPUT_BASE_URI, sample_id)
        
        if not os.path.exists(file_path):
            return f"Missing: {file_path}"

        # 1. 读取数据
        adata = sc.read_h5ad(file_path)
        
        # 为每个细胞的obs_names添加样本ID前缀
        adata.obs_names = f"{sample_id}_" + adata.obs_names

        # 处理时间点信息 添加time列
        adata = process_anndata_timepoints(adata, time_col='Timepoint', new_col='time')
        
        # 2. 准备变量名
        #adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()
        
        # 先过滤细胞质量，再切片
        # 只要原始数据中检测到 >200 个基因，就视为有效细胞
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_cells(adata, min_counts=1)  # 确保每个细胞至少有1个UMI
        
        if adata.n_obs == 0:
            return "Skipped (Low quality raw cells)"

        # 3. 极速对齐逻辑
        target_genes = global_target_genes
        target_n_vars = len(target_genes)
        target_gene_map = global_target_gene_map

        # 计算交集
        common_genes = [g for g in adata.var_names if g in target_gene_map]
        
        if len(common_genes) == 0:
            # 全零矩阵 (但保留了有效细胞的占位)
            new_X = scipy.sparse.csr_matrix((adata.n_obs, target_n_vars), dtype=np.float32)
            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes
        else:
            # 先切片保留交集基因
            adata = adata[:, common_genes].copy()
        
            # 极速映射
            if not scipy.sparse.isspmatrix_csr(adata.X):
                #adata.X = adata.X.tocsr()
                adata.X = scipy.sparse.csr_matrix(adata.X)
            
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
        
        # 4. 后续处理
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        cellrank_link_cell(adata)
        
        # 5. 打标签
        if is_full_val == 1:
            adata.obs['split_label'] = 3
        else:
            n_cells = adata.n_obs
            split_labels = np.random.choice(
                [0, 1, 2], 
                size=n_cells, 
                p=[0.9, 0.05, 0.05]
            )
            adata.obs['split_label'] = split_labels
        
        adata.obs['split_label'] = adata.obs['split_label'].astype(np.int32)
        
        # 确保 float32
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)

        # 6. 写入 SOMA (如果目录已存在，先删除旧的，避免冲突)
        if os.path.exists(soma_uri):
            shutil.rmtree(soma_uri)
            
        tiledbsoma.io.from_anndata(
            experiment_uri=soma_uri,
            anndata=adata,
            measurement_name="RNA"
        )
        return "Success"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_BASE_URI):
        os.makedirs(OUTPUT_BASE_URI)
        print(f"Created output directory: {OUTPUT_BASE_URI}")
    else:
        print(f"Output directory exists: {OUTPUT_BASE_URI}")
    
    # 2. 读取数据
    print("Loading gene order...")
    target_genes = pd.read_csv(GENE_ORDER_PATH, sep='\t', header=None)[0].values
    
    print("Loading and sampling CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # 采样
    if len(df) > NUM_SAMPLES:
        sampled_df = df.sample(n=NUM_SAMPLES, random_state=42)
    else:
        sampled_df = df
        print(f"Warning: CSV only has {len(df)} rows, using all.")
    
    print(f"Target sample size: {len(sampled_df)}")
    
    tasks = list(sampled_df.iterrows())
    
    # 4. 多进程处理
    print(f"Starting parallel processing with {MAX_WORKERS} workers...")
    
    results = {
        "Success": 0,
        "Skipped (Low quality raw cells)": 0,
        "Errors": 0
    }
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_init, initargs=(target_genes,)) as executor:
        future_to_file = {executor.submit(process_single_file, task): task[1]['file_path'] for task in tasks}
        
        pbar = tqdm(total=len(tasks), desc="Processing H5ADs")
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                status = future.result()
                
                if status == "Success":
                    results["Success"] += 1
                elif status.startswith("Skipped"):
                    # 这里的 Key 要和上面 results 里的 key 对应
                    results["Skipped (Low quality raw cells)"] += 1
                else:
                    results["Errors"] += 1
                    
            except Exception as exc:
                tqdm.write(f"Critical exception for {file_path}: {exc}")
                results["Errors"] += 1
            
            pbar.update(1)
            
        pbar.close()

    print("\n" + "="*30)
    print("处理完成")
    print(f"Success: {results['Success']}")
    print(f"Skipped: {results['Skipped (Low quality raw cells)']}")
    print(f"Errors : {results['Errors']}")
    print(f"数据保存到目录: {OUTPUT_BASE_URI}/")
    print("="*30)


# In[ ]:




