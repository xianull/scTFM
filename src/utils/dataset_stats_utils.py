import os
import tiledbsoma
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Dict
import heapq

def _count_one_shard(args: Tuple[str, int]) -> Tuple[str, int]:
    """
    单个 Worker 的任务：计算一个分片里符合 split_label 的细胞数
    必须是顶层函数，以便 pickle 序列化。
    
    Returns:
        (shard_name, cell_count)
    """
    uri, split_label = args
    try:
        # 显式创建独立的 Context，避免多进程共享 Context 导致的 C++ 层死锁
        ctx = tiledbsoma.SOMATileDBContext()
        with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
            query = exp.obs.read(
                value_filter=f"split_label == {split_label}",
                column_names=["soma_joinid"]
            ).concat()
            shard_name = os.path.basename(uri)
            return (shard_name, len(query))
    except Exception:
        return (os.path.basename(uri), 0)

def balanced_shard_assignment(
    shard_sizes: Dict[str, int], 
    num_workers: int
) -> Dict[int, List[str]]:
    """
    使用贪心算法将 shards 分配给 workers，使每个 worker 的总细胞数尽可能均衡。
    
    算法：多路归并（类似 Multiway Number Partitioning）
    1. 维护一个最小堆，记录每个 worker 的当前总细胞数
    2. 对 shards 按大小降序排序
    3. 每次将最大的 shard 分配给当前负载最小的 worker
    
    Args:
        shard_sizes: {shard_name: cell_count}
        num_workers: 全局 worker 总数 (world_size * num_workers_per_gpu)
        
    Returns:
        {worker_id: [shard_names]}
    """
    # 初始化：每个 worker 的负载和分配列表
    # 使用最小堆：(当前总细胞数, worker_id)
    heap = [(0, i) for i in range(num_workers)]
    heapq.heapify(heap)
    
    # 每个 worker 分配到的 shards
    assignment = {i: [] for i in range(num_workers)}
    
    # 按 shard 大小降序排序（优先分配大 shard）
    sorted_shards = sorted(shard_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # 贪心分配
    for shard_name, cell_count in sorted_shards:
        if cell_count == 0:  # 跳过空 shard
            continue
        # 取出当前负载最小的 worker
        current_load, worker_id = heapq.heappop(heap)
        # 分配 shard
        assignment[worker_id].append(shard_name)
        # 更新负载并放回堆
        heapq.heappush(heap, (current_load + cell_count, worker_id))
    
    # 打印负载分布（用于调试）
    final_loads = sorted(heap, key=lambda x: x[0])
    min_load = final_loads[0][0]
    max_load = final_loads[-1][0]
    avg_load = sum(x[0] for x in final_loads) / len(final_loads)
    imbalance = (max_load - min_load) / avg_load * 100 if avg_load > 0 else 0
    
    print(f"⚖️  [Load Balancing] Workers: {num_workers}")
    print(f"   Min: {min_load:,} cells | Max: {max_load:,} cells | Avg: {avg_load:,.0f} cells")
    print(f"   Imbalance: {imbalance:.2f}%")
    
    return assignment

def get_dataset_stats(
    root_dir: str, 
    split_label: int, 
    batch_size: int, 
    num_workers: int = 16, 
    world_size: int = 1,
    num_workers_per_gpu: int = 16
) -> Tuple[int, int, Dict[str, int]]:
    """
    多进程并行扫描 TileDB 数据集，计算总细胞数、步数和每个 shard 的大小。
    
    Args:
        root_dir: 数据集根目录
        split_label: 0=Train, 1=Val
        batch_size: 单卡 Batch Size
        num_workers: 并行扫描的进程数 (建议设为 CPU 核心数的一半)
        world_size: DDP 总 GPU 数
        num_workers_per_gpu: 每个 GPU 的 DataLoader workers 数
        
    Returns:
        (total_cells, total_steps, shard_sizes_dict)
    """
    if not os.path.exists(root_dir):
        print(f"⚠️ [Stats] 路径不存在: {root_dir}")
        return 0, 0, {}
        
    sub_uris = sorted([
        os.path.join(root_dir, d) 
        for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    if not sub_uris:
        return 0, 0, {}
    
    print(f"📊 [Stats] 启动多进程扫描 {len(sub_uris)} 个 Shards (Split={split_label})...")
    
    # 准备任务参数
    tasks = [(uri, split_label) for uri in sub_uris]
    
    # 动态调整 worker 数，不超过任务数也不超过 CPU 核心数
    max_workers = min(num_workers, len(tasks), os.cpu_count() or 1)

    # 使用 ProcessPoolExecutor 并行处理
    shard_sizes = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_count_one_shard, tasks)
        for shard_name, cell_count in results:
            shard_sizes[shard_name] = cell_count
    
    total_cells = sum(shard_sizes.values())
    
    # 计算 DDP 环境下的 Global Batch Size
    global_batch_size = batch_size * world_size
    if global_batch_size == 0:
        return 0, 0, shard_sizes
        
    total_steps = math.ceil(total_cells / global_batch_size)
    
    print(f"✅ [Stats] 完成: {total_cells} cells | Global Batch: {global_batch_size} | Epoch Steps: {total_steps}")
    
    return total_cells, total_steps, shard_sizes
