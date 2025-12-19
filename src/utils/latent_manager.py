"""
Latent 数据自动管理模块

功能：
1. 检测 latent 数据是否存在
2. 根据 raw 数据目录自动推理 latent 目录路径
3. 在需要时自动提取 latent

使用方式：
在 train.py 中调用 ensure_latent_data() 即可自动处理
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import torch
import numpy as np
import tiledbsoma
import tiledbsoma.io
import anndata as ad
import scipy.sparse
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_latent_dir(raw_data_dir: str, latent_suffix: str = "_latents") -> str:
    """
    根据 raw 数据目录推理 latent 目录路径。

    示例：
        /fast/data/scTFM/rtf/TEDD/tile_4000_fix
        → /fast/data/scTFM/rtf/TEDD/tile_4000_fix_latents

    Args:
        raw_data_dir: 原始 TileDB 数据目录
        latent_suffix: latent 目录后缀（默认 "_latents"）

    Returns:
        latent 目录路径
    """
    raw_path = Path(raw_data_dir).resolve()
    return str(raw_path.parent / f"{raw_path.name}{latent_suffix}")


def check_latent_exists(latent_dir: str, raw_data_dir: str) -> Tuple[bool, str]:
    """
    检查 latent 数据是否存在且有效。

    验证条件：
    1. latent 目录存在
    2. latent 目录下有子目录（shards）
    3. shard 数量与 raw 数据一致
    4. 每个 shard 包含有效的 TileDB-SOMA 结构

    Args:
        latent_dir: latent 数据目录
        raw_data_dir: 原始数据目录（用于验证 shard 数量）

    Returns:
        (is_valid, message)
    """
    latent_path = Path(latent_dir)
    raw_path = Path(raw_data_dir)

    # 1. 检查目录是否存在
    if not latent_path.exists():
        return False, f"Latent 目录不存在: {latent_dir}"

    # 2. 获取 shard 列表
    latent_shards = sorted([
        d for d in os.listdir(latent_dir)
        if os.path.isdir(os.path.join(latent_dir, d))
    ])

    raw_shards = sorted([
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ])

    if len(latent_shards) == 0:
        return False, f"Latent 目录为空: {latent_dir}"

    # 3. 检查 shard 数量是否一致
    if len(latent_shards) != len(raw_shards):
        return False, f"Shard 数量不匹配: latent={len(latent_shards)}, raw={len(raw_shards)}"

    # 4. 抽样检查几个 shard 是否有效
    sample_shards = latent_shards[:min(3, len(latent_shards))]
    ctx = tiledbsoma.SOMATileDBContext()

    for shard_name in sample_shards:
        shard_uri = os.path.join(latent_dir, shard_name)
        try:
            with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
                # 检查是否有 obs 和 X 数据
                obs = exp.obs.read().concat().to_pandas()
                if 'next_cell_id' not in obs.columns:
                    return False, f"Shard {shard_name} 缺少 next_cell_id 列"
                if 'time' not in obs.columns:
                    return False, f"Shard {shard_name} 缺少 time 列"
        except Exception as e:
            return False, f"Shard {shard_name} 读取失败: {e}"

    return True, f"Latent 数据有效 ({len(latent_shards)} shards)"


def extract_latents(
    raw_data_dir: str,
    latent_dir: str,
    ae_ckpt_path: str,
    batch_size: int = 2048,
    measurement_name: str = "RNA",
    device: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    从原始数据提取 latent 表示。

    Args:
        raw_data_dir: 原始 TileDB 数据目录
        latent_dir: 输出 latent 目录
        ae_ckpt_path: AE 模型 checkpoint 路径
        batch_size: 编码批次大小
        measurement_name: 测量名称
        device: 计算设备
        force: 是否强制重新提取

    Returns:
        是否成功
    """
    from src.models.ae_module import AELitModule

    log.info("=" * 60)
    log.info("🚀 自动提取 Latent 表示")
    log.info("=" * 60)
    log.info(f"  Raw 数据目录: {raw_data_dir}")
    log.info(f"  Latent 输出目录: {latent_dir}")
    log.info(f"  AE Checkpoint: {ae_ckpt_path}")

    # 1. 检查 AE checkpoint
    if not os.path.exists(ae_ckpt_path):
        log.error(f"❌ AE checkpoint 不存在: {ae_ckpt_path}")
        return False

    # 2. 检查原始数据目录
    if not os.path.exists(raw_data_dir):
        log.error(f"❌ Raw 数据目录不存在: {raw_data_dir}")
        return False

    # 3. 检查是否需要强制重新提取
    if os.path.exists(latent_dir) and not force:
        is_valid, msg = check_latent_exists(latent_dir, raw_data_dir)
        if is_valid:
            log.info(f"✅ {msg}，跳过提取")
            return True
        else:
            log.warning(f"⚠️ {msg}，将重新提取")
            shutil.rmtree(latent_dir)

    # 4. 创建输出目录
    os.makedirs(latent_dir, exist_ok=True)

    # 5. 加载 AE 模型（从 checkpoint 旁边的 hydra 配置读取 net 结构）
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    log.info(f"🔧 加载 AE 模型到 {device}...")

    # 尝试从 .hydra/config.yaml 读取模型配置
    ckpt_dir = Path(ae_ckpt_path).parent.parent  # checkpoints/ -> run_dir/
    hydra_config_path = ckpt_dir / ".hydra" / "config.yaml"

    if hydra_config_path.exists():
        import yaml
        from omegaconf import OmegaConf
        import hydra

        log.info(f"📄 从 Hydra 配置加载 net 结构: {hydra_config_path}")
        with open(hydra_config_path, 'r') as f:
            saved_cfg = OmegaConf.create(yaml.safe_load(f))

        # 实例化 net
        net = hydra.utils.instantiate(saved_cfg.model.net)

        # 加载 checkpoint，传入 net
        ae_model = AELitModule.load_from_checkpoint(
            ae_ckpt_path,
            map_location=device,
            net=net
        )
    else:
        # Fallback: 直接加载（可能需要 checkpoint 包含完整信息）
        log.warning(f"⚠️ 未找到 Hydra 配置，尝试直接加载 checkpoint")
        ae_model = AELitModule.load_from_checkpoint(ae_ckpt_path, map_location=device)

    ae_model.eval()
    ae_model.to(device)
    encoder = ae_model.net.encode

    # 6. 扫描所有 shards
    shard_dirs = sorted([
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ])

    log.info(f"📊 发现 {len(shard_dirs)} 个 shards")

    # 7. 并行处理 shards（流水线：预读取 -> GPU编码 -> 写入）
    success_count = 0
    latent_dim = None
    num_io_workers = min(4, len(shard_dirs))  # IO 并行度

    log.info(f"🚀 使用 {num_io_workers} 个 IO 线程并行处理")

    # 预读取队列
    prefetch_queue: List[Dict] = []
    max_prefetch = 4  # 最多预读取 4 个 shard

    def prefetch_shard(shard_name: str) -> Optional[Dict]:
        """预读取单个 shard 的数据（CPU 操作）"""
        input_uri = os.path.join(raw_data_dir, shard_name)
        output_uri = os.path.join(latent_dir, shard_name)

        try:
            ctx = tiledbsoma.SOMATileDBContext(tiledb_config={
                "py.init_buffer_bytes": 512 * 1024**2,
                "sm.memory_budget": 4 * 1024**3,
            })

            with tiledbsoma.Experiment.open(input_uri, context=ctx) as exp:
                obs = exp.obs.read().concat().to_pandas()
                n_cells = len(obs)
                n_vars = exp.ms[measurement_name].var.count

                if n_cells == 0:
                    return None

                x_uri = os.path.join(input_uri, "ms", measurement_name, "X", "data")

            with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                data = X.read().tables().concat()
                row_indices = data["soma_dim_0"].to_numpy()
                col_indices = data["soma_dim_1"].to_numpy()
                values = data["soma_data"].to_numpy()

            X_raw = np.zeros((n_cells, n_vars), dtype=np.float32)
            X_raw[row_indices, col_indices] = values

            return {
                "shard_name": shard_name,
                "input_uri": input_uri,
                "output_uri": output_uri,
                "X_raw": X_raw,
                "obs": obs,
            }
        except Exception as e:
            log.warning(f"⚠️ 预读取 {shard_name} 失败: {e}")
            return None

    def write_shard(data: Dict, X_latent: np.ndarray, latent_dim: int) -> bool:
        """写入单个 shard 的 latent 数据（CPU 操作）"""
        try:
            adata_latent = ad.AnnData(
                X=scipy.sparse.csr_matrix(X_latent.astype(np.float32)),
                obs=data["obs"]
            )
            adata_latent.var_names = [f"latent_{i}" for i in range(latent_dim)]

            if os.path.exists(data["output_uri"]):
                shutil.rmtree(data["output_uri"])

            tiledbsoma.io.from_anndata(
                experiment_uri=data["output_uri"],
                anndata=adata_latent,
                measurement_name=measurement_name
            )
            return True
        except Exception as e:
            log.warning(f"⚠️ 写入 {data['shard_name']} 失败: {e}")
            return False

    # 使用线程池进行 IO 并行
    with ThreadPoolExecutor(max_workers=num_io_workers) as io_executor:
        # 提交所有预读取任务
        prefetch_futures = {
            io_executor.submit(prefetch_shard, shard_name): shard_name
            for shard_name in shard_dirs
        }

        # 写入任务队列
        write_futures = []

        # 流水线处理
        pbar = tqdm(total=len(shard_dirs), desc="提取 Latent")

        for future in as_completed(prefetch_futures):
            shard_name = prefetch_futures[future]
            try:
                data = future.result()
                if data is None:
                    pbar.update(1)
                    continue

                # GPU 编码（串行，避免 GPU 竞争）
                X_raw = data["X_raw"]
                n_cells = X_raw.shape[0]
                n_batches = (n_cells + batch_size - 1) // batch_size

                latents = []
                for i in range(n_batches):
                    start = i * batch_size
                    end = min(start + batch_size, n_cells)
                    X_batch = torch.from_numpy(X_raw[start:end]).float().to(device)

                    with torch.no_grad():
                        z_batch = encoder(X_batch)
                    latents.append(z_batch.cpu().numpy())

                X_latent = np.concatenate(latents, axis=0)
                current_latent_dim = X_latent.shape[1]

                if latent_dim is None:
                    latent_dim = current_latent_dim

                # 释放原始数据内存
                del data["X_raw"]

                # 异步写入
                write_future = io_executor.submit(write_shard, data, X_latent, latent_dim)
                write_futures.append((write_future, shard_name))

                pbar.update(1)

            except Exception as e:
                log.warning(f"⚠️ 处理 {shard_name} 失败: {e}")
                pbar.update(1)

        pbar.close()

        # 等待所有写入完成
        log.info("⏳ 等待写入完成...")
        for write_future, shard_name in tqdm(write_futures, desc="写入 Shard"):
            try:
                if write_future.result():
                    success_count += 1
            except Exception as e:
                log.warning(f"⚠️ 写入 {shard_name} 失败: {e}")

    # 8. 总结
    log.info("=" * 60)
    log.info(f"✅ Latent 提取完成！")
    log.info(f"   成功: {success_count}/{len(shard_dirs)} shards")
    log.info(f"   Latent 维度: {latent_dim}")
    log.info(f"   输出目录: {latent_dir}")
    log.info("=" * 60)

    return success_count == len(shard_dirs)


def ensure_latent_data(
    mode: str,
    raw_data_dir: str,
    ae_ckpt_path: Optional[str] = None,
    latent_dir: Optional[str] = None,
    batch_size: int = 2048,
    device: Optional[str] = None,
) -> str:
    """
    确保 latent 数据可用，返回实际的 data_dir。

    这是供 train.py 调用的主入口函数。

    Args:
        mode: 训练模式 ("latent" 或 "raw")
        raw_data_dir: 原始 TileDB 数据目录
        ae_ckpt_path: AE checkpoint 路径（latent 模式必需）
        latent_dir: 指定的 latent 目录（可选，不指定则自动推理）
        batch_size: 提取时的批次大小
        device: 计算设备

    Returns:
        实际使用的 data_dir 路径

    Raises:
        ValueError: 配置错误时抛出
    """
    # Raw 模式：直接返回原始数据目录
    if mode == "raw":
        if not os.path.exists(raw_data_dir):
            raise ValueError(f"Raw 数据目录不存在: {raw_data_dir}")
        log.info(f"📁 [Raw Mode] 使用原始数据: {raw_data_dir}")
        return raw_data_dir

    # Latent 模式
    if mode == "latent":
        # 推理 latent 目录
        if latent_dir is None:
            latent_dir = get_latent_dir(raw_data_dir)

        log.info(f"📁 [Latent Mode] 检查 Latent 数据...")
        log.info(f"   Raw 目录: {raw_data_dir}")
        log.info(f"   Latent 目录: {latent_dir}")

        # 检查 latent 数据是否存在
        is_valid, msg = check_latent_exists(latent_dir, raw_data_dir)

        if is_valid:
            log.info(f"✅ {msg}")
            return latent_dir

        # Latent 不存在，需要提取
        log.warning(f"⚠️ {msg}")

        # 检查 AE checkpoint
        if ae_ckpt_path is None:
            raise ValueError(
                f"Latent 数据不存在且未指定 ae_ckpt_path！\n"
                f"请通过以下方式之一解决：\n"
                f"1. 手动运行: python src/utils/extract_latents.py --ckpt_path <AE_CKPT> --input_dir {raw_data_dir} --output_dir {latent_dir}\n"
                f"2. 配置 model.ae_ckpt_path 参数，系统将自动提取"
            )

        if not os.path.exists(ae_ckpt_path):
            raise ValueError(f"AE checkpoint 不存在: {ae_ckpt_path}")

        # 自动提取 latent
        log.info("🚀 开始自动提取 Latent...")
        success = extract_latents(
            raw_data_dir=raw_data_dir,
            latent_dir=latent_dir,
            ae_ckpt_path=ae_ckpt_path,
            batch_size=batch_size,
            device=device,
        )

        if not success:
            raise RuntimeError(f"Latent 提取失败！请检查日志或手动提取。")

        return latent_dir

    # 未知模式
    raise ValueError(f"未知的训练模式: {mode}，支持 'latent' 或 'raw'")
