import pyrootutils
import torch.multiprocessing as mp
import torch
import os
import json
import tempfile
from pathlib import Path

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import List, Optional

from src.utils.pylogger import get_pylogger
from src.utils.dataset_stats_utils import get_dataset_stats

log = get_pylogger(__name__)


def infer_latent_dim_from_ae(cfg: DictConfig) -> None:
    """
    从 AE checkpoint 推断 latent_dim 并设置 input_dim。

    此函数在所有进程中执行（包括 DDP 子进程），
    以确保模型参数形状一致。
    """
    if not cfg.get("train"):
        return

    mode = cfg.model.get("mode")
    if mode != "latent":
        return

    ae_ckpt_path = cfg.model.get("ae_ckpt_path")
    if not ae_ckpt_path:
        return

    try:
        from pathlib import Path
        import yaml

        ckpt_dir = Path(ae_ckpt_path).parent.parent
        hydra_config_path = ckpt_dir / ".hydra" / "config.yaml"

        if hydra_config_path.exists():
            with open(hydra_config_path, 'r') as f:
                ae_cfg = yaml.safe_load(f)

            latent_dim = ae_cfg.get('model', {}).get('net', {}).get('latent_dim')
            if latent_dim:
                OmegaConf.set_struct(cfg, False)
                cfg.model.net.input_dim = latent_dim
                OmegaConf.set_struct(cfg, True)
                log.info(f"🔧 自动设置 input_dim={latent_dim} (从 AE checkpoint 读取)")
    except Exception as e:
        log.warning(f"⚠️ 无法从 AE checkpoint 推断 latent_dim: {e}")


def ensure_latent_data_if_needed(cfg: DictConfig, do_extract: bool = True) -> None:
    """
    根据 model.mode 自动处理 latent 数据（仅适用于 RTF 训练）。

    此函数只在以下条件同时满足时执行：
    1. cfg.train = True（训练模式）
    2. cfg.model.mode 存在（RTF 模型特有配置，AE 模型没有此字段）

    逻辑：
    - mode=raw: 直接使用 raw_data_dir
    - mode=latent: 检查 latent 数据是否存在，不存在则自动提取

    Args:
        cfg: Hydra 配置对象
        do_extract: 是否执行实际提取（DDP 子进程设为 False，只做路径推理）
    """
    # 只在训练时处理
    if not cfg.get("train"):
        return

    # 检查是否是 RTF 训练（AE 模型没有 mode 配置，会直接返回）
    mode = cfg.model.get("mode")
    if mode is None:
        # AE 训练或其他没有 mode 的模型，跳过
        return

    # 获取 raw_data_dir
    raw_data_dir = cfg.data.get("raw_data_dir")
    if raw_data_dir is None:
        # 向后兼容：如果没有 raw_data_dir，使用 data_dir
        raw_data_dir = cfg.data.get("data_dir")
        if raw_data_dir is None:
            return

    from src.utils.latent_manager import ensure_latent_data, get_latent_dir

    # 获取配置
    ae_ckpt_path = cfg.model.get("ae_ckpt_path")
    latent_dir = cfg.data.get("latent_dir")

    try:
        if do_extract:
            # 主进程：完整流程（检查 + 必要时提取）
            actual_data_dir = ensure_latent_data(
                mode=mode,
                raw_data_dir=raw_data_dir,
                ae_ckpt_path=ae_ckpt_path,
                latent_dir=latent_dir,
                batch_size=2048,  # 提取时使用较大 batch
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            # DDP 子进程：仅推理路径，不执行提取
            if mode == "raw":
                actual_data_dir = raw_data_dir
            elif mode == "latent":
                actual_data_dir = latent_dir if latent_dir else get_latent_dir(raw_data_dir)
            else:
                actual_data_dir = raw_data_dir

        # 更新配置中的 data_dir
        OmegaConf.set_struct(cfg, False)
        cfg.data.data_dir = actual_data_dir
        OmegaConf.set_struct(cfg, True)

        log.info(f"📁 数据目录已设置: {actual_data_dir}")

    except Exception as e:
        log.error(f"❌ Latent 数据处理失败: {e}")
        raise


def log_hyperparameters_to_wandb(
    cfg: DictConfig,
    loggers: List[Logger],
) -> None:
    """将Hydra配置记录到WandB。"""
    wandb_logger: Optional[WandbLogger] = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    if wandb_logger is None:
        log.warning("No WandbLogger found, skipping hyperparameter logging.")
        return

    experiment = wandb_logger.experiment
    if not hasattr(experiment, 'config') or not hasattr(experiment.config, 'update'):
        return

    hparams = {}
    config_keys = ["model", "data", "trainer", "callbacks", "task_name", "seed", "train", "test"]
    for key in config_keys:
        if key in cfg:
            value = cfg[key]
            if OmegaConf.is_config(value):
                hparams[key] = OmegaConf.to_container(value, resolve=True)
            else:
                hparams[key] = value

    experiment.config.update(hparams, allow_val_change=True)
    log.info("Hyperparameters logged to WandB successfully.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # ---------------------------------------------------------------------------
    # [关键] 自动推断 latent_dim（所有进程都执行，确保模型参数一致）
    # ---------------------------------------------------------------------------
    infer_latent_dim_from_ae(cfg)

    # ---------------------------------------------------------------------------
    # [新增] 自动处理 Latent 数据
    # - 主进程（DDP spawn 前）：检查并提取 latent 数据
    # - DDP 子进程：仅更新 data_dir，不执行实际提取
    # ---------------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    is_pre_ddp = (local_rank == -1 and rank == -1)

    if cfg.get("train"):
        # 所有进程都需要更新 data_dir（DDP 子进程只做路径推理，不执行提取）
        ensure_latent_data_if_needed(cfg, do_extract=is_pre_ddp)

    # ---------------------------------------------------------------------------
    # [关键] 使用文件缓存避免 DDP 多进程重复计算
    # 主进程（spawn 前）计算并保存，子进程（spawn 后）直接读取
    # ---------------------------------------------------------------------------

    if cfg.get("train"):
        try:
            data_dir = cfg.data.get("data_dir")
            batch_size = cfg.data.get("batch_size", 256)

            # 计算 World Size
            devices = cfg.trainer.get("devices")
            if devices == "auto":
                world_size = torch.cuda.device_count()
            elif isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices):
                world_size = len(devices)
            elif isinstance(devices, int):
                world_size = devices
            elif isinstance(devices, str) and devices.isdigit():
                world_size = int(devices)
            else:
                if isinstance(devices, str) and "," in devices:
                    world_size = len(devices.split(","))
                else:
                    world_size = 1

            # 使用 task_name + 数据目录 hash 作为缓存目录，避免不同任务/数据集冲突
            task_name = cfg.get("task_name", "default")
            import hashlib
            data_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
            cache_dir = Path(tempfile.gettempdir()) / "scTFM_cache" / task_name / data_hash
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"stats_bs{batch_size}_ws{world_size}.json"

            # 获取 DataLoader workers 数（用于负载均衡）
            num_workers_per_gpu = cfg.data.get("num_workers", 16)
            
            if is_pre_ddp:
                # 主进程：计算并缓存
                log.info(f"📊 [Main Process] Calculating dataset stats (World Size={world_size})...")
                
                from src.utils.dataset_stats_utils import balanced_shard_assignment
                
                # 一致性训练任务不使用 split_label 筛选
                split_label = None if task_name == "consistency_flow" else 0
                
                total_cells, total_steps, shard_sizes = get_dataset_stats(
                    root_dir=data_dir,
                    split_label=split_label, 
                    batch_size=batch_size,
                    num_workers=16,  # 只有一个进程计算，可以开大
                    world_size=world_size,
                    num_workers_per_gpu=num_workers_per_gpu
                )
                
                # 计算负载均衡的分配方案
                total_workers = world_size * num_workers_per_gpu
                assignment = balanced_shard_assignment(shard_sizes, total_workers)
                
                # 保存到缓存文件
                cache_file.write_text(json.dumps({
                    "total_cells": total_cells,
                    "total_steps": total_steps,
                    "world_size": world_size,
                    "batch_size": batch_size,
                    "num_workers_per_gpu": num_workers_per_gpu,
                    "shard_sizes": shard_sizes,
                    "assignment": {str(k): v for k, v in assignment.items()}  # JSON keys must be strings
                }))
                
                log.info(f"✅ [Main] Cached stats + assignment: {total_steps} steps → {cache_file}")
                
            else:
                # DDP 子进程：读取缓存
                if cache_file.exists():
                    stats = json.loads(cache_file.read_text())
                    total_steps = stats["total_steps"]
                    log.info(f"📥 [Rank {local_rank}] Loaded from cache: {total_steps} steps")
                else:
                    log.warning(f"⚠️ [Rank {local_rank}] Cache not found, recalculating...")
                    from src.utils.dataset_stats_utils import balanced_shard_assignment
                    split_label = None if task_name == "consistency_flow" else 0
                    total_cells, total_steps, shard_sizes = get_dataset_stats(
                        root_dir=data_dir, split_label=split_label, 
                        batch_size=batch_size, num_workers=4, world_size=world_size,
                        num_workers_per_gpu=num_workers_per_gpu
                    )
            
            # 设置配置
            # 一致性训练的 batch 数取决于链数量而非细胞数，不使用 limit_train_batches
            if total_steps > 0 and task_name != "consistency_flow":
                OmegaConf.set_struct(cfg, False)
                cfg.trainer.limit_train_batches = total_steps
                OmegaConf.set_struct(cfg, True)
                
        except Exception as e:
            log.warning(f"❌ Failed to handle dataset stats: {e}")

    # ---------------------------------------------------------------------------
    # [Hydra/Lightning 初始化]
    # ---------------------------------------------------------------------------

    # 1. Seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 2. DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")

    # 如果有缓存的负载均衡方案，注入到 DataModule
    if cfg.get("train"):
        try:
            data_dir = cfg.data.get("data_dir")
            batch_size = cfg.data.get("batch_size", 256)
            devices = cfg.trainer.get("devices")

            if devices == "auto":
                world_size = torch.cuda.device_count()
            elif isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices):
                world_size = len(devices)
            elif isinstance(devices, int):
                world_size = devices
            elif isinstance(devices, str) and devices.isdigit():
                world_size = int(devices)
            else:
                if isinstance(devices, str) and "," in devices:
                    world_size = len(devices.split(","))
                else:
                    world_size = 1

            # 使用与上面一致的 cache 路径
            task_name = cfg.get("task_name", "default")
            import hashlib
            data_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
            cache_dir = Path(tempfile.gettempdir()) / "scTFM_cache" / task_name / data_hash
            cache_file = cache_dir / f"stats_bs{batch_size}_ws{world_size}.json"

            if cache_file.exists():
                stats = json.loads(cache_file.read_text())
                if "assignment" in stats:
                    # 注入负载均衡方案到配置
                    # [CRITICAL] 保持字符串 key，因为 Dataset 查找时用 str(global_worker_id)
                    OmegaConf.set_struct(cfg, False)
                    cfg.data.shard_assignment = stats["assignment"]  # 保持原始字符串 key
                    OmegaConf.set_struct(cfg, True)
                    log.info(f"📥 Loaded shard assignment from cache ({len(stats['assignment'])} workers)")
        except Exception as e:
            log.warning(f"Failed to load shard assignment: {e}")
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 4. Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 5. Callbacks
    callbacks: List[Callback] = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 6. Logger
    logger: List[Logger] = []
    if cfg.get("logger"):
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    if logger:
        log_hyperparameters_to_wandb(cfg, logger)

    # 7. Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 8. Train
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # 9. Test
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    # [关键] 必须设置为 spawn，否则 ProcessPoolExecutor 和 Lightning DDP 都会出问题
    mp.set_start_method('spawn', force=True)
    main()
