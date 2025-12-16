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
    # [关键] 使用文件缓存避免 DDP 多进程重复计算
    # 主进程（spawn 前）计算并保存，子进程（spawn 后）直接读取
    # ---------------------------------------------------------------------------
    
    if cfg.get("train"):
        import json
        import tempfile
        from pathlib import Path
        
        # 获取当前进程的 Rank
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        rank = int(os.environ.get("RANK", -1))
        
        # 判断是否是主进程（spawn 前：两个都是 -1；spawn 后：会有具体值）
        is_pre_ddp = (local_rank == -1 and rank == -1)
        
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
            
            # 使用数据目录的 hash 作为缓存文件名，避免不同数据集冲突
            cache_key = f"{data_dir}_{batch_size}_{world_size}".replace("/", "_")
            cache_file = Path(tempfile.gettempdir()) / f"scTFM_stats_{cache_key}.json"
            
            # 获取 DataLoader workers 数（用于负载均衡）
            num_workers_per_gpu = cfg.data.get("num_workers", 16)
            
            if is_pre_ddp:
                # 主进程：计算并缓存
                log.info(f"📊 [Main Process] Calculating dataset stats (World Size={world_size})...")
                
                from src.utils.dataset_stats_utils import balanced_shard_assignment
                
                total_cells, total_steps, shard_sizes = get_dataset_stats(
                    root_dir=data_dir,
                    split_label=0, 
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
                    total_cells, total_steps, shard_sizes = get_dataset_stats(
                        root_dir=data_dir, split_label=0, 
                        batch_size=batch_size, num_workers=4, world_size=world_size,
                        num_workers_per_gpu=num_workers_per_gpu
                    )
            
            # 设置配置
            if total_steps > 0:
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
            
            cache_key = f"{data_dir}_{batch_size}_{world_size}".replace("/", "_")
            cache_file = Path(tempfile.gettempdir()) / f"scTFM_stats_{cache_key}.json"
            
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
