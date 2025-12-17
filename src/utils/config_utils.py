"""
配置工具函数，用于根据 mode 自动调整参数。
"""
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

def adjust_config_by_mode(cfg: DictConfig) -> DictConfig:
    """
    根据 model.mode 自动调整配置参数。
    
    支持两种模式：
    1. latent: Latent Space Training
       - model.net.input_dim = latent_dim (例如 512)
       - data.latent_key = "X_latent"
       - data.batch_size = 512 (可以更大)
    
    2. raw: Raw Gene Space Training
       - model.net.input_dim = n_genes (例如 28231)
       - data.latent_key = null
       - data.batch_size = 128 (显存限制)
    
    Args:
        cfg: Hydra 配置对象
    
    Returns:
        调整后的配置对象
    """
    # 解除 struct 模式以允许修改
    OmegaConf.set_struct(cfg, False)
    
    # 获取 mode（如果不存在，默认为 latent）
    mode = cfg.model.get("mode", "latent")
    
    if mode == "latent":
        # Latent Mode 配置
        latent_dim = cfg.model.get("latent_dim", 512)  # 默认 512
        
        # 自动设置 backbone input_dim
        if "net" in cfg.model and "input_dim" in cfg.model.net:
            cfg.model.net.input_dim = latent_dim
        
        # 自动设置 data.latent_key
        if "data" in cfg:
            cfg.data.latent_key = "X_latent"
            
            # 如果没有显式设置 batch_size，使用较大的默认值
            if cfg.data.get("batch_size") is None:
                cfg.data.batch_size = 512
        
        print(f"✅ [Config] Latent Mode 已启用: input_dim={latent_dim}, latent_key='X_latent', batch_size={cfg.data.get('batch_size', 512)}")
    
    elif mode == "raw":
        # Raw Mode 配置
        n_genes = cfg.model.get("n_genes", 28231)  # 默认 28231
        
        # 自动设置 backbone input_dim
        if "net" in cfg.model and "input_dim" in cfg.model.net:
            cfg.model.net.input_dim = n_genes
        
        # 自动设置 data.latent_key
        if "data" in cfg:
            cfg.data.latent_key = None
            
            # 如果没有显式设置 batch_size，使用较小的默认值（节省显存）
            if cfg.data.get("batch_size") is None:
                cfg.data.batch_size = 128
        
        print(f"✅ [Config] Raw Mode 已启用: input_dim={n_genes}, latent_key=None, batch_size={cfg.data.get('batch_size', 128)}")
    
    else:
        raise ValueError(f"❌ 不支持的 mode: {mode}，必须是 'latent' 或 'raw'")
    
    # 恢复 struct 模式
    OmegaConf.set_struct(cfg, True)
    
    return cfg


def get_mode_specific_params(mode: str) -> Dict[str, Any]:
    """
    根据 mode 返回推荐参数。
    
    Args:
        mode: "latent" 或 "raw"
    
    Returns:
        参数字典
    """
    if mode == "latent":
        return {
            "input_dim": 512,
            "latent_key": "X_latent",
            "batch_size": 512,
            "description": "Latent Space Training (推荐): 在 AE 潜在空间训练，速度快，显存占用低"
        }
    elif mode == "raw":
        return {
            "input_dim": 28231,
            "latent_key": None,
            "batch_size": 128,
            "description": "Raw Gene Space Training: 在原始基因空间训练，显存占用高，速度慢"
        }
    else:
        raise ValueError(f"❌ 不支持的 mode: {mode}")

