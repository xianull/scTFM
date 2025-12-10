import argparse
import os
import sys
import torch
import scanpy as sc
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
import logging

# 确保项目根目录在路径中，以便能够导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.ae_module import AELitModule

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(config_path, checkpoint_path, device):
    """
    加载模型配置和权重。
    """
    logger.info(f"正在加载配置文件: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    logger.info("正在实例化网络架构...")
    # 实例化网络 (例如 VanillaAE)
    # 我们使用配置文件中的 model.net 部分
    if 'model' in cfg and 'net' in cfg.model:
        net_cfg = cfg.model.net
    elif 'net' in cfg:
        net_cfg = cfg.net
    else:
        # 尝试直接在 cfg 中找，或者假设 cfg 就是 net 的配置
        # 针对 hydra 保存的 config.yaml，结构通常是 {model: {net: ...}, ...}
        raise ValueError("无法在配置中找到 model.net 或 net 部分")

    net = instantiate(net_cfg)
    
    logger.info(f"正在加载检查点: {checkpoint_path}")
    # 加载 LightningModule
    # 我们传递 optimizer=None 和 scheduler=None，因为推理时不需要它们
    # 但它们是 AELitModule __init__ 方法的必需参数
    model = AELitModule.load_from_checkpoint(
        checkpoint_path,
        net=net,
        optimizer=None,
        scheduler=None,
        map_location=device
    )
    
    model.eval()
    model.to(device)
    return model

def process_h5ad(h5ad_path, model, device, output_path, batch_size=1024, layer=None):
    """
    读取 h5ad，执行推理，并保存结果。
    """
    logger.info(f"正在读取数据: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    # 检查输入维度
    # 尝试从网络的第一层获取输入维度 (假设是 Linear 层)
    input_dim = None
    if hasattr(model.net, 'encoder') and isinstance(model.net.encoder[0], torch.nn.Linear):
        input_dim = model.net.encoder[0].in_features
    elif hasattr(model.net, 'input_dim'):
         input_dim = model.net.input_dim
            
    if input_dim and adata.shape[1] != input_dim:
        logger.warning(f"模型期望输入维度为 {input_dim}，但 adata 维度为 {adata.shape[1]}。"
                       "请确保数据预处理正确 (例如基因数量一致)。")

    # 准备数据加载器
    # 假设数据已经经过了模型期望的预处理 (例如 log1p)
    # AE 通常期望 log1p 归一化后的数据
    
    if layer:
        logger.info(f"使用图层: {layer}")
        X = adata.layers[layer]
    else:
        logger.info("使用 .X 作为输入")
        X = adata.X
        
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # 创建简单的 Tensor Dataset
    dataset = torch.tensor(X, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    logger.info("开始推理...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # AELitModule forward 返回 net(x)
            # VanillaAE forward 返回 (recon_x, z)
            # 我们只需要 z (Latent 向量)
            _, z = model(batch)
            embeddings.append(z.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    logger.info(f"Embeddings 形状: {embeddings.shape}")
    
    # 存储到 adata.obsm
    adata.obsm['X_ae_emb'] = embeddings
    
    # 保存结果
    if output_path:
        logger.info(f"正在保存结果到: {output_path}")
        adata.write_h5ad(output_path)
    else:
        logger.info("未指定输出路径，跳过保存。")

    return adata

def main():
    parser = argparse.ArgumentParser(description="Autoencoder 模型推理脚本")
    parser.add_argument("--h5ad", type=str, required=True, help="输入 .h5ad 文件的路径")
    parser.add_argument("--config", type=str, required=True, help=".hydra/config.yaml 文件的路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点 (.ckpt) 的路径")
    parser.add_argument("--output", type=str, required=True, help="保存带有 embedding 的 .h5ad 文件的路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备 (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1024, help="推理批次大小")
    parser.add_argument("--layer", type=str, default=None, help="使用的 AnnData layer，默认为 .X")

    args = parser.parse_args()
    
    if not os.path.exists(args.h5ad):
        logger.error(f"输入文件未找到: {args.h5ad}")
        sys.exit(1)
        
    if not os.path.exists(args.config):
        logger.error(f"配置文件未找到: {args.config}")
        sys.exit(1)
        
    if not os.path.exists(args.ckpt):
        logger.error(f"检查点文件未找到: {args.ckpt}")
        sys.exit(1)

    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    model = load_model(args.config, args.ckpt, device)
    
    process_h5ad(args.h5ad, model, device, args.output, args.batch_size, args.layer)
    
    logger.info("完成。")

if __name__ == "__main__":
    main()
