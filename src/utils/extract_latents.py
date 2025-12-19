#!/usr/bin/env python
"""
提取 Latent 表示脚本

功能：
1. 加载训练好的 AE 模型
2. 读取 TileDB-SOMA 数据
3. 通过 AE 编码到 Latent Space
4. 将 Latent 向量存储到新的 TileDB-SOMA（保留 obs 元数据）

使用方法：
python scripts/extract_latents.py \
    --ckpt_path logs/ae_stage1/checkpoints/best.ckpt \
    --input_dir /fast/data/scTFM/rtf/TEDD/tile_4000_fix \
    --output_dir /fast/data/scTFM/rtf/TEDD/latents \
    --batch_size 2048
"""

import os
import argparse
import shutil
from tqdm import tqdm
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
import numpy as np
import tiledbsoma
import tiledbsoma.io
import anndata as ad
import pandas as pd
import scipy.sparse

from src.models.ae_module import AELitModule


def extract_latents_from_shard(
    shard_uri: str,
    output_uri: str,
    encoder,
    device: torch.device,
    batch_size: int = 2048,
    measurement_name: str = "RNA"
):
    """
    从单个 shard 提取 latent 表示。
    
    Args:
        shard_uri: 输入 shard 路径
        output_uri: 输出 shard 路径
        encoder: AE 编码器
        device: 计算设备
        batch_size: 批次大小
        measurement_name: 测量名称
    """
    print(f"📦 Processing: {os.path.basename(shard_uri)}")
    
    try:
        # 1. 读取原始数据
        ctx = tiledbsoma.SOMATileDBContext()
        with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
            # 读取 obs（包含所有元数据：next_cell_id, prev_cell_id, time 等）
            obs = exp.obs.read().concat().to_pandas()
            
            # 读取 X 数据（基因表达）
            X_raw = exp.ms[measurement_name].X[measurement_name].read().tables().concat()
            X_raw = X_raw.to_pandas().to_numpy()
            
            if X_raw.shape[0] == 0:
                print(f"⚠️  Skipped: {shard_uri} (empty)")
                return False
        
        # 2. 批量编码到 Latent Space
        n_cells = X_raw.shape[0]
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        latents = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_cells)
            
            # 转换为 Tensor
            X_batch = torch.from_numpy(X_raw[start:end]).float().to(device)
            
            # 编码
            with torch.no_grad():
                z_batch = encoder(X_batch)
            
            latents.append(z_batch.cpu().numpy())
        
        # 合并所有 batch
        X_latent = np.concatenate(latents, axis=0)
        
        # 3. 创建新的 AnnData（latent + 元数据）
        adata_latent = ad.AnnData(
            X=scipy.sparse.csr_matrix(X_latent.astype(np.float32)),
            obs=obs  # 保留所有元数据！
        )
        
        # 添加 var_names（latent 维度）
        latent_dim = X_latent.shape[1]
        adata_latent.var_names = [f"latent_{i}" for i in range(latent_dim)]
        
        # 4. 写入新的 TileDB-SOMA
        if os.path.exists(output_uri):
            shutil.rmtree(output_uri)
        
        tiledbsoma.io.from_anndata(
            experiment_uri=output_uri,
            anndata=adata_latent,
            measurement_name=measurement_name
        )
        
        print(f"✅ Saved: {os.path.basename(output_uri)} ({n_cells} cells, latent_dim={latent_dim})")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {shard_uri}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="提取 Latent 表示")
    parser.add_argument("--ckpt_path", type=str, required=True, help="AE 模型权重路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入 TileDB 根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 Latent TileDB 根目录")
    parser.add_argument("--batch_size", type=int, default=2048, help="编码批次大小")
    parser.add_argument("--measurement_name", type=str, default="RNA", help="Measurement 名称")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda:0, cpu, auto)")
    
    args = parser.parse_args()
    
    # 1. 检查输入路径
    if not os.path.exists(args.input_dir):
        raise ValueError(f"❌ 输入路径不存在: {args.input_dir}")
    
    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📂 输出目录: {args.output_dir}")
    
    # 3. 加载 AE 模型
    print(f"🔧 加载 AE 模型: {args.ckpt_path}")
    
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    ae_model = AELitModule.load_from_checkpoint(args.ckpt_path, map_location=device)
    ae_model.eval()
    ae_model.to(device)
    
    # 获取编码器
    encoder = ae_model.net.encode
    
    print(f"✅ AE 模型已加载到 {device}")
    
    # 4. 扫描所有 shards
    shard_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    print(f"📊 发现 {len(shard_dirs)} 个 shards")
    
    # 5. 逐个处理 shards
    success_count = 0
    
    for shard_name in tqdm(shard_dirs, desc="提取 Latent"):
        input_shard_uri = os.path.join(args.input_dir, shard_name)
        output_shard_uri = os.path.join(args.output_dir, shard_name)
        
        success = extract_latents_from_shard(
            shard_uri=input_shard_uri,
            output_uri=output_shard_uri,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            measurement_name=args.measurement_name
        )
        
        if success:
            success_count += 1
    
    # 6. 总结
    print("\n" + "=" * 50)
    print(f"✅ 完成！成功处理 {success_count}/{len(shard_dirs)} 个 shards")
    print(f"📂 Latent 数据保存到: {args.output_dir}")
    print("=" * 50)
    
    # 7. 验证输出
    print("\n验证输出...")
    output_shards = sorted([
        d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d))
    ])
    
    if len(output_shards) > 0:
        # 读取第一个 shard 验证结构
        test_uri = os.path.join(args.output_dir, output_shards[0])
        ctx = tiledbsoma.SOMATileDBContext()
        
        with tiledbsoma.Experiment.open(test_uri, context=ctx) as exp:
            obs = exp.obs.read().concat().to_pandas()
            latent_dim = exp.ms[args.measurement_name].var.count
            
            print(f"\n📋 输出数据结构:")
            print(f"  - Latent Dim: {latent_dim}")
            print(f"  - Obs 列: {list(obs.columns)}")
            print(f"  - 包含 next_cell_id: {'next_cell_id' in obs.columns}")
            print(f"  - 包含 prev_cell_id: {'prev_cell_id' in obs.columns}")
            print(f"  - 包含 time: {'time' in obs.columns}")
    
    print("\n🎉 Latent 提取完成！现在可以训练 RTF 了：")
    print(f"python src/train.py -cn train_rtf \\")
    print(f"  model.mode=latent \\")
    print(f"  model.latent_dim={latent_dim} \\")
    print(f"  data.data_dir={args.output_dir}")


if __name__ == "__main__":
    main()

