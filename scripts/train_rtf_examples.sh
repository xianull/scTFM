#!/bin/bash
# RTF (Rectified Flow) 训练示例脚本

# ===================================================================
# 示例 1: Latent Mode + Rectified Flow (默认，推荐)
# ===================================================================
echo "示例 1: Latent Mode + Rectified Flow"
python src/train.py -cn train_rtf \
  model.mode=latent \
  model.latent_dim=512 \
  model.flow_type=rectified_flow \
  data.data_dir=/fast/data/scTFM/rtf/TEDD/latents \
  data.batch_size=512 \
  trainer.devices=[0,1,2,3]

# ===================================================================
# 示例 2: Raw Mode + Flow Matching
# ===================================================================
echo "示例 2: Raw Mode + Flow Matching"
python src/train.py -cn train_rtf \
  model.mode=raw \
  model.n_genes=28231 \
  model.flow_type=flow_matching \
  data.data_dir=/fast/data/scTFM/rtf/TEDD/tile_4000_fix \
  data.batch_size=128 \
  trainer.devices=[0,1]

# ===================================================================
# 示例 3: Latent Mode + Velocity Flow + 双向数据增强
# ===================================================================
echo "示例 3: Latent Mode + Velocity Flow + 双向数据增强"
python src/train.py -cn train_rtf \
  model.mode=latent \
  model.flow_type=velocity_flow \
  data.direction=both \
  data.batch_size=256 \
  trainer.devices=[0,1,2,3,4,5,6,7]

# ===================================================================
# 示例 4: 8卡 DDP + 高性能配置
# ===================================================================
echo "示例 4: 8卡 DDP + 高性能配置"
python src/train.py -cn train_rtf \
  model.mode=latent \
  model.flow_type=rectified_flow \
  data.batch_size=512 \
  data.num_workers=12 \
  data.prefetch_factor=4 \
  trainer.devices=[0,1,2,3,4,5,6,7] \
  trainer.max_epochs=100 \
  trainer.precision=16-mixed

# ===================================================================
# 示例 5: 小规模测试 (快速验证)
# ===================================================================
echo "示例 5: 小规模测试"
python src/train.py -cn train_rtf \
  model.mode=latent \
  model.net.hidden_size=256 \
  model.net.depth=4 \
  data.batch_size=128 \
  trainer.devices=[0] \
  trainer.max_epochs=5 \
  trainer.limit_train_batches=100

# ===================================================================
# 示例 6: Hyperparameter Sweep (Hydra Multirun)
# ===================================================================
echo "示例 6: Hyperparameter Sweep"
python src/train.py -cn train_rtf -m \
  model.mode=latent \
  model.flow_type=rectified_flow,flow_matching,velocity_flow \
  model.optimizer.lr=1e-4,5e-4,1e-3 \
  data.batch_size=256,512

