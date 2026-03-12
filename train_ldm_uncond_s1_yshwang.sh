#!/bin/bash

# 가상환경 활성화
source /workspace/garmagenet-impl/.venv/garmagenet/bin/activate

# PYTHONPATH 설정
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "Starting STAGE 1: Unconditional GarmageNet LDM training (yshwang branch)..."
echo "Config: Batch 4096, LR 5e-4 (FIXED), Original Parameters (BBox weight 7.0, 6-AdaLN)"

# Unconditional 모드 실행 (pointcloud 관련 인자 제거)
python src/ldm.py \
    --data data/garmages --use_data_root \
    --list /workspace/garmagenet-impl/data/garmageset_split_9_1_14537.pkl \
    --option garmagenet --lr 5e-4 \
    --surfvae /workspace/garmagenet-impl/checkpoints/vae_e0090.pt \
    --cache_dir /workspace/garmagenet-impl/cache/encoder_mode \
    --expr GarmageNet_Uncond_S1_B4096_Original_Restored \
    --train_nepoch 20000 --test_nepoch 500 --save_nepoch 1000 --batch_size 4096 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --dropout 0.1 \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs \
    --gpu 0
