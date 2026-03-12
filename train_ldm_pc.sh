#!/bin/bash

# 가상환경 활성화 및 PYTHONPATH 설정
# source .venv/garmagenet/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Starting GarmageNet LDM training with Pointcloud condition..."

python src/ldm.py \
    --data data/garmages --use_data_root \
    --list data/garmageset_split_9_1_14537.pkl \
    --option garmagenet --lr 5e-4 \
    --surfvae checkpoints/vae_e0090.pt \
    --cache_dir cache/encoder_mode \
    --expr GarmageNet_conditional_pc \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 1000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --dropout 0.1 \
    --pointcloud_encoder POINT_E --pointcloud_feature_dir data/pc_features \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature \
    --gpu 0
