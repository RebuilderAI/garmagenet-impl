"""
Prepare POINT_E features for conditional LDM training.

Step 1: Samples point clouds from OBJ meshes (or uses pre-sampled ones)
Step 2: Extracts POINT_E features from sampled point clouds

Output directory structure (matching garmage.py expectations):
    {output_dir}/{uuid}/feature_POINT_E/{uuid}_surface_uniform.npy  # shape: (1, 512)
    {output_dir}/{uuid}/sampled_pc/{uuid}_surface_uniform.npy       # shape: (2048, 3)
"""

import os
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm

import numpy as np
import torch


def extract_features(
    pc_sampled_dir: str,
    output_dir: str,
    cache_dir: str,
    batch_size: int = 64,
):
    """Extract POINT_E features from pre-sampled point clouds."""

    # Import POINT_E
    from src.models.pc_backbone.point_e.evals.feature_extractor import PointNetClassifier

    # Create cache dir for POINT_E model weights
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize POINT_E encoder
    print(f"Loading POINT_E model (cache: {cache_dir})...")
    encoder = PointNetClassifier(
        devices=["cuda"],
        cache_dir=cache_dir,
        device_batch_size=batch_size,
    )
    print("POINT_E model loaded.")

    # Find all sampled point clouds
    pc_dirs = sorted(glob(os.path.join(pc_sampled_dir, "*")))
    pc_dirs = [d for d in pc_dirs if os.path.isdir(d)]
    print(f"Found {len(pc_dirs)} point cloud directories.")

    # Track stats
    total = len(pc_dirs)
    skipped = 0
    errors = 0

    for pc_dir in tqdm(pc_dirs, desc="Extracting POINT_E features"):
        uuid = os.path.basename(pc_dir)

        # Check if already processed
        feature_out_path = os.path.join(output_dir, uuid, "feature_POINT_E", f"{uuid}_surface_uniform.npy")
        pc_out_path = os.path.join(output_dir, uuid, "sampled_pc", f"{uuid}_surface_uniform.npy")

        if os.path.exists(feature_out_path) and os.path.exists(pc_out_path):
            skipped += 1
            continue

        # Load sampled point cloud
        npy_files = sorted(glob(os.path.join(pc_dir, "*.npy")))
        if len(npy_files) == 0:
            print(f"WARNING: No .npy files found in {pc_dir}")
            errors += 1
            continue

        try:
            pc = np.load(npy_files[0])  # shape: (2048, 3)

            # Extract feature
            feature = encoder.get_features(pc)  # shape: (1, 512)

            # Save feature
            os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
            np.save(feature_out_path, feature)

            # Save sampled pc (copy to expected location)
            os.makedirs(os.path.dirname(pc_out_path), exist_ok=True)
            np.save(pc_out_path, pc)

        except Exception as e:
            print(f"ERROR processing {uuid}: {e}")
            errors += 1
            continue

    print(f"\n=== Feature Extraction Complete ===")
    print(f"Total: {total}")
    print(f"Processed: {total - skipped - errors}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")

    # Verify a random sample
    verify_sample = glob(os.path.join(output_dir, "*/feature_POINT_E/*.npy"))
    if verify_sample:
        sample = np.load(verify_sample[0])
        print(f"\nVerification - sample feature shape: {sample.shape}")
        print(f"  mean: {sample.mean():.4f}, std: {sample.std():.4f}")
        print(f"  min: {sample.min():.4f}, max: {sample.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract POINT_E features from sampled point clouds")
    parser.add_argument("--pc_sampled_dir", type=str, required=True,
                        help="Directory containing sampled point clouds (from prepare_pc_cond_sample.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for features (pointcloud_feature_dir for training)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="POINT_E model cache directory (default: ~/point_e_model_cache)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for POINT_E inference")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = os.path.join(os.path.expanduser("~"), "point_e_model_cache")

    extract_features(
        pc_sampled_dir=args.pc_sampled_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
