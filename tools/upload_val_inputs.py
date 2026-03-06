#!/usr/bin/env python3
"""
Retroactively log VAE validation input images to an existing wandb run.

The original trainer only logged output (reconstruction) images.
This script loads the same validation dataset, extracts input images,
and logs them as Val-Geo-Input / Val-Mask-Input to the existing run.

Usage:
    python tools/upload_val_inputs.py \
        --run-id s5ei9dux \
        --data /home/rebuilderai/GarmageSet/garmages \
        --use_data_root \
        --list /home/rebuilderai/GarmageSet/datalist/garmageset_split_9_1_14537.pkl \
        --data_fields surf_ncs surf_mask
"""
import argparse
import os
import sys
import pickle

import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_validation_data(args):
    """Load validation dataset using same logic as VaeData."""
    print("Loading validation data...")
    with open(args.list, "rb") as f:
        data_list = pickle.load(f)['val']

    if args.use_data_root:
        data_list = [
            os.path.join(args.data, os.path.basename(x)) for x in data_list
            if os.path.exists(os.path.join(args.data, os.path.basename(x)))
        ]

    print(f"Total validation items: {len(data_list)}")

    # Load all validation data (same as VaeData.__next_chunk__)
    cache = []
    for path in tqdm(data_list, desc="Loading validation data"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            surf_ncs = []
            if 'surf_ncs' in args.data_fields:
                surf_ncs.append(data['surf_ncs'].astype(np.float32))
            if 'surf_wcs' in args.data_fields:
                surf_ncs.append(data['surf_wcs'].astype(np.float32))
            if 'surf_uv_ncs' in args.data_fields:
                surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
            if 'surf_normals' in args.data_fields:
                surf_ncs.append(data['surf_normals'].astype(np.float32))
            if 'surf_mask' in args.data_fields:
                surf_ncs.append(data['surf_mask'].astype(np.float32) * 2.0 - 1.0)

            surf_ncs = np.concatenate(surf_ncs, axis=-1)
            cache.append(surf_ncs)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    cache = np.vstack(cache)
    print(f"Validation data shape: {cache.shape}")
    return cache


def get_val_image_steps(run_id: str, entity: str, project: str) -> list:
    """Get step numbers where Val-Geo was logged by parsing image paths."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    rows = list(run.scan_history(keys=["Val-Geo"]))
    steps = []
    for row in rows:
        vg = row.get("Val-Geo", {})
        if isinstance(vg, dict) and "path" in vg:
            # path format: media/images/Val-Geo_6945_hash.png
            parts = os.path.basename(vg["path"]).split("_")
            # Find the step number (second-to-last part before hash)
            try:
                step = int(parts[1])  # Val-Geo_6945_hash.png => 6945
                steps.append(step)
            except (IndexError, ValueError):
                continue
    print(f"Found {len(steps)} validation image steps: {steps}")
    return steps


def make_input_images(val_cache, data_fields, sample_size=16):
    """Create input image grids matching trainer's make_grid format."""
    # Take first `sample_size` samples (fixed, not random)
    samples = val_cache[:sample_size]  # (16, H, W, C)
    samples = torch.FloatTensor(samples).permute(0, 3, 1, 2)  # (16, C, H, W)

    vis_log = {}

    # Geometry (NCS) - first 3 channels
    if 'surf_ncs' in data_fields:
        geo_grid = make_grid(samples[:, :3, ...], nrow=8, normalize=True, value_range=(-1, 1))
        vis_log['Val-Geo-Input'] = geo_grid[:3, ...]

    # Mask - last channel
    if 'surf_mask' in data_fields:
        mask_grid = make_grid(samples[:, -1:, ...], nrow=8, normalize=True, value_range=(-1, 1))
        vis_log['Val-Mask-Input'] = mask_grid[-1:, ...]

    # WCS - first 3 channels (if present)
    if 'surf_wcs' in data_fields:
        wcs_start = 3 if 'surf_ncs' in data_fields else 0
        wcs_grid = make_grid(samples[:, wcs_start:wcs_start+3, ...], nrow=8, normalize=False)
        wcs_grid[wcs_grid != 0.0] = (wcs_grid[wcs_grid != 0.0] + 1) / 2
        vis_log['Val-Geo-WCS-Input'] = wcs_grid[:3, ...]

    # UV - last 3 channels before mask
    if 'surf_uv_ncs' in data_fields:
        uv_end = -1 if 'surf_mask' in data_fields else samples.shape[1]
        uv_grid = make_grid(samples[:, uv_end-3:uv_end, ...], nrow=8, normalize=True, value_range=(-1, 1))
        vis_log['Val-UV-Input'] = uv_grid[-3:, ...]

    # Normals - channels 3:6
    if 'surf_normals' in data_fields:
        norm_grid = make_grid(samples[:, 3:6, ...], nrow=8, normalize=True, value_range=(-1, 1))
        vis_log['Val-Normal-Input'] = norm_grid[:3, ...]

    return vis_log


def main():
    parser = argparse.ArgumentParser(description="Upload validation input images to existing wandb run")
    parser.add_argument('--run-id', type=str, required=True, help='wandb run ID')
    parser.add_argument('--entity', type=str, default='jeremiah91-rebuilderai')
    parser.add_argument('--project', type=str, default='GarmentGen')
    parser.add_argument('--data', type=str, required=True, help='Path to data folder')
    parser.add_argument('--use_data_root', action='store_true')
    parser.add_argument('--list', type=str, required=True, help='Path to datalist pickle')
    parser.add_argument('--data_fields', nargs='+', default=['surf_ncs', 'surf_mask'])
    parser.add_argument('--dry-run', action='store_true', help='Show what would be logged without actually logging')
    args = parser.parse_args()

    # Step 1: Get steps where validation images were logged
    steps = get_val_image_steps(args.run_id, args.entity, args.project)
    if not steps:
        print("ERROR: No validation image steps found in the run.")
        sys.exit(1)

    # Step 2: Load validation data
    val_cache = load_validation_data(args)

    # Step 3: Create input image grids
    vis_log = make_input_images(val_cache, args.data_fields)
    print(f"Created input images: {list(vis_log.keys())}")

    if args.dry_run:
        print("\n[DRY RUN] Would log the following to each step:")
        for key, tensor in vis_log.items():
            print(f"  {key}: shape={tensor.shape}")
        print(f"  Steps: {steps}")
        return

    # Step 4: Resume wandb run and log input images
    # wandb requires monotonically increasing steps, so we can't log to past steps.
    # We log images WITHOUT custom step metrics so MediaBrowser in Reports can find them
    # on the default _step axis. Each val step gets a separate log call → separate _step.
    import wandb
    from torchvision.transforms.functional import to_pil_image

    print(f"\nResuming wandb run {args.run_id}...")
    wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume='allow'
    )

    # Log input images on the default _step axis (no define_metric).
    # MediaBrowser uses _step for its slider, so this ensures images are discoverable.
    for step in steps:
        log_dict = {}
        for key, tensor in vis_log.items():
            # wandb.Image expects (H, W, C) numpy or PIL Image, not (C, H, W) tensor
            pil_img = to_pil_image(tensor.clamp(0, 1))
            log_dict[key] = wandb.Image(
                pil_img,
                caption=f"{key.replace('-', ' ')} (original val step {step})"
            )
        wandb.log(log_dict)
        print(f"  Logged input images (original val step={step})")

    wandb.finish()
    print(f"\nDone! Logged input images to {len(steps)} steps in run {args.run_id}.")


if __name__ == "__main__":
    main()
