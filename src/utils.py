import os
import json
from glob import glob
from typing import List, Optional, Tuple, Union

from PIL import Image
import torch
import numpy as np


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def get_wandb_logging_meta(wandb_logging_dir):
    """
    Get the meta data for wandb logging
    """   
    
    if not os.path.exists(wandb_logging_dir): return None, 0
    
    latest_run_dir = os.path.join(wandb_logging_dir, 'latest-run')
    if not os.path.exists(latest_run_dir):
        all_runs = sorted(glob(os.path.join(wandb_logging_dir, 'run-*')))
        if len(all_runs) < 1: return None, 0
        else: latest_run_dir = all_runs[-1]
    
    run_id = os.path.basename(glob(os.path.join(latest_run_dir, 'run-*.wandb'))[0]).split('.')[0].split('-')[-1]
    
    summary_fp = os.path.join(latest_run_dir, 'files', 'wandb-summary.json')
    with open(summary_fp, 'r') as f: summary_meta = json.load(f)
    run_step = summary_meta.get('_step', 0)
    
    print('Resumming wandb logging from %s. RUN_ID: %s RUN_STEP: %d.'%(latest_run_dir, run_id, run_step))
    
    return run_id, run_step


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def resize_image(image: Image.Image, new_size) -> Image.Image:
    # image = Image.open(file_path)
    width, height = image.size
    scale = min(new_size / width, new_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width == new_size and new_height == new_size:
        return image
    else:
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        bg_color = image.getpixel((0, 0))
        background = Image.new('RGB', (new_size, new_size), bg_color)
        offset = ((new_size - new_width) // 2, (new_size - new_height) // 2)
        background.paste(resized_image, offset)
        return background