import os
import pickle
import shutil
import argparse
from tqdm import tqdm
from glob import glob

import torch
import numpy as np
from diffusers import DDPMScheduler
from torchvision.utils import make_grid
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from src.network import AutoencoderKLFastDecode, TextEncoder, PointcloudEncoder, SketchEncoder
from src.utils import randn_tensor
from src.constant import data_fields_dict
from src.bbox_utils import get_bbox
from src.vis import draw_bbox_geometry
from src.pc_utils import normalize_pointcloud


# Visualize pointcloud condition
def pointcloud_condition_visualize(vertices, output_fp=None):
    if hasattr(vertices, 'numpy'):
        vertices = vertices.numpy()
    vertices = np.asarray(vertices)
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices should be ndarray in (Nx3)"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False
            ),
            showlegend=False
        )
    ])

    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=3)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 800
    if output_fp:
        fig.write_image(output_fp.replace(".pkl", "_pcCondOrig.png"), width=RESO, height=RESO, scale=2.5)


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def init_models(args):
    device = args.device
    block_dims = args.block_dims
    sample_size = args.reso
    latent_channels = args.latent_channels
    latent_size = sample_size//(2**(len(block_dims)-1))

    surf_vae = AutoencoderKLFastDecode(
        in_channels=args.img_channels,
        out_channels=args.img_channels,
        down_block_types=['DownEncoderBlock2D']*len(block_dims),
        up_block_types=['UpDecoderBlock2D']*len(block_dims),
        block_out_channels=block_dims,
        layers_per_block=2,
        act_fn='silu',
        latent_channels=latent_channels,
        norm_num_groups=8,
        sample_size=sample_size
    )
    surf_vae.load_state_dict(torch.load(args.vae, map_location=device), strict=False)
    surf_vae.to(device).eval()

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    # Load conditioning model
    if args.text_encoder is not None: text_enc = TextEncoder(args.text_encoder, device=device)
    else: text_enc = None
    if args.pointcloud_encoder is not None: pointcloud_enc = PointcloudEncoder(args.pointcloud_encoder, device=device)
    else: pointcloud_enc = None
    if args.sketch_encoder is not None: sketch_enc = SketchEncoder(args.sketch_encoder, device="cpu")
    else: sketch_enc = None

    # condition dimention
    if args.text_encoder is not None:
        condition_dim = text_enc.text_emb_dim
    elif args.pointcloud_encoder is not None:
        condition_dim = pointcloud_enc.pointcloud_emb_dim
    elif args.sketch_encoder is not None:
        condition_dim = sketch_enc.sketch_emb_dim
    else:
        condition_dim = -1

    # Initialize network
    model_p_dim = 8
    model_z_dim = 0
    for k in args.latent_data_fields:
        model_z_dim += data_fields_dict[k]["len"]

    # Load Generation Model ===
    if args.denoiser_type == 'default':
        from src.network import GarmageNet
        print("Default Transformer-Encoder denoiser.")
        model = GarmageNet(
            p_dim=model_p_dim,
            z_dim=model_z_dim,
            num_heads=12,
            embed_dim=args.embed_dim,
            condition_dim=condition_dim,
            num_layer=args.num_layer,
            num_cf=-1
        )
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(args.garmagenet)['model_state_dict'])
    model.to(device).eval()
    print('[DONE] Models initialized.')

    return {
        'surf_vae': surf_vae,
        'ddpm_scheduler': ddpm_scheduler,
        'model': model,
        'text_enc': text_enc,
        'pointcloud_enc': pointcloud_enc,
        'sketch_enc': sketch_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size
    }


def inference_one(
        models,
        args,
        caption='',
        pointcloud_feature=None,
        sampled_pc_cond=None,
        sketch_features=None,
        output_fp='',
        dedup=True,
        vis=False,
        data_fp=None,
        data_id_trainval=None,
):
    max_surf = 32
    device = args.device

    ddpm_scheduler = models['ddpm_scheduler']
    model = models['model']
    surf_vae = models['surf_vae']
    text_enc = models['text_enc']

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    # get condition embedding
    if args.text_encoder is not None:
        condition_emb = text_enc(caption)
    elif args.pointcloud_encoder is not None:
        condition_emb = pointcloud_feature[None,...]
    elif args.sketch_encoder is not None:
        condition_emb = sketch_features[None,...]
    else:
        condition_emb = None
    if condition_emb is not None:
        condition_emb = condition_emb.to(device)

    surf_bbox, surf_uv_bbox = None, None

    # GeometryLatent+BBox Denoising ---------------------------------------------------------------
    latent_len = model.z_dim
    latent = randn_tensor((1, max_surf, 64), device=device)
    pos = randn_tensor((1, max_surf, 8), device=device)
    mask = torch.zeros((1, max_surf), dtype=torch.bool, device=device)
    ddpm_scheduler.set_timesteps(1000//10)
    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Denoising"):
            timesteps = t.reshape(-1).to(device)
            pred = model(
                pos=pos,
                z=latent,
                timesteps=timesteps,
                mask=mask,
                class_label=None,
                cond_global=condition_emb,
                cond_local = None,
                is_train=False
            )
            latent = ddpm_scheduler.step(pred[...,:64], t, latent).prev_sample
            pos = ddpm_scheduler.step(pred[...,64:], t, pos).prev_sample

    # Filter out invalid token by the variance
    panel_valid_mask = latent.var(-1)>1e-3

    # check filter ambiguity (different between bbox filter and latent filter)
    if True:
        filt_diff = torch.sum((latent.var(-1)>1e-3) != (pos.var(-1)>1e-3))
        if filt_diff>0:
            print(f"\n\n=== Filter ambiguity size {filt_diff} happen between latent and bbox. ===\n\n")

    n_surfs = panel_valid_mask.sum(-1)
    latent = latent[panel_valid_mask]
    pos = pos[panel_valid_mask]

    geo_latent = latent
    surf_bbox_ccwh = pos[...,:6].detach().cpu().numpy()
    # ccwh2xyxy
    surf_bbox = np.concatenate(
        [surf_bbox_ccwh[...,:3]-surf_bbox_ccwh[...,3:]/2,
         surf_bbox_ccwh[...,:3]+surf_bbox_ccwh[...,3:]/2],axis=-1)
    surf_uv_bbox_scale = pos[...,6:].detach().cpu().numpy()

    # VAE Decoding ------------------------------------------------------------------------
    with torch.no_grad(): decoded_garmage = surf_vae(geo_latent.view(-1, latent_channels, latent_size, latent_size))

    # save vis garmage channel-wise
    if vis:
        pred_img = make_grid(decoded_garmage, nrow=6, normalize=True, value_range=(-1,1))
        fig, ax = plt.subplots(len(args.garmage_data_fields), 1, figsize=(40, 40))

        current_channel = 0
        for d_idx, d_type in enumerate(args.garmage_data_fields):
            c_st = current_channel
            c_ed = current_channel + data_fields_dict[d_type]["len"]
            current_channel = c_ed
            cur_img = pred_img[c_st:c_ed, ...]
            if data_fields_dict[d_type]["len"] == 2:
                pad_shape = torch.tensor(cur_img.shape)
                pad_shape[0] = 1
                pad_shape = tuple(pad_shape)
                pad = torch.full(pad_shape, 0.5, dtype=cur_img.dtype, device=cur_img.device)
                cur_img = torch.cat([pad, cur_img], dim=0)  # shape: (4, 4)
            ax[d_idx].imshow(cur_img.permute(1, 2, 0).detach().cpu().numpy())
            ax[d_idx].set_title(data_fields_dict[d_type]["title"])

        plt.tight_layout()
        plt.axis('off')

        if output_fp: plt.savefig(output_fp.replace('.pkl', '_geo_img.png'), transparent=True, dpi=72)
        else: plt.show()
        plt.close()

    # pharse Garmage by garmage_data_fields
    decoded_garmage = decoded_garmage.permute(0, 2, 3, 1).detach().cpu().numpy()
    surf_ncs, surf_wcs, surf_uv_ncs, surf_normals, surf_ncs_mask = None, None, None, None, None
    current_channel = 0
    for d_idx, d_type in enumerate(args.garmage_data_fields):
        c_st = current_channel
        c_ed = current_channel + data_fields_dict[d_type]["len"]
        current_channel = c_ed

        if d_type == 'surf_ncs':
            surf_ncs = decoded_garmage[..., c_st:c_ed].reshape(n_surfs, -1, 3)
        elif d_type == 'surf_wcs':
            surf_wcs = decoded_garmage[..., c_st:c_ed].reshape(n_surfs, -1, 3)
        elif d_type == 'surf_uv_ncs':
            surf_uv_ncs = decoded_garmage[..., c_st:c_ed].reshape(n_surfs, -1, 2)
        elif d_type == 'surf_normals':
            surf_normals = decoded_garmage[..., c_st:c_ed].reshape(n_surfs, -1, 3)
        elif d_type == 'surf_mask':
            surf_ncs_mask = decoded_garmage[..., c_st:c_ed].reshape(n_surfs, -1) > 0.0
        else:
            raise NotImplementedError

    # get 3d bbox
    if surf_wcs is not None and surf_bbox is None:
        surf_bbox = [np.concatenate(get_bbox(surf_wcs[i][surf_ncs_mask[i]])) for i in range(n_surfs)]
        surf_bbox = np.stack(surf_bbox)
    elif surf_wcs is None and surf_bbox is not None:
        surf_wcs = _denormalize_pts(surf_ncs, surf_bbox)

    # Get 2d bbox
    """
    Current version for opensource only generation 2d-bbox scale.
    It is also valid to train model generate 2d bbox directly.
    """
    if surf_uv_bbox is None and surf_uv_bbox_scale is not None:
        surf_uv_bbox = np.zeros((n_surfs, 4))
        surf_uv_bbox[:, :2] = surf_uv_bbox[:,:2] - surf_uv_bbox_scale/2
        surf_uv_bbox[:, 2:] = surf_uv_bbox[:,2:] + surf_uv_bbox_scale/2

    # plotly visualization
    if vis:
        colormap = plt.cm.coolwarm
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]

        draw_bbox_geometry(
            bboxes = surf_bbox,
            bbox_colors = colors,
            points = surf_wcs,
            point_masks = surf_ncs_mask,
            point_colors = colors,
            num_point_samples = 5000,
            title = caption,
            output_fp = output_fp.replace('.pkl', '_pointcloud.png'),
            # show_num=True
            )

    result = {
        'surf_bbox': surf_bbox,        # (N, 6)
        'surf_uv_bbox': surf_uv_bbox,  # (N, 4)
        'surf_ncs': surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption,             # str
        'data_fp': data_fp,
        'data_id': data_id_trainval,
        # 'denoising': denoising_dict,
        'args': vars(args)
    }

    for k in result:
        if isinstance(result[k], np.ndarray):
            result[k] = result[k].tolist()

    # visualize pointcloud condition
    if args.pointcloud_encoder is not None:
        sampled_pts_normalized = normalize_pointcloud(sampled_pc_cond, 1)
        pointcloud_condition_visualize(sampled_pts_normalized, output_fp)
        result["sampled_pc_cond"] = sampled_pc_cond

    if output_fp:
        with open(output_fp, 'wb') as f: pickle.dump(result, f)

    torch.cuda.empty_cache()
    print('[DONE] save to:', output_fp)


def check_cond(args):
    # Ensure only 1 kind of condition embedding applied.
    not_none_count = sum(x is not None for x in [args.text_encoder, args.pointcloud_encoder, args.sketch_encoder])
    assert not_none_count in (0, 1)


def run_cache(args):
    models = init_models(args)

    os.makedirs(args.output, exist_ok=True)

    with open(args.cache, 'rb') as f: data_cache = pickle.load(f)

    n_samples = len(data_cache['item_idx'])
    if args.max_samples is not None:
        n_samples = min(n_samples, args.max_samples)
    for sample_data_idx in tqdm(range(n_samples)):
        if "caption" in data_cache:
            caption = data_cache['caption'][sample_data_idx]
        else:
            print("No caption in cache.")
            caption = None
        if args.pointcloud_encoder is not None:
            if "pccond_item_idx" in data_cache:
                choice = 0  # 0:surface_uniform, 1:fps, 2:non_uniformX
                pccond_idx = data_cache["pccond_item_idx"][sample_data_idx]
                pointcloud_features = data_cache["pointcloud_feature"][pccond_idx[0]:pccond_idx[1]]
                pointcloud_features = pointcloud_features[choice]
                sampled_pc_cond = data_cache["sampled_pc_cond"][pccond_idx[0]:pccond_idx[1]]
                sampled_pc_cond = sampled_pc_cond[choice]
            else:
                pointcloud_features = data_cache["pointcloud_feature"][sample_data_idx]
                sampled_pc_cond = data_cache["sampled_pc_cond"][sample_data_idx]
        else:
            pointcloud_features = None
            sampled_pc_cond = None

        if args.sketch_encoder is not None:
            sketch_features = data_cache["sketch_feature"][sample_data_idx]
        else:
            sketch_features = None

        output_fp = os.path.join(args.output, f'{sample_data_idx:04d}.pkl')

        data_fp = data_cache.get('data_fp', None)
        if data_fp is not None:
            data_fp = data_fp[sample_data_idx]
        data_id_trainval = data_cache.get('data_id', None)
        if data_id_trainval is not None:
            data_id_trainval = data_id_trainval[sample_data_idx]

        inference_one(
            models,
            args = args,
            caption = caption,
            pointcloud_feature = pointcloud_features,
            sampled_pc_cond = sampled_pc_cond,
            sketch_features = sketch_features,
            dedup = True,
            output_fp = output_fp,
            vis = True,
            data_fp = data_fp,
            data_id_trainval = data_id_trainval,
        )

    print('[DONE]')


def run_image(args):
    models = init_models(args)

    sketch_enc = models["sketch_enc"]

    os.makedirs(args.output, exist_ok=True)

    reference_sketch_dir = args.inference_data
    reference_sketch_fp_list = sorted(glob(os.path.join(reference_sketch_dir, "*.png")))
    for idx, sketch_fp in tqdm(enumerate(reference_sketch_fp_list)):
        try:
            sketch_features = sketch_enc.sketch_embedder_fn(sketch_fp)
            sketch_features = torch.tensor(sketch_features[args.condition_type], device=args.device).squeeze(0)
            shutil.copy(sketch_fp, args.output)
            output_fp = os.path.join(args.output, os.path.basename(sketch_fp).replace(".png", ".pkl"))

            inference_one(
                models,
                args=args,
                caption="",
                pointcloud_feature=None,
                sampled_pc_cond=None,
                sketch_features=sketch_features,
                dedup=True,
                output_fp=output_fp,
                vis=True,
                data_fp=sketch_fp,
                data_id_trainval="",
            )
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, choices=["cache", "image", "caption", "pointcloud"], default="cache")

    # path configuration
    parser.add_argument(
        '--vae', type=str,
        default=None,
        help='Path to VAE model')
    parser.add_argument(
        '--garmagenet', type=str,
        default=None,
        help='Path to garmagenet model')
    parser.add_argument(
        "--denoiser_type",
        type=str, choices=['default'],
        default='default', help="Choose ldm type.")
    parser.add_argument(
        '--cache', type=str,
        default=None,
        help='Path to cache file')
    parser.add_argument(
        '--inference_data', type=str, default=None,
        help='Sketch encoder type.')

    parser.add_argument(
        '--output', type=str, default='generated',
        help='Path to output directory')
    
    parser.add_argument('--garmage_data_fields', nargs='+', type=str, default=[])
    parser.add_argument('--latent_data_fields', nargs='+', type=str, default=[])
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Block dimensions of the VAE model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--img_channels', type=int, default=4)
    parser.add_argument('--reso', type=int, default=256)
    parser.add_argument("--padding", type=str, default="zero", choices=["repeat", "zero"])
    parser.add_argument('--embed_dim', type=int, default=768, help='Embding dim of ldm model.')
    parser.add_argument('--num_layer', type=int, nargs='+', default=12, help='Layer num of ldm model.')  # TE:int HYdit:list
    parser.add_argument('--text_encoder', type=str, default=None, choices=[None, 'CLIP'], help='Text encoder type.')
    parser.add_argument('--pointcloud_encoder', type=str, default=None, choices=[None, 'POINT_E'], help='Pointcloud encoder type.')
    parser.add_argument('--sketch_encoder', type=str, default=None, choices=[None, 'LAION2B', "RADIO_V2.5-G", "RADIO_V2.5-H", "RADIO_V2.5-H_spatial"], help='Sketch encoder type.')
    parser.add_argument("--condition_type", type=str, default='summary', choices=['summary', 'spatial'],
                        help="Text encoder type when applying text as generation condition.")

    parser.add_argument('--device', type=str, default="cuda", help='')
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples to run inference on (default: all)')

    args = parser.parse_args()

    if args.task == 'cache' and not os.path.exists(args.cache):
        raise FileNotFoundError(f"File {args.cache} not found.")
    if args.task == 'cache' and args.inference_data is not None:
        raise ValueError('inference_data cannot be used with cache.')
    if args.task != 'cache' and not os.path.exists(args.inference_data):
        raise FileNotFoundError(f"File {args.inference_data} not found.")

    check_cond(args)

    task_choice = {
        "cache": run_cache,
        "image": run_image,
    }
    task_choice[args.task](args)