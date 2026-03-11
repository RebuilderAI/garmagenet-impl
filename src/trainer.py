import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn

import wandb
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import AutoencoderKL, DDPMScheduler

from src.network import TextEncoder, SketchEncoder, PointcloudEncoder, GarmageNet, AutoencoderKLFastEncode
from src.utils import get_wandb_logging_meta

def steps_per_epoch(L, B):
    return math.ceil(L / B)

class VAETrainer():
    def __init__(self, args, train_dataset, val_dataset):
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.data_fields = args.data_fields

        assert train_dataset.num_channels == val_dataset.num_channels, \
            'Expecting same dimensions for train and val dataset, got %d (train) and %d (val).'%(train_dataset.num_channels, val_dataset.num_channels)

        num_channels = train_dataset.num_channels
        sample_size = train_dataset.resolution
        latent_channels = args.latent_channels

        self.vae_type = args.vae_type
        if self.vae_type == "kl":
            model = AutoencoderKL(
                in_channels=num_channels,
                out_channels=num_channels,
                down_block_types=['DownEncoderBlock2D']*len(args.block_dims),
                up_block_types= ['UpDecoderBlock2D']*len(args.block_dims),
                block_out_channels=args.block_dims,
                layers_per_block=2,
                act_fn='silu',
                latent_channels=latent_channels,
                norm_num_groups=8,
                sample_size=sample_size,
            )
            # latentcode in different length should adjust KLloss param.
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'], strict=False)
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'], strict=False)
            else: model.load_state_dict(state_dict, strict=False)
            print('Load SurfZNet checkpoint from %s.'%(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        if self.vae_type == "kl":
            self.network_params = list(self.model.parameters())
            self.optimizer = torch.optim.AdamW(
                self.network_params,
                lr=args.lr,
                weight_decay=1e-5
            )
        else:
            raise NotImplementedError

        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2.0**16,
            growth_factor=1.5,
            growth_interval=100 * steps_per_epoch(len(self.train_dataset), args.batch_size)
        )

        # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initilizer dataloader
        self.num_workers = 16
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            # print("This may cause error if batch size has changed.")

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        epoch_data_time = 0.0
        epoch_compute_time = 0.0
        data_start = time.time()

        # Train
        for surf_data in self.train_dataloader:
            epoch_data_time += time.time() - data_start
            compute_start = time.time()

            with torch.cuda.amp.autocast():
                surf_data = surf_data.to(self.device, non_blocking=True).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE
                if self.vae_type == "kl":
                    posterior = self.model.encode(surf_data).latent_dist
                    z = posterior.sample()      # = posterior.mean + torch.randn_like(posterior.std)*posterior.std
                    dec = self.model.decode(z).sample

                    # Loss functions
                    kl_loss = posterior.kl().mean()
                    mse_loss = self.loss_fn(dec, surf_data)
                    total_loss = mse_loss + 1e-6 * kl_loss
                else:
                    raise NotImplementedError

                # Update model
                self.scaler.scale(total_loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            epoch_compute_time += time.time() - compute_start

            # logging
            if self.iters % 10 == 0:
                if self.vae_type == "kl":
                    _z = posterior.mode()
                    wandb.log({
                        "epoch": self.epoch,
                        "loss-mse": mse_loss,
                        "loss-kl": kl_loss,
                        "z-min": z.min(),
                        "z-max": z.max(),
                        "z-mean": z.mean(),
                        "z-std": z.std(),
                        "mode-min": _z.min(),
                        "mode-max": _z.max(),
                        "mode-mean": _z.mean(),
                        "mode-std": _z.std(),
                        "gpu-mem-MB": torch.cuda.max_memory_allocated() / 1024**2,
                    }, step=self.iters)
                else:
                    raise NotImplementedError

            self.iters += 1
            progress_bar.update(1)
            data_start = time.time()

        progress_bar.close()

        # Log epoch-level timing for bottleneck diagnosis
        wandb.log({
            "timing/data_load_sec": epoch_data_time,
            "timing/compute_sec": epoch_compute_time,
            "timing/data_pct": epoch_data_time / max(epoch_data_time + epoch_compute_time, 1e-6) * 100,
        }, step=self.iters)
        print(f"  Epoch {self.epoch} timing: data={epoch_data_time:.1f}s, compute={epoch_compute_time:.1f}s, "
              f"data%={epoch_data_time / max(epoch_data_time + epoch_compute_time, 1e-6) * 100:.1f}%")

        # update train dataset
        self.train_dataset.update()
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2)

        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        print('Running validation...')
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0

        if self.vae_type == "kl":
            eval_loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

        val_images = None

        with torch.no_grad():
            for surf_data in self.val_dataloader:
                surf_data = surf_data.to(self.device).permute(0,3,1,2) # (N, H, W, C) => (N, C, H, W)

                # Pass through VAE
                if self.vae_type == "kl":
                    posterior = self.model.encode(surf_data).latent_dist
                    z = posterior.sample()      # = posterior.mean + torch.randn_like(posterior.std)*posterior.std
                    dec = self.model.decode(z).sample
                else:
                    raise NotImplementedError

                loss = eval_loss(dec, surf_data).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_data)

                if val_images is None and dec.shape[0] > 16:
                    sample_idx = torch.randperm(dec.shape[0])[:16]
                    val_images = make_grid(dec[sample_idx, ...], nrow=8, normalize=True, value_range=(-1,1))
                    val_inputs = make_grid(surf_data[sample_idx, ...], nrow=8, normalize=True, value_range=(-1,1))

                    vis_log = {}
                    if 'surf_ncs' in self.data_fields:
                        vis_log['Val-Geo-Input'] = wandb.Image(val_inputs[:3, ...], caption="Geometry input.")
                        vis_log['Val-Geo'] = wandb.Image(val_images[:3, ...], caption="Geometry output.")
                    if 'surf_wcs' in self.data_fields:
                        val_inputs2 = make_grid(surf_data[sample_idx, :3], nrow=8, normalize=False)
                        val_inputs2[val_inputs2 != 0.0] = (val_inputs2[val_inputs2 != 0.0] + 1) / 2
                        vis_log['Val-Geo-WCS-Input'] = wandb.Image(val_inputs2[:3, ...], caption="Geometry WCS input.")
                        val_images2 = make_grid(dec[sample_idx, :3], nrow=8, normalize=False)
                        val_images2[val_images2 != 0.0] = (val_images2[val_images2 != 0.0] + 1) / 2
                        vis_log['Val-Geo-WCS'] = wandb.Image(val_images2[:3, ...], caption="Geometry WCS output.")
                    if 'surf_uv_ncs' in self.data_fields:
                        vis_log['Val-UV-Input'] = wandb.Image(val_inputs[-3:, ...], caption="UV input.")
                        vis_log['Val-UV'] = wandb.Image(val_images[-3:, ...], caption="UV output.")
                    if 'surf_normals' in self.data_fields:
                        vis_log['Val-Normal-Input'] = wandb.Image(val_inputs[3:6, ...], caption="Normal input.")
                        vis_log['Val-Normal'] = wandb.Image(val_images[3:6, ...], caption="Normal output.")
                    if 'surf_mask' in self.data_fields:
                        vis_log['Val-Mask-Input'] = wandb.Image(val_inputs[-1:, ...], caption="Mask input.")
                        vis_log['Val-Mask'] = wandb.Image(val_images[-1:, ...], caption="Mask output.")

                    wandb.log(vis_log, step=self.iters)

                    # Save validation images to disk
                    vis_dir = os.path.join(self.log_dir, 'val_images')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_image(val_inputs[:3, ...], os.path.join(vis_dir, f'input_e{self.epoch:04d}.png'))
                    save_image(val_images[:3, ...], os.path.join(vis_dir, f'recon_e{self.epoch:04d}.png'))
                    if 'surf_mask' in self.data_fields:
                        save_image(val_inputs[-1:, ...], os.path.join(vis_dir, f'input_mask_e{self.epoch:04d}.png'))
                        save_image(val_images[-1:, ...], os.path.join(vis_dir, f'recon_mask_e{self.epoch:04d}.png'))
                    print(f'  Saved val images to {vis_dir}/ (epoch {self.epoch})')

        if self.vae_type == "kl":
            mse = total_loss / total_count
            self.model.train()  # set to train
            wandb.log({"Val-mse": mse}, step=self.iters)
        else:
            raise NotImplementedError

        self.val_dataset.update()
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2)

    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(ckpt_log_dir, f'vae_e{self.epoch:04d}.pt'))
        return


from src.constant import get_condition_dim


class GarmageNetTrainer():
    def __init__(self, args, train_dataset, val_dataset):
        self.args = args

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.gpu is not None:
            self.device_ids = args.gpu
        else:
            self.device_ids = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # Initialize condition encoder
        self.cond_encoder = None
        if args.text_encoder is not None:
            self.text_encoder = TextEncoder(args.text_encoder, self.device)
            self.cond_encoder = self.text_encoder
        if args.pointcloud_encoder is not None:
            self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
            self.cond_encoder = self.pointcloud_encoder
        if args.sketch_encoder is not None:
            self.sketch_encoder = SketchEncoder(args.sketch_encoder, self.device)
            self.cond_encoder = self.sketch_encoder

        self.condition_dim = get_condition_dim(args, self)

        # Load pretrained surface vae (fast encode version)
        surf_vae_encoder = AutoencoderKLFastEncode(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            down_block_types=['DownEncoderBlock2D']*len(self.block_dims),
            up_block_types= ['UpDecoderBlock2D']*len(self.block_dims),
            block_out_channels=self.block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=self.latent_channels,
            norm_num_groups=8,
            sample_size=self.sample_size,
        )

        surf_vae_encoder.load_state_dict(torch.load(args.surfvae, map_location=self.device), strict=False)
        surf_vae_encoder = nn.DataParallel(surf_vae_encoder, device_ids=self.device_ids) # distributed inference
        self.surf_vae_encoder = surf_vae_encoder.to(self.device).eval()

        self.train_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, self.z_scaled)
        self.val_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, train_dataset.z_scaled)
        if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # Initialize network
        p_dim = 8     # 3D-bbox(6) + 2D-scale(2)
        z_dim = (self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels  
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = GarmageNet(
                p_dim=p_dim,
                z_dim=z_dim,
                embed_dim=args.embed_dim,
                condition_dim=self.condition_dim,
                num_layer=args.num_layer,
                num_cf=train_dataset.num_classes
                )
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'z_scaled' in state_dict:
                self.z_scaled = train_dataset.z_scaled = val_dataset.z_scaled = state_dict['z_scaled']
            if 'bbox_scaled' in state_dict:
                self.bbox_scaled = train_dataset.bbox_scaled = val_dataset.bbox_scaled = state_dict['bbox_scaled']
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'], strict=False)
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'], strict=False)
            else: model.load_state_dict(state_dict, strict=False)
            print('Load checkpoint from %s.'%(args.weight))

        model = nn.DataParallel(model, device_ids=self.device_ids) # distributed training
        self.model = model.to(self.device).train()

        self.device = self.model.module.parameters().__next__().device

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler ===
        if args.scheduler == "DDPM":
            self.scheduler_type = 'DDPM'
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule='linear',
                prediction_type='epsilon',
                beta_start=0.0001,
                beta_end=0.02,
                clip_sample=False,
            )
        else:
            raise NotImplementedError

        # Initialize optimizer
        self.network_params =  list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=args.lr,
            betas=(0.9, 0.999),  # More standard momentum
            weight_decay=1e-6,
            eps=1e-6,            # Increased for AMP stability
        )
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2.0 ** 16,
            growth_factor=1.5,
            growth_interval=2000  # Fixed interval (was 5000 * steps_per_epoch ≈ 55000, too large)
        )
        if args.finetune:
            try:
                if "optimizer" in state_dict:
                    self.optimizer.load_state_dict(state_dict["optimizer"])
                if "scaler" in state_dict:
                    self.scaler.load_state_dict(state_dict["scaler"])
            except (ValueError, RuntimeError) as e:
                print(f"[WARN] Skipping optimizer/scaler load (architecture changed): {e}")

        # LR Scheduler: warmup (1000 epochs) + cosine decay
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = 1000
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(args.train_nepoch - warmup_epochs, 1), eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        print(f"[LR Schedule] Warmup {warmup_epochs} epochs → Cosine decay to 1e-6 over {args.train_nepoch} epochs")

        # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # # Initilizer dataloader
        num_worker = 4
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=num_worker
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=num_worker
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            # print("This may cause error if batch size has changed.")

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                pos, z, mask, cls, caption, pointcloud_feature, sketch_feature  = data
                pos, z, mask, cls = \
                    pos.to(self.device), z.to(self.device), \
                    mask.to(self.device), cls.to(self.device)
                
                z = z * self.z_scaled
                pos = torch.concatenate([
                    (pos[..., 3:6] + pos[..., 0:3]) * 0.5,  # 3D bbox center
                    (pos[..., 3:6] - pos[..., 0:3]),        # 3D bbox size
                    (pos[..., 8:10] - pos[..., 6:8]),       # 2D bbox size
                ], dim=-1)
                
                mask[...] = False

                # NOTE: pointcloud and sketch features are pre-computed and loaded in dataset.
                cond_global, cond_local = None, None
                if hasattr(self, 'text_encoder'):
                    cond_global = self.text_encoder(caption).to(self.device)
                elif hasattr(self, 'pointcloud_encoder'):
                    cond_local = pointcloud_feature.to(self.device)
                elif hasattr(self, 'sketch_encoder'):
                    cond = sketch_feature.to(self.device)
                    if cond.ndim == 3: cond_local = cond
                    elif cond.ndim == 2: cond_global = cond
                    else: raise NotImplementedError
                
                bsz = len(pos)

                self.optimizer.zero_grad() # zero gradient

                # forward ===
                if self.scheduler_type == "DDPM":
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                    
                    z_noise = torch.randn(z.shape).to(self.device)
                    z_noised = self.noise_scheduler.add_noise(z, z_noise, timesteps)

                    pos_noise = torch.randn(pos.shape).to(self.device)
                    pos_noised = self.noise_scheduler.add_noise(pos, pos_noise, timesteps)

                    # Predict noise
                    pred_noise = self.model(
                        pos = pos_noised,
                        z = z_noised,
                        timesteps = timesteps,
                        mask = mask,
                        class_label = cls,
                        cond_global = cond_global,
                        cond_local = cond_local,
                        is_train = True
                        )

                    # Loss
                    loss_z_noise = self.loss_fn(
                        pred_noise[~mask][:, :z.shape[-1]], 
                        z_noise[~mask][:, :z.shape[-1]]
                        )
                    loss_pos_noise = self.loss_fn(
                        pred_noise[~mask][:, z.shape[-1]:], 
                        pos_noise[~mask][:,]
                    )

                    # Loss
                    total_loss = loss_z_noise * 1.0 + loss_pos_noise * 1.0  # Reduced weight for stability (was 7.0)
                
                else:
                    raise NotImplementedError


                # === NaN detection (non-intrusive safety check) ===
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f'\n[FATAL] NaN/Inf at epoch {self.epoch}, iter {self.iters}')
                    ckpt_path = os.path.join(self.log_dir, 'ckpts', 'emergency_pre_nan.pt')
                    torch.save({
                        'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                        'z_scaled': self.z_scaled, 'bbox_scaled': self.bbox_scaled,
                        'optimizer': self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(),
                        'epoch': self.epoch, 'iters': self.iters,
                    }, ckpt_path)
                    raise RuntimeError(f'NaN loss at epoch {self.epoch}')

                # Update model ===
                # NOTE: torch.autograd.set_detect_anomaly(True) is useful for debugging NaNs but 
                # significantly slows down training (approx. 2-3x slower). Only enable when debugging.
                # with torch.autograd.set_detect_anomaly(True):
                #     self.scaler.scale(total_loss).backward()
                self.scaler.scale(total_loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # restore to 50.0 (was 1.0, too tight)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # logging
                if self.iters % 10 == 0:
                    log_dict = {
                        "epoch": self.epoch,
                        "total_loss": total_loss,
                        'loss_latent': loss_z_noise.item(),
                        'loss_bbox': loss_pos_noise.item(),
                    }
                    if hasattr(self, 'scheduler'):
                        log_dict['lr'] = self.scheduler.get_last_lr()[0]
                    wandb.log(log_dict, step=self.iters)

                self.iters += 1
                progress_bar.update(1)

        progress_bar.close()
        # update train dataset
        if hasattr(self.train_dataset, 'data_chunks') and len(self.train_dataset.data_chunks) > 1:
            print('Updating train data chunks...')
            self.train_dataset.update()
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                batch_size=self.batch_size, num_workers=16)

        self.epoch += 1

        # Step LR scheduler (per epoch)
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        if self.epoch % 1000 == 0:
            torch.cuda.empty_cache()
        return

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        
        mse_loss = nn.MSELoss(reduction='none')
        l1_loss = nn.L1Loss(reduction='none')

        # Calculating metrics on denoising steps ===
        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        val_timesteps = [10,50,100,200,500,1000]
        latent_loss = [0]*len(val_timesteps)
        bbox_loss = [0]*len(val_timesteps)
        bbox_L1 = [0]*len(val_timesteps)

        for batch_idx, data in enumerate(self.val_dataloader):
            pos, z, mask, cls, caption, pointcloud_feature, sketch_feature  = data
            pos, z, mask, cls = \
                pos.to(self.device), z.to(self.device), \
                mask.to(self.device), cls.to(self.device)

            z = z * self.z_scaled
            pos = torch.concatenate([
                (pos[..., 3:6] + pos[..., 0:3]) * 0.5,  # 3D bbox center
                (pos[..., 3:6] - pos[..., 0:3]),        # 3D bbox size
                (pos[..., 8:10] - pos[..., 6:8]),       # 2D bbox size
            ], dim=-1)

            mask[...] = False

            # NOTE: pointcloud and sketch features are pre-computed and loaded in dataset.
            cond_global, cond_local = None, None
            if hasattr(self, 'text_encoder'):
                cond_global = self.text_encoder(caption).to(self.device)
            elif hasattr(self, 'pointcloud_encoder'):
                cond_local = pointcloud_feature.to(self.device)
            elif hasattr(self, 'sketch_encoder'):
                cond = sketch_feature.to(self.device)
                if cond.ndim == 3: cond_local = cond
                elif cond.ndim == 2: cond_global = cond
                else: raise NotImplementedError

            bsz = len(pos)

            total_count += len(pos)

            with torch.no_grad():
                for idx, step in enumerate(val_timesteps):
                    # Evaluate at timestep
                    # Add noise
                    if self.scheduler_type == "DDPM":
                        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                        timesteps = torch.randint(step - 1, step, (bsz,), device=self.device).long()

                        z_noise = torch.randn(z.shape).to(self.device)
                        z_noised = self.noise_scheduler.add_noise(z, z_noise, timesteps)

                        pos_noise = torch.randn(pos.shape).to(self.device)
                        pos_noised = self.noise_scheduler.add_noise(pos, pos_noise, timesteps)

                        # Predict noise
                        pred_noise = self.model(
                            pos = pos_noised,
                            z = z_noised,
                            timesteps = timesteps,
                            mask = mask,
                            class_label = cls,
                            cond_global = cond_global,
                            cond_local = cond_local,
                            is_train = True
                            )

                        # Loss
                        # If without pos embed, token generated position is random.
                        loss_z_noise = mse_loss(
                            pred_noise[~mask][:, :z.shape[-1]],
                            z_noise[~mask][:, :z.shape[-1]]
                        ).mean(-1).sum().item()
                        loss_pos_noise = mse_loss(
                            pred_noise[~mask][:, z.shape[-1]:],
                            pos_noise[~mask][:,]
                        ).mean(-1).sum().item()
                        bbox_l1 = l1_loss(
                            pred_noise[~mask][:, z.shape[-1]:],
                            pos_noise[~mask][:,]
                        ).mean(-1).sum().item()
                    
                    else:
                        raise NotImplementedError

                    # total_loss[idx] += loss
                    latent_loss[idx] += loss_z_noise
                    bbox_loss[idx] += loss_pos_noise
                    bbox_L1[idx] += bbox_l1

            progress_bar.update(1)
        progress_bar.close()

        # logging
        mse_latent_ = [loss / total_count for loss in latent_loss]
        wandb.log(dict([(f"val-latent-{step:04d}", mse_latent_[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)
        bbox_loss_ = [loss / total_count for loss in bbox_loss]
        wandb.log(dict([(f"val-bbox-{step:04d}", bbox_loss_[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)
        bbox_L1_ = [loss / total_count for loss in bbox_L1]
        wandb.log(dict([(f"val-bbox-L1-{step:04d}", bbox_L1_[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)

        # cal matric panel-wise
        max_face = self.val_dataset.max_face
        bbox_L1_pw_ = [loss / (total_count * max_face) for loss in bbox_L1]
        wandb.log(dict([(f"val-bbox_pw-L1-{step:04d}", bbox_L1_pw_[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)

        self.model.train() # set to train

        if hasattr(self.val_dataset, 'data_chunks') and len(self.val_dataset.data_chunks) > 1:
            self.val_dataset.update()
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                batch_size=self.batch_size, num_workers=16)

        # Calculating metrics on result ===
        return
    
    def save_model(self, save_last=False):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)

        if not save_last:
            ckpt_name = f'geometrygen_e{self.epoch:04d}.pt'
        else:
            ckpt_name = f'last.pt'

        torch.save(
            {
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                'z_scaled': self.z_scaled,
                'bbox_scaled': self.bbox_scaled,
                'optimizer': self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict()
            },
            os.path.join(ckpt_log_dir, ckpt_name))
        return
