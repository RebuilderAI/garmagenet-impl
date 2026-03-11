import os
import pickle
import random
from tqdm import tqdm
from glob import glob

import torch
import numpy as np

from src.constant import _PANEL_CLS_NUM


class VaeData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """
    def __init__(self,
                 input_data,
                 input_list,
                 data_fields=['surf_ncs'],
                 validate=False,
                 aug=False,
                 chunksize=-1,
                 args=None):
        self.args = args

        self.validate = validate
        self.aug = aug

        self.data_root = input_data
        self.data_fields = data_fields

        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, "rb") as tf:
            self.data_list = pickle.load(tf)['val' if validate else 'train']
            if args.use_data_root:
                self.data_list = [
                    os.path.join(self.data_root, os.path.basename(x)) for x in self.data_list \
                    if os.path.exists(os.path.join(self.data_root, os.path.basename(x)))
                ]
        print("Total items: ", len(self.data_list))

        self.chunksize = chunksize if chunksize > 0 and chunksize < len(self.data_list) else len(self.data_list)
        if self.validate: self.chunksize = self.chunksize // 2
        self.chunk_idx = -1
        self.data_chunks = [self.data_list[i:i+self.chunksize] for i in range(0, len(self.data_list), self.chunksize)]
        print('Data chunks: num_chunks=%d, chunksize=%d.'%(len(self.data_chunks), self.chunksize))

        self.__next_chunk__(lazy=False)


    def __next_chunk__(self, lazy=False):

        if lazy and np.random.rand() < 0.5: return

        chunk_idx = (self.chunk_idx + 1) % len(self.data_chunks)
        if self.chunk_idx == chunk_idx: return
        else: self.chunk_idx = chunk_idx
        print('Switching to chunk %d/%d'%(self.chunk_idx, len(self.data_chunks)))

        cache = []
        for uid in tqdm(self.data_chunks[self.chunk_idx]):
            path = uid

            try:
                with open(path, "rb") as tf: data = pickle.load(tf)

                surf_ncs = []
                # NCS VAE
                if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
                if 'surf_wcs' in self.data_fields: surf_ncs.append(data['surf_wcs'].astype(np.float32))
                if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
                if 'surf_normals' in self.data_fields: surf_ncs.append(data['surf_normals'].astype(np.float32))
                if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32)*2.0-1.0)

                surf_ncs = np.concatenate(surf_ncs, axis=-1)
                cache.append(surf_ncs)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        self.cache = np.vstack(cache)
        self.num_channels = self.cache.shape[-1]
        self.resolution = self.cache.shape[1]

        print('Load chunk [%03d/%03d]: '%(self.chunk_idx, len(self.data_chunks)), self.cache.shape)

    def update(self): self.__next_chunk__(lazy=True)

    def __len__(self): return len(self.cache) * 8

    def __getitem__(self, index):
        return torch.FloatTensor(self.cache[index%len(self.cache)])


class LazyVaeData(torch.utils.data.Dataset):
    """
    Lazy-loading VAE Dataset with pre-extracted per-panel .npy files.

    On first run, extracts needed fields from each garment pickle and saves
    individual .npy files per panel (~1MB each vs 50MB pickle).
    Subsequent runs load directly from .npy — 50x less I/O per item.
    No in-process caching, so no RAM explosion with multi-worker DataLoader.

    RAM usage: ~few MB (index only).
    Return type matches VaeData exactly: torch.FloatTensor of shape (H, W, C).
    """
    def __init__(self,
                 input_data,
                 input_list,
                 data_fields=['surf_ncs'],
                 validate=False,
                 aug=False,
                 chunksize=-1,  # ignored, kept for arg compatibility
                 args=None):
        self.args = args
        self.validate = validate
        self.data_root = input_data
        self.data_fields = data_fields

        print('Loading %s data (lazy mode)...' % ('validation' if validate else 'training'))
        with open(input_list, "rb") as tf:
            file_list = pickle.load(tf)['val' if validate else 'train']
            if args.use_data_root:
                file_list = [
                    os.path.join(self.data_root, os.path.basename(x)) for x in file_list
                    if os.path.exists(os.path.join(self.data_root, os.path.basename(x)))
                ]
        print("Total garment files: ", len(file_list))

        # Pre-extracted panel directory
        fields_key = '_'.join(sorted(self.data_fields))
        self.panel_dir = os.path.join(os.path.dirname(input_list), f'panels_{fields_key}')

        # Build panel .npy files (one-time extraction) and get paths
        self.panel_paths = self._build_panel_npy(file_list, input_list, validate)
        print("Total panels (dataset length): ", len(self.panel_paths))

        # Load one sample to determine shape metadata
        sample = np.load(self.panel_paths[0])
        self.num_channels = sample.shape[-1]
        self.resolution = sample.shape[0]
        print(f"Panel shape: ({self.resolution}, {self.resolution}, {self.num_channels})")

    def _build_panel_npy(self, file_list, input_list, validate):
        """
        Extract per-panel .npy files from garment pickles (one-time).
        Each .npy is ~1MB (256x256x4 float32) vs 50MB pickle.
        Returns list of .npy file paths.
        """
        split_name = 'val' if validate else 'train'
        index_cache_fp = os.path.join(self.panel_dir, f'index_{split_name}.pkl')

        # Check if already extracted
        if os.path.exists(index_cache_fp):
            with open(index_cache_fp, 'rb') as f:
                panel_paths = pickle.load(f)
            if len(panel_paths) > 0 and os.path.exists(panel_paths[0]):
                print(f'Loaded {len(panel_paths)} pre-extracted panels from {self.panel_dir}')
                return panel_paths
            else:
                print('Panel cache is stale, re-extracting...')

        # Extract panels to individual .npy files
        os.makedirs(self.panel_dir, exist_ok=True)
        panel_paths = []
        skipped = 0

        for fp in tqdm(file_list, desc=f'Extracting panels to .npy ({split_name})'):
            try:
                with open(fp, 'rb') as f:
                    data = pickle.load(f)

                garment_id = os.path.splitext(os.path.basename(fp))[0]
                n_panels = data['surf_ncs'].shape[0]

                for i in range(n_panels):
                    channels = []
                    if 'surf_ncs' in self.data_fields:
                        channels.append(data['surf_ncs'][i].astype(np.float32))
                    if 'surf_wcs' in self.data_fields:
                        channels.append(data['surf_wcs'][i].astype(np.float32))
                    if 'surf_uv_ncs' in self.data_fields:
                        channels.append(data['surf_uv_ncs'][i].astype(np.float32))
                    if 'surf_normals' in self.data_fields:
                        channels.append(data['surf_normals'][i].astype(np.float32))
                    if 'surf_mask' in self.data_fields:
                        channels.append(data['surf_mask'][i].astype(np.float32) * 2.0 - 1.0)

                    panel = np.concatenate(channels, axis=-1)
                    npy_path = os.path.join(self.panel_dir, f'{garment_id}_p{i:02d}.npy')
                    np.save(npy_path, panel)
                    panel_paths.append(npy_path)
            except Exception as e:
                print(f"Error extracting {fp}: {e}")
                skipped += 1
                continue

        if skipped > 0:
            print(f'Skipped {skipped} files during extraction.')

        # Save index
        with open(index_cache_fp, 'wb') as f:
            pickle.dump(panel_paths, f)
        print(f'Extracted {len(panel_paths)} panels to {self.panel_dir} '
              f'(~{len(panel_paths) // 1024}GB)')

        return panel_paths

    def update(self):
        """No-op for trainer compatibility."""
        pass

    def __len__(self):
        return len(self.panel_paths)

    def __getitem__(self, index):
        panel = np.load(self.panel_paths[index])
        return torch.FloatTensor(panel)


class GarmageNetData(torch.utils.data.Dataset):
    """ Surface latent geometry Dataloader """
    def __init__(
            self,
            input_data,
            input_list,
            validate=False,
            aug=False,
            args=None,
    ):
        self.args = args

        self.max_face = args.max_face
        self.bbox_scaled = args.bbox_scaled

        self.validate = validate
        self.aug = aug

        # Load data
        self.data_root = input_data
        print('Data Root: ', self.data_root)

        self.cache_fp = os.path.join(
            args.cache_dir if args.cache_dir else args.surfvae.replace('.pt', '').replace('log', 'cache'),
            'garmagenet_%s.pkl' % ('validate' if validate else 'train')
        )
        print("dataset init ")

        self.data_fields = args.data_fields
        self.padding = args.padding

        self.condition_type = args.condition_type
        self.feature_kwd = args.feature_kwd
        print(f"Condition Type: {self.condition_type}.")
        if "pointcloud_feature" in self.data_fields:
            print("Use Pointcloud feature.")
            # self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, "cuda")
            self.pointcloud_sampled_dir = args.pointcloud_sampled_dir
            if args.pointcloud_sampled_dir and not os.path.exists(self.pointcloud_sampled_dir):
                raise FileNotFoundError(f"pointcloud_sampled_dir {self.pointcloud_sampled_dir} not exists")
            self.pointcloud_feature_dir = args.pointcloud_feature_dir
            if args.pointcloud_feature_dir and not os.path.exists(self.pointcloud_feature_dir):
                raise FileNotFoundError(f"pointcloud_feature_dir {self.pointcloud_feature_dir} not exists")
        if "sketch_feature" in self.data_fields:
            print("Use Sketch feature.")
            self.sketch_feature_dir = args.sketch_feature_dir
            if not os.path.exists(self.sketch_feature_dir):
                raise FileNotFoundError(f"sketch_feature_dir {self.sketch_feature_dir} not exists")

        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, 'rb') as f: self.data_list = pickle.load(f)['val' if self.validate else 'train']
        self._cache_only = False
        if args.use_data_root:
            filtered = [os.path.join(self.data_root, os.path.basename(x)) for x in self.data_list
                        if os.path.exists(os.path.join(self.data_root, os.path.basename(x)))]
            if len(filtered) > 0:
                self.data_list = filtered
            elif os.path.exists(self.cache_fp):
                # Cache-only mode: raw garmage pkl files not available, use pre-built latent cache.
                # This allows LDM training without the full ~638GB garmages/ directory.
                print(f'[Cache-only mode] Raw data not found in {self.data_root}. '
                      f'Will use pre-built cache at {self.cache_fp}')
                self._cache_only = True
            else:
                raise FileNotFoundError(
                    f"No raw data in {self.data_root} and no cache at {self.cache_fp}. "
                    f"Provide either raw garmage data or pre-built latent cache files.")

        if self._cache_only:
            self._init_from_cache()
        else:
            random.shuffle(self.data_list)
            print('Total items: ', len(self.data_list), self.data_list[0])
            self.load_one_to_init()

    def _init_from_cache(self):
        """Initialize metadata from pre-built cache when raw garmage files are unavailable.
        This enables LDM training without the ~638GB garmages/ directory — only the
        pre-encoded latent cache files (~425MB) are needed."""
        import math
        print(f'[Cache-only] Loading metadata from {self.cache_fp}')
        with open(self.cache_fp, 'rb') as f:
            cache = pickle.load(f)

        # pos_dim: surf_pos stores (bbox_min, bbox_max) pairs → half is the actual dimension
        self.pos_dim = cache['surf_pos'].shape[-1] // 2
        self.num_classes = _PANEL_CLS_NUM if 'surf_cls' in self.data_fields else 0

        # num_channels: count surface data channels from data_fields
        _ch_map = {'surf_ncs': 3, 'surf_wcs': 3, 'surf_uv_ncs': 2, 'surf_normals': 3, 'surf_mask': 1}
        self.num_channels = sum(_ch_map.get(f, 0) for f in self.data_fields)

        # resolution: reverse-engineer from latent dimension + VAE block structure
        # latent_dim = (resolution / 2^(n_blocks-1))^2 * latent_channels
        latent_dim = cache['latent'].shape[-1]
        bottleneck = int(math.sqrt(latent_dim / self.args.latent_channels))
        self.resolution = bottleneck * (2 ** (len(self.args.block_dims) - 1))

        # Replace data_list with index range matching cache item count
        num_items = len(cache['item_idx'])
        self.data_list = list(range(num_items))
        random.shuffle(self.data_list)

        print(f'[Cache-only] channels={self.num_channels}, resolution={self.resolution}, '
              f'pos_dim={self.pos_dim}, num_classes={self.num_classes}, items={num_items}')
        del cache

    def load_one_to_init(self):
        """
        __load_one__ need encoder while processing pointcloud condition
        """
        for i in range(0, len(self.data_list)):
            if os.path.exists(self.data_list[i]):
                break
        with open(self.data_list[i], 'rb') as f:
            data = pickle.load(f)
        # Load surfpos
        surf_pos = []
        if 'surf_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_bbox_wcs'].astype(np.float32))
        if 'surf_uv_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_uv_bbox_wcs'].astype(np.float32))
        surf_pos = np.concatenate(surf_pos, axis=-1)
        # Load surfncs
        surf_ncs = []
        if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
        if 'surf_wcs' in self.data_fields: surf_ncs.append(data['surf_wcs'].astype(np.float32))
        if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
        if 'surf_normals' in self.data_fields: surf_ncs.append(data['surf_normals'].astype(np.float32))
        if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32) * 2.0 - 1.0)
        surf_ncs = np.concatenate(surf_ncs, axis=-1)
        if not hasattr(self, 'num_channels'): self.num_channels = surf_ncs.shape[-1]
        if not hasattr(self, 'resolution'): self.resolution = surf_ncs.shape[1]
        if not hasattr(self, 'pos_dim'): self.pos_dim = surf_pos.shape[-1] // 2
        if not hasattr(self, 'num_classes'): self.num_classes = _PANEL_CLS_NUM if 'surf_cls' in self.data_fields else 0

    def __len__(self):
        return len(self.data_list)

    def init_encoder(self, z_encoder, cond_encoder, z_scaled=None):
        print("Init z_encoder...")
        self.z_encoder = z_encoder

        if "pointcloud_feature" in self.data_fields:
            self.pointcloud_encoder = cond_encoder

        if z_scaled is not None: self.z_scaled = z_scaled
        if not hasattr(self, 'cache'): self.__load_and_encode_all__()

    def __init_zero_latent__(self):
        if hasattr(self, 'zero_latent'): return

        try:
            z_device = self.z_encoder.module.parameters().__next__().device
        except:
            z_device = self.z_encoder.parameters().__next__().device

        with torch.no_grad():
            self.zero_latent = self.z_encoder(
                torch.zeros((1, self.num_channels, self.resolution, self.resolution), device=z_device)
                ).flatten(start_dim=1).detach().cpu()

        print('Init zero latent: ', self.zero_latent.shape,self.zero_latent.min(), self.zero_latent.max(), self.zero_latent.mean(), self.zero_latent.std())

    def __load_one__(self, data_fp):
        with open(data_fp, 'rb') as f: data = pickle.load(f)
        # Load surfpos
        surf_pos = []
        if 'surf_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_bbox_wcs'].astype(np.float32))
        if 'surf_uv_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_uv_bbox_wcs'].astype(np.float32))
        surf_pos = np.concatenate(surf_pos, axis=-1)

        # Load surfncs
        surf_ncs = []
        if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
        if 'surf_wcs' in self.data_fields: surf_ncs.append(data['surf_wcs'].astype(np.float32))
        if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
        if 'surf_normals' in self.data_fields: surf_ncs.append(data['surf_normals'].astype(np.float32))
        if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32)*2.0-1.0)
        surf_ncs = np.concatenate(surf_ncs, axis=-1)

        # Load caption
        caption = data['caption'] if 'caption' in self.data_fields else ''
        data_fp = data["data_fp"]

        # Load pointcloud feature
        if 'pointcloud_feature' in self.data_fields:
            # if pointcloud feature prepared
            if self.pointcloud_feature_dir is not None:

                assert self.pointcloud_sampled_dir is None

                sample_type_list = ["surface_uniform", "fps", "non_uniform"]
                pointcloud_feature = []
                sampled_pc_cond = []
                for sample_type in sample_type_list:
                    try:
                        pointcloud_feature_fp = glob(os.path.join(self.pointcloud_feature_dir, data_fp, f"feature_{self.args.pointcloud_encoder}", f"*{sample_type}*.npy"))[0]
                        sampled_pc_fp = glob(os.path.join(self.pointcloud_feature_dir, data_fp, f"sampled_pc", f"*{sample_type}*.npy"))[0]
                        feature = np.load(pointcloud_feature_fp)
                        if feature.ndim==2:
                            feature = feature[0]
                        pc_cond = np.load(sampled_pc_fp)
                        pointcloud_feature.append(feature)
                        sampled_pc_cond.append(pc_cond)
                    except Exception as e:
                        continue
                if len(pointcloud_feature) == 0:
                    raise FileNotFoundError(data_fp)
            else:
                # If no pre-sampled point cloud (uniformly sampled from the OBJ) is prepared in advance, then directly sample the point cloud from the GT garment.
                if self.pointcloud_sampled_dir is None:
                    n_surfs = len(data['surf_bbox_wcs'])
                    surf_wcs = data["surf_wcs"].reshape(n_surfs, -1, 3)
                    surf_mask = data["surf_mask"].reshape(n_surfs, -1)
                    valid_pts = surf_wcs[surf_mask]
                    sampled_pc_cond = valid_pts[np.random.randint(0, len(valid_pts), size=2048)]
                # Using pre-sampled pointcloud.
                else:
                    pointcloud_sample_fp = sorted(glob(os.path.join(os.path.join(self.pointcloud_sampled_dir, data_fp), "*.npy")))
                    if len(pointcloud_sample_fp) == 0:
                        raise FileExistsError(f"pointcloud_sample_fp: {pointcloud_sample_fp} not exist.")
                    else:
                        pointcloud_sample_fp = pointcloud_sample_fp[0]
                    sampled_pc_cond = np.load(pointcloud_sample_fp)
                pointcloud_feature = self.pointcloud_encoder(sampled_pc_cond)

        # Load sketch feature
        if "sketch_feature" in self.data_fields:
            if 'sketch_feature' not in data or data['sketch_feature'] is None:
                sketch_feature_fp = sorted(glob(os.path.join(os.path.join(self.sketch_feature_dir, data_fp), f"*_{self.feature_kwd}.pkl")))
                if len(sketch_feature_fp) == 0:
                    FileExistsError(f"pointcloud_sample_fp: {pointcloud_sample_fp} not exist. This may because no corresponding image.")
                else:
                    sketch_feature_fp = sketch_feature_fp[0]
                with open(sketch_feature_fp, "rb") as f:
                    sketch_feature = pickle.load(f)[self.condition_type]
                    if isinstance(sketch_feature, torch.Tensor):
                        sketch_feature = sketch_feature.detach().cpu().numpy()
            else:
                sketch_feature = data['sketch_feature']

            if sketch_feature.ndim == 1:
                sketch_feature = sketch_feature[np.newaxis, ...]

        # Load semantics
        surf_cls = data['surf_cls'][..., None] if 'surf_cls' in self.data_fields \
            else np.zeros((self.max_face, 1)) - 1

        if not hasattr(self, 'num_channels'): self.num_channels = surf_ncs.shape[-1]
        if not hasattr(self, 'resolution'): self.resolution = surf_ncs.shape[1]
        if not hasattr(self, 'pos_dim'): self.pos_dim = surf_pos.shape[-1] // 2
        if not hasattr(self, 'num_classes'): self.num_classes = _PANEL_CLS_NUM if 'surf_cls' in self.data_fields else 0

        return (
            torch.FloatTensor(surf_pos),
            torch.FloatTensor(surf_ncs),
            torch.LongTensor(surf_cls),
            torch.FloatTensor(pointcloud_feature) if 'pointcloud_feature' in self.data_fields else None,
            sampled_pc_cond if 'pointcloud_feature' in self.data_fields else None,
            torch.FloatTensor(sketch_feature) if 'sketch_feature' in self.data_fields else None,
            caption,
            data["data_fp"]
        )

    def __pad_latents__(self, surf_pos, surf_latents, surf_cls):

        n_surfs, n_pads = surf_pos.shape[0], self.max_face-surf_pos.shape[0]

        if self.padding == 'repeat':
            rand_idx = torch.randperm(self.max_face)
            pad_idx = torch.randint(0, n_surfs, (n_pads,))
            surf_pos = torch.cat([surf_pos, surf_pos[pad_idx, ...]], dim=0)[rand_idx, ...]
            surf_latents = torch.cat([surf_latents, surf_latents[pad_idx, ...]], dim=0)[rand_idx, ...]
            surf_cls = torch.cat([surf_cls, surf_cls[pad_idx, ...]], dim=0)[rand_idx, ...]
            pad_mask = torch.cat([
                torch.zeros(n_surfs, dtype=bool), torch.ones(n_pads, dtype=bool)
                ], dim=0)[rand_idx, ...]

        elif self.padding == 'zero':
            pad_idx = torch.randperm(n_surfs)
            pad_mask = torch.cat([
                torch.zeros(n_surfs, dtype=bool), torch.ones(n_pads, dtype=bool)
            ], dim=0)
            surf_pos = torch.cat([
                surf_pos[pad_idx, ...], torch.zeros((n_pads, *surf_pos.shape[1:]), dtype=surf_pos.dtype, device=surf_pos.device)
            ], dim=0)
            surf_latents = torch.cat([
                surf_latents[pad_idx, ...], torch.zeros((n_pads, *surf_latents.shape[1:]), dtype=surf_latents.dtype, device=surf_latents.device)
            ], dim=0)
            surf_cls = torch.cat([
                surf_cls[pad_idx, ...], torch.zeros((n_pads, *surf_cls.shape[1:]), dtype=surf_cls.dtype, device=surf_cls.device)-1
            ], dim=0)

        elif self.padding == 'zerolatent':
            pad_idx = torch.randperm(n_surfs)
            pad_mask = torch.cat([
                torch.zeros(n_surfs, dtype=bool), torch.ones(n_pads, dtype=bool)
            ], dim=0)
            surf_pos = torch.cat([
                surf_pos[pad_idx, ...], torch.zeros((n_pads, *surf_pos.shape[1:]), dtype=surf_pos.dtype, device=surf_pos.device)
            ], dim=0)
            surf_latents = torch.cat([
                surf_latents[pad_idx, ...], self.zero_latent.repeat(n_pads, 1)
            ], dim=0)
            surf_cls = torch.cat([
                surf_cls[pad_idx, ...], torch.zeros((n_pads, *surf_cls.shape[1:]), dtype=surf_cls.dtype, device=surf_cls.device)-1
            ], dim=0)

        else:
            raise ValueError('Invalid padding mode: %s'%self.padding)

        return surf_pos, surf_latents, surf_cls, pad_mask

    def __load_and_encode_all__(self):

        try:
            z_device = self.z_encoder.module.parameters().__next__().device
        except:
            z_device = self.z_encoder.parameters().__next__().device

        if os.path.exists(self.cache_fp):
            with open(self.cache_fp, 'rb') as f: self.cache = pickle.load(f)
            if not hasattr(self, 'z_scaled'): self.z_scaled = 1.0 / (self.cache['latent'].std() + 1e-6)
            if self.padding == 'zerolatent' and not hasattr(self, 'zero_latent'): self.__init_zero_latent__()
            print('Load cache from %s: '%(self.cache_fp), self.cache['surf_pos'].shape, self.cache['latent'].shape, ' z_scaled: ', self.z_scaled)
            return

        print('Encoding all items from: ', self.data_root)

        if not (hasattr(self, 'z_encoder') and self.z_encoder is not None): return

        cache = {
            'data_id': [],
            'surf_pos': [],
            'latent': [],
            'surf_cls': [],
            'caption': [],
            'pointcloud_feature': [],
            'sampled_pc_cond': [],
            'sketch_feature': [],
            'item_idx': [],  # start and end index of each item in the cache
            'data_fp': []
            }
        if 'pointcloud_feature' in self.data_fields and self.pointcloud_feature_dir:
            cache['pccond_item_idx'] = []
            pccond_start_idx, pccond_end_idx = 0, 0

        start_idx, end_idx = 0, 0
        for data_id in tqdm(self.data_list):
            data_fp = data_id
            try:
                surf_pos, surf_ncs, surf_cls, pointcloud_feature, sampled_pc_cond, sketch_feature, caption, data_fp_orig = self.__load_one__(data_fp)
                assert surf_pos.shape[0] <= self.max_face
                with torch.no_grad():
                    z = self.z_encoder(surf_ncs.permute(0, 3, 1, 2).to(z_device)).flatten(start_dim=1)

                start_idx = end_idx
                end_idx = start_idx + surf_pos.shape[0]

                if 'pointcloud_feature' in self.data_fields and self.pointcloud_feature_dir:
                    pccond_start_idx = pccond_end_idx
                    pccond_end_idx = pccond_start_idx + pointcloud_feature.shape[0]
                    cache['pccond_item_idx'].append((pccond_start_idx, pccond_end_idx))

                cache['surf_pos'].append(surf_pos)
                cache['surf_cls'].append(surf_cls)
                cache['caption'].append(caption)
                cache['item_idx'].append((start_idx, end_idx))
                cache['data_id'].append(data_id)
                cache['pointcloud_feature'].append(pointcloud_feature)
                if isinstance(sampled_pc_cond, list):
                    cache['sampled_pc_cond'].extend(sampled_pc_cond)
                else:
                    cache['sampled_pc_cond'].append(sampled_pc_cond)
                cache['sketch_feature'].append(sketch_feature)
                cache['latent'].append(z.to(surf_pos.device).to(surf_pos.dtype))
                cache['data_fp'].append(data_fp_orig)

            except Exception as e:
                print(f"Error loading {data_fp}: {e}")
                continue

        cache['surf_pos'] = torch.cat(cache['surf_pos'], dim=0)
        cache['latent'] = torch.cat(cache['latent'], dim=0)
        cache['surf_cls'] = torch.cat(cache['surf_cls'], dim=0)
        if "pointcloud_feature" in self.data_fields:
            cache['pointcloud_feature'] = torch.cat(cache['pointcloud_feature'], dim=0)
            cache['sampled_pc_cond'] = np.stack(cache['sampled_pc_cond'], axis=0)
        if "sketch_feature" in self.data_fields:
            cache['sketch_feature'] = torch.cat(cache['sketch_feature'], dim=0)

        self.cache = cache

        print('Saving latents tp %s: '%self.cache_fp, self.cache['latent'].shape, self.cache['latent'].std())
        os.makedirs(os.path.dirname(self.cache_fp), exist_ok=True)
        with open(self.cache_fp, 'wb') as f: pickle.dump(cache, f)

        if not hasattr(self, 'z_scaled'):
            self.z_scaled = 1.0 / (self.cache['latent'].std() + 1e-6)
            print("Initialize z_scaled: ", self.z_scaled)

        if self.padding == 'zerolatent' and not hasattr(self, 'zero_latent'):
            with torch.no_grad():
                self.zero_latent = self.z_encoder(
                    torch.zeros((1, self.num_channels, self.resolution, self.resolution), device=z_device)
                    ).flatten(start_dim=1).detach().cpu()
            print('Init zero latent: ', self.zero_latent.shape,self.zero_latent.min(), self.zero_latent.max(), self.zero_latent.mean(), self.zero_latent.std())

        del self.z_encoder
        torch.cuda.empty_cache()

        print('Encode all data: ', self.cache['latent'].shape, self.cache['surf_pos'].shape)

    def __getitem__(self, index):

        start_idx, end_idx = self.cache['item_idx'][index%len(self.cache['item_idx'])]
        if "pccond_item_idx" in self.cache:
            pccond_start_idx, pccond_end_idx = self.cache['pccond_item_idx'][index%len(self.cache['pccond_item_idx'])]

        caption = self.cache['caption'][index%len(self.cache['item_idx'])]

        surf_pos = self.cache['surf_pos'][start_idx:end_idx, ...] * self.bbox_scaled
        surf_latents = self.cache['latent'][start_idx:end_idx, ...] * self.z_scaled
        surf_cls = self.cache['surf_cls'][start_idx:end_idx, ...]

        surf_pos, surf_latents, surf_cls, pad_mask = self.__pad_latents__(
            surf_pos, surf_latents, surf_cls)

        if caption:
            caption = [x.strip().lower() for x in caption.split(',') if x.strip().lower() != 'dress']
            random.shuffle(caption)
            if np.random.rand() > 0.5:
                if len(caption) > 4: caption=caption[:random.randint(4, len(caption))]
                if 'dress' not in caption: caption.insert(0, 'dress')

            caption = ', '.join(caption)

        if  'pointcloud_feature' in self.data_fields:
            if "pccond_item_idx" in self.cache:
                pointcloud_feature = self.cache['pointcloud_feature'][pccond_start_idx:pccond_end_idx, ...]
                pointcloud_feature = pointcloud_feature[torch.randint(0, len(pointcloud_feature), (1,))][0]
            else:
                pointcloud_feature = self.cache['pointcloud_feature'][index%len(self.cache['item_idx'])]
        else:
            pointcloud_feature = 0
        sketch_feature = self.cache['sketch_feature'][index%len(self.cache['item_idx'])]  \
            if 'sketch_feature' in self.data_fields else 0

        return (
            surf_pos, surf_latents, pad_mask, surf_cls, caption, pointcloud_feature, sketch_feature
        )