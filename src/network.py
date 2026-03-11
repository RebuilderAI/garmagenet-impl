import os
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Union

from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

from src.constant import get_condition_dim


def sincos_embedding(input, dim, max_period=10000):
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class AutoencoderKLFastEncode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        sample_mode: str = "sample"
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.sample_mode = sample_mode

    def forward(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if self.sample_mode == "sample":
            latent_z = DiagonalGaussianDistribution(moments).sample()
        elif self.sample_mode == "mode":    
            latent_z = DiagonalGaussianDistribution(moments).mode()  # mode converge faster
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")
        return latent_z


class AutoencoderKLFastDecode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        decoded = self._decode(z).sample
        return decoded    


class TextEncoder:
    def __init__(self, encoder='CLIP', device='cuda'):

        self.device = device

        if encoder == 'CLIP':
            import transformers
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                "<Path2Model>/FLUX.1-dev", subfolder='tokenizer')
            text_encoder = transformers.CLIPTextModel.from_pretrained(
                "<Path2Model>/FLUX.1-dev", subfolder='text_encoder')
            self.text_encoder = nn.DataParallel(text_encoder).to(device).eval()
            self.text_emb_dim = 768
            self.text_embedder_fn = self._get_clip_text_embeds
        else:
            raise ValueError(f'Unsupported encoder {encoder}.')

        print(f"[DONE] Init {encoder} text encoder.")

    def _get_clip_text_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 16
    ):

        # print('*** [Text Encoder] prompt = ', prompt)

        with torch.no_grad():
            prompt = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            # print(text_input_ids)
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            prompt_embeds = self.text_encoder(input_ids=text_input_ids.to(self.device), output_hidden_states=False)

            # Use pooled output of CLIPTextModel
            prompt_embeds = prompt_embeds.pooler_output
            prompt_embeds = prompt_embeds.to(self.device)
            
            # print("*** CLIP prompt_embeds: ", prompt_embeds.shape, prompt_embeds.min(), prompt_embeds.max())

        return prompt_embeds

    def __call__(self, prompt):
        return self.text_embedder_fn(prompt)


class PointcloudEncoder:
    def __init__(self, encoder='POINT_E', device='cuda'):

        self.device = device

        if encoder == 'POINT_E':
            from src.models.pc_backbone.point_e.evals.feature_extractor import PointNetClassifier
            self.pointcloud_emb_dim = 512
            # Use home-dir cache if the original path doesn't exist
            import os
            cache_dir = '/data/lsr/models/PFID_evaluator'
            if not os.path.exists(cache_dir):
                cache_dir = os.path.expanduser('~/point_e_model_cache')
                os.makedirs(cache_dir, exist_ok=True)
            self.pointcloud_encoder = PointNetClassifier(devices=[self.device], cache_dir=cache_dir, device_batch_size=1)
            self.pointcloud_embedder_fn = self._get_pointe_pointcloud_embeds
        else:
            raise NotImplementedError

        # Test encoding text
        print(f"[DONE] Init {encoder} text encoder.")

    def _get_pointe_pointcloud_embeds(
            self,
            point_cloud
    ):
        feature_embedding =  self.pointcloud_encoder.get_features(point_cloud)
        return feature_embedding

    def __call__(self, point_cloud):
        return self.pointcloud_embedder_fn(point_cloud)


class SketchEncoder:
    def __init__(self, encoder='LAION2B', device="cuda:0"):
        self.device = device
        if encoder == 'LAION2B':
            import timm
            from safetensors import safe_open
            from src.models.sketch_feature_extractor.vit.utils.sketch_utils import _transform

            image_resolution = 224
            self.img_process = _transform(image_resolution)
            self.sketch_emb_dim = 1280
            VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'
            safetensors_path = '<Path2Model>/models--timm--vit_huge_patch14_clip_224.laion2b/snapshots/b8441fa3f968a5e469c166176ee82f8ce8dbc4eb/model.safetensors'
            vit_model = timm.create_model(VIT_MODEL, pretrained=False).to(self.device)
            vit_model.eval()
            with safe_open(safetensors_path, framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
                vit_model.load_state_dict(state_dict)
            self.sketch_encoder = vit_model
            self.sketch_embedder_fn = self._get_laion2b_sketch_embeds
        elif encoder == 'RADIO_V2.5-G':
            self.sketch_emb_dim = 1536
            self.sketch_encoder = None
            self.sketch_embedder_fn = None
        elif encoder in ['RADIO_V2.5-H', 'RADIO_V2.5-H_spatial']:
            from src.utils import resize_image
            self.resize_fn = resize_image
            # download online
            radio_model = torch.hub.load(
                'NVlabs/RADIO',
                'radio_model',
                version="radio_v2.5-h",
                progress=True,
                skip_validation=True
            )
            radio_model.cuda().eval()
            self.model = radio_model
            self.sketch_emb_dim = 3840 if encoder=='RADIO_V2.5-H' else 1280
            self.sketch_encoder = radio_model
            self.sketch_embedder_fn = self._get_radiov2_5h_sketch_embeds
        else:
            raise NotImplementedError
        print(f"[DONE] Init {encoder} sketch encoder.")

    def _get_laion2b_sketch_embeds(self, sketch_fp):
        image = Image.open(sketch_fp).convert('RGB')
        image = self.img_process(image)
        image = image.to(self.device).unsqueeze(0)
        sketch_features = self.sketch_encoder.forward_features(image).squeeze()
        sketch_features = sketch_features[1:]

        output_dict = {
            "spatial": sketch_features.detach().cpu().numpy(),
        }
        return output_dict

    def _get_radiov2_5h_sketch_embeds(self, sketch_fp, RESO=224):
        # image process ===
        image = Image.open(sketch_fp)
        image.load()

        width, height = image.size
        scale = min(RESO / width, RESO / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        bg_color = image.getpixel((0, 0))

        if new_width == RESO and new_height == RESO:
            pass
        else:
            background = Image.new('RGB', (RESO, RESO), bg_color)
            offset = ((RESO - new_width) // 2, (RESO - new_height) // 2)
            background.paste(resized_image, offset)
            resized_image = background

        # feature extract ===
        x = pil_to_tensor(resized_image).to(dtype=torch.float32, device='cuda') / 255.0
        x = x.unsqueeze(0)

        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):

            summary, spatial_feat = self.model(x)
            summary_np = summary[0].detach().cpu().numpy()
            spatial_feat = spatial_feat.detach().cpu().numpy()
            output_dict = {
                "summary": summary_np,
                "spatial": spatial_feat,
            }

        return output_dict


    def __call__(self, sketch_fp):
        return self.sketch_embedder_fn(sketch_fp)


class SpatialDiTBlock(nn.Module):
    """
    A DiT Block that includes Cross-Attention for spatial conditioning.
    Flow: AdaLN -> Self-Attn -> Cross-Attn -> AdaLN -> MLP
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        
        # Cross-Attention Layer
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN Modulation: 
        # Regresses 6 parameters: 
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # We generally treat Cross-Attn as "always on", or add gating for it too.
        # Here we stick to standard DiT-AdaLNZero for the self-attn/mlp parts.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, t, context=None, mask=None):
        """
        x: (B, N, D) - 3D Latents
        t: (B, D)    - Timestep Embeddings
        context: (B, M, D) - Image Features (RADIO)
        mask: (B, N) - Padding mask for x
        """
        
        # print('SpatialDiTBlock input:', x.shape, t.shape, context.shape, mask.shape if mask is not None else None)

        # 1. Regress Modulation Parameters from Timestep
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t)[:, None].chunk(6, dim=-1)
        )

        # 2. Self-Attention Block (Time-Modulated)
        x_norm = self.modulate(self.norm1(x), shift_msa, scale_msa)

        # Handle mask for Self-Attention if needed (tgt_key_padding_mask)
        # Note: nn.MultiheadAttention expects key_padding_mask
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + gate_msa * attn_out

        # 3. Cross-Attention Block (Spatially Aware)
        # We usually apply standard Norm before Cross-Attn
        if context is not None:
            x_norm_cross = self.norm_cross(x)
            if context.ndim == 2:
                context=context.unsqueeze(1)
            cross_out, _ = self.cross_attn(query=x_norm_cross, key=context, value=context)
            x = x + cross_out

        # 4. MLP Block (Time-Modulated)
        x_norm = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp * mlp_out

        return x


class GarmageNet(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """
    def __init__(self, p_dim=8, z_dim=3*4*4, out_dim=-1, embed_dim=768, num_heads=12, condition_dim=-1, num_layer=12, num_cf=-1):
        super(GarmageNet, self).__init__()
        
        self.p_dim = p_dim
        self.z_dim = z_dim

        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        
        self.out_dim = out_dim if out_dim > 0 else z_dim + p_dim
        
        self.n_classes = num_cf
        self.n_heads = num_heads        
        self.n_layer = num_layer[0] if isinstance(num_layer, List) else num_layer
        
        self.__init_embeddings()
        self.__init_net()        

    def __init_embeddings(self):
        
        self.z_embed = nn.Sequential(
            nn.Linear(self.z_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        nn.init.kaiming_normal_(self.z_embed[0].weight, mode="fan_in")

        if self.p_dim > 0:   
            self.p_embed = nn.Sequential(
                nn.Linear(self.p_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            nn.init.kaiming_normal_(self.p_embed[0].weight, mode="fan_in")

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.n_classes > 0: self.class_embed = Embedder(self.n_classes, self.embed_dim)
        
        if self.condition_dim > 0: 
            self.cond_embed = nn.Sequential(
                nn.Linear(self.condition_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )         

    def __init_net(self):
        self.net = nn.ModuleList([
            SpatialDiTBlock(self.embed_dim, self.n_heads) for _ in range(self.n_layer)
        ])
        
        for block in self.net:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        self.final_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        
        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.out_dim)
        )
  
    def forward(
            self, pos, z, timesteps, mask=None, class_label=None, 
            cond_global=None,       # global features like text prompt
            cond_local=None,        # local-aware features like pointcloud/sketch
            is_train=False):

        bsz, seq_len, _ = pos.shape

        x = self.z_embed(z) # [B, N, D]
        # print('*** x: ', x.shape, x.min(), x.max())
        if self.p_dim > 0: x = x + self.p_embed(pos)    # Add position signal (3D bounding box and 2D scale)
        # print('*** x: ', x.shape, x.min(), x.max())

        t_embs = self.time_embed(sincos_embedding(timesteps, self.embed_dim))
        # print('*** t_embs: ', t_embs.shape, t_embs.min(), t_embs.max())

        # Add class embedding to time embeddings
        if self.n_classes > 0 and class_label is not None:
            if is_train: class_label[torch.rand(class_label.shape) <= 0.1] = 0 # 10% random drop
            assert class_label.ndim==2
            t_embs = t_embs + self.class_embed(class_label).squeeze(-2)
            # print('*** t_embs: ', t_embs.shape, t_embs.min(), t_embs.max())

        cond_token = None

        # Processing global features (e.g. text prompt)
        if self.condition_dim > 0 and cond_global is not None:
            cond_token = self.cond_embed(cond_global)
            # t_embs = t_embs + cond_token
            # cond_token = None
            # print('*** t_embs: ', t_embs.shape, t_embs.min(), t_embs.max())

        # Processing local-aware features (e.g. pointcloud/sketch)
        if self.condition_dim > 0 and cond_local is not None:
            cond_token = self.cond_embed(cond_local)
            cond_token = cond_token[:, None] if len(cond_token.shape) == 2 else cond_token  # [B, n_surfs, emb_dim]
            # print('*** cond_token: ', cond_token.shape, cond_token.min(), cond_token.max())
        
        for block in self.net: x = block(x, t_embs, context=cond_token, mask=mask)
            
        x = self.final_norm(x)
        pred = self.fc_out(x)

        return pred