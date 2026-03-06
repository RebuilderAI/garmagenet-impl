import os
import argparse

from datasets.garmage import *
from trainer import *


def get_args_ldm():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--expr', type=str, default="expr_default", help='environment')
    parser.add_argument('--log_dir', type=str, default="log", help='name of the log folder.')
    parser.add_argument("--option", type=str, choices=['garmagenet'], default='garmagenet')
    parser.add_argument("--denoiser_type", type=str, choices=['default'], default='default', help="Choose ldm type.")
    parser.add_argument("--scheduler", type=str, default="DDPM", choices=["DDPM"], help="")
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights.')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning.')
    parser.add_argument("--gpu", type=int, nargs='+', default=None, help="GPU IDs to use for training (default: None).")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')

    parser.add_argument('--train_nepoch', type=int, default=3000, help='Number of epochs to train for.')
    parser.add_argument('--test_nepoch', type=int, default=10, help='Number of epochs to test model.')
    parser.add_argument('--save_nepoch', type=int, default=1000, help='Number of epochs to save model.')
    parser.add_argument('--save_last_nepoch', type=int, default=1000, help='Number of epochs to save model.')

    # Dataset parameters
    parser.add_argument('--data', type=str, default=None, required=True,
                        help='Path to data folder')
    parser.add_argument('--use_data_root', action="store_true",
                        help='If data list store relative path, use this flag.')
    parser.add_argument('--list', type=str, default=None, required=True,
                        help='Path to the datalist file.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to cached data (with latents).')
    parser.add_argument('--max_face', type=int, default=32, help='Maximum number of panels.')
    parser.add_argument('--bbox_scaled', type=float, default=1.0, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=None, help='scaled the latent z')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation.')
    parser.add_argument('--data_fields', nargs='+', default=['surf_ncs', 'surf_mask'], help="Data fields to encode.")
    parser.add_argument("--padding", default="zero", type=str, choices=['repeat', 'zero', 'zerolatent'])
    parser.add_argument('--chunksize', type=int, default=256, help='Chunk size for data loading.')

    # Encoder parameters
    parser.add_argument("--text_encoder", type=str, default=None, choices=[None, 'CLIP', 'T5', 'GME'], help="Text encoder type when applying text as generation condition.")
    parser.add_argument("--pointcloud_encoder", type=str, default=None, choices=[None, 'POINT_E'], help="Pointcloud encoder type when applying pointcloud as generation condition.")
    parser.add_argument("--pointcloud_sampled_dir", type=str, default=None,  help="Prepared pointcloud.")
    parser.add_argument("--pointcloud_feature_dir", type=str, default=None,  help="Prepared pointcloud feature.")
    parser.add_argument("--sketch_encoder", type=str, default=None, choices=[None, 'LAION2B', "RADIO_V2.5-G", "RADIO_V2.5-H", "RADIO_V2.5-H_spatial"], help="Sketch encoder type when applying sketch as generation condition.")
    parser.add_argument("--sketch_feature_dir", type=str, default=None,  help="Prepared sketch feature.")
    parser.add_argument("--condition_type", type=str, default='summary', choices=['summary', 'spatial'], help="Text encoder type when applying text as generation condition.")
    parser.add_argument("--feature_kwd",  type=str, default="0", help="Key for feature choice.")

    # Model parameters
    parser.add_argument('--surfvae', type=str, default=None, required=True, help='Path to pretrained VAE weights')
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Block dimensions of the VAE model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channel of the VAE model.')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embding dimension of LDM model.')
    parser.add_argument('--num_layer', type=int, nargs='+', default=12, help='Layer num of LDM model.')
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    # saved folder
    args.log_dir = f'{args.log_dir}/{args.expr}'
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    return args


def run(args):
    # Initialize dataset and trainer ===
    if args.option == 'garmagenet':
        train_dataset = GarmageNetData(
            args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = GarmageNetData(
            args.data, args.list, validate=True, aug=False, args=args)
        ldm = GarmageNetTrainer(args, train_dataset, val_dataset)
    else:
        raise NotImplementedError(args.option)

    print('Start training...')
    
    # Main training loop
    start_epoch = ldm.epoch
    for _ in range(start_epoch, args.train_nepoch):

        # Train for one epoch
        ldm.train_one_epoch()        

        # Evaluate model performance on validation set
        if ldm.epoch % args.test_nepoch == 0:
            ldm.test_val()

        # save model
        if ldm.epoch % args.save_nepoch == 0:
            ldm.save_model()

        # save last model per epoch
        if ldm.epoch % 1000 == 0:
            ldm.save_model(save_last=True)
    return


if __name__ == "__main__":
    args = get_args_ldm()
    run(args)