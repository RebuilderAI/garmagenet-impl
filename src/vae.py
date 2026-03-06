import os
import argparse

from trainer import VAETrainer
from datasets.garmage import VaeData, LazyVaeData


def get_args_vae():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--expr', type=str, default="surface_vae", help='Experiment name')
    parser.add_argument('--log_dir', type=str, default="log", help='Name of the log folder.')
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')

    parser.add_argument('--train_nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')

    # Dataset configuration
    parser.add_argument('--data', type=str, default=None, required=True, help='Path to data folder')
    parser.add_argument('--use_data_root', action="store_true", help='If data list store relative path, use this flag.')
    parser.add_argument('--list', type=str, default=None, required=True, help='Path to the datalist file.')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument('--data_fields', nargs='+', default=['surf_ncs', 'surf_mask'], help="Data fields to encode.")
    parser.add_argument('--chunksize', type=int, default=-1, help='Chunk size for data loading')
    parser.add_argument('--lazy_loading', action='store_true', help='Use lazy disk-based loading instead of RAM caching')

    # Model parameters
    parser.add_argument("--vae_type", choices=["kl"], default="kl")
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Block dimensions of the VAE model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')

    args = parser.parse_args()
    
    # saved folder
    args.log_dir = f'{args.log_dir}/{args.expr}'
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    return args


def run(args):
    print('Args:', args)
    
    # Initialize dataset loader and trainer
    DatasetClass = LazyVaeData if args.lazy_loading else VaeData
    train_dataset = DatasetClass(
        args.data, args.list, data_fields=args.data_fields,
        validate=False, aug=args.data_aug, chunksize=args.chunksize, args=args)
    val_dataset = DatasetClass(
        args.data, args.list, data_fields=args.data_fields,
        validate=True, aug=False, chunksize=args.chunksize, args=args)
    vae = VAETrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')
    
    for _ in range(args.train_nepoch):  

        # Train for one epoch
        vae.train_one_epoch()

        # Evaluate model performance on validation set
        if vae.epoch % args.test_nepoch == 0:
            vae.test_val()

        # save model
        if vae.epoch % args.save_nepoch == 0:
            vae.save_model()
    return
           

if __name__ == "__main__":
    args = get_args_vae()
    run(args)