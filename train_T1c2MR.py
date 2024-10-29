import os
import argparse
from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_loader_kidney
from core.trainer import Trainer


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Trainer(args)

    loaders = Munch(train=get_loader_kidney(root=args.train_img_dir,
                                          data_txt_file=['train_source.txt',      # source
                                                         'train_target1.txt',     # target
                                                         'train_target2.txt',      # target
                                                         'train_target3.txt',],    # target
                                          img_size=args.img_size,
                                          batch_size=args.batch_size,
                                          prob=args.randcrop_prob,
                                          num_workers=args.num_workers),
                    val=get_loader_kidney(root=args.train_img_dir,
                                          data_txt_file=['val_source.txt',],
                                          img_size=args.img_size,
                                          batch_size=args.val_batch_size,
                                          prob=args.randcrop_prob,
                                          num_workers=args.num_workers),
                    )

    solver.train(loaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')
    parser.add_argument('--style_dim', type=int, default=4096, help='Dimensions of Style Dictionary')
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=8)

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=10, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_idt', type=float, default=1, help='weight for identity loss')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5, help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=80000, help='Number of total iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for D and G')
    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    # misc
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator')

    # directory for dataset
    parser.add_argument('--train_img_dir', type=str, default='/path/to/dataset', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='/path/to/dataset', help='Directory containing validation images')

    # directory for saving data
    expr_dir = 'output_dir'
    parser.add_argument('--expr_dir', type=str, default=expr_dir, help='Directory for saving training data')
    parser.add_argument('--sample_dir', type=str, default=f'{expr_dir}/samples', help='Directory for saving generated images')
    parser.add_argument('--show_dir', type=str, default=f'{expr_dir}/show_visul', help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default=f'{expr_dir}/checkpoints', help='Directory for saving network checkpoints')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=5000)

    args = parser.parse_args()
    main(args)
