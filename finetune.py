import argparse
import torch
import os

import lightning as L
from dotenv import load_dotenv


def get_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # fmt:off
    parser.add_argument('--config', help='path to configuration file')
    parser.add_argument("--seed", type=int, default=0, help='Set global seed')
    parser.add_argument("--run_id", type=str, required=True, help='Unique identifier for a run')

    parser.add_argument("--img_dirs", nargs='+', help='Folder(s) containing image data')
    parser.add_argument("--mask_dirs", nargs='+', help='Folder(s) containing segmentation masks')

    parser.add_argument("--log_dir", type=str, help='For storing artifacts')
    parser.add_argument("--wandb_project", type=str, required=True, help='Wandb project name')
    parser.add_argument("--num_gpus", type=int, default=4, help='number of gpus')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')

    parser.add_argument("--img_x_size", type=int, default=512, help='height of image')
    parser.add_argument("--img_y_size", type=int, default=512, help='width of image')

    parser.add_argument("--batch_size", type=int, default=10, help='Batch size to train with')
    parser.add_argument("--lr", type=float, default=0.0001, help='Max learning rate used during training') 
    parser.add_argument("--epochs", type=int, default=200, help='Number of training epochs') 
    parser.add_argument("--weight_decay", type=float, default=0.05, help='weight decay of optimizer')  ## from centralai codebase
    parser.add_argument("--pretrain_path", type=str, default=None, help="If starting training from a pretrained checkpoint, list the full path to the model with this flag.")
    # fmt:on

    args = parser.parse_args()

    return args


def main(args):
    pass


if __name__ == "__main__":
    args = get_args()
    print("Module Command Line Arguments: ", vars(args))
    load_dotenv()
    L.seed_everything(args.seed)

    # To speed up training
    torch.set_float32_matmul_precision("medium")

    main(args)
