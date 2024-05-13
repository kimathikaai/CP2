import argparse
import builtins
import copy
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime
from enum import Enum

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import wandb
# from mmcv.utils import Config
from mmengine.config import Config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import builder
import loader
from datasets.pretrain_dataset import DatasetType, get_pretrain_dataset
from networks.segment_network import PretrainType

DEFAULT_QUEUE_SIZE = 65536


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Copy-Paste Contrastive Pretraining on ImageNet')

    parser.add_argument('--config', help='path to configuration file')
    parser.add_argument("--run_id", required=True, type=str, help='Unique identifier for a run')

    # Logging
    parser.add_argument("--log_dir", type=str, required=True, help='Where to store logs')
    parser.add_argument("--wandb_project", type=str, required=True, help='Wandb project name')

    # Data
    parser.add_argument("--data_dirs", metavar='DIR', nargs='+', help='Folder(s) containing image data', required=True)
    parser.add_argument("--directory_type", type=str, choices=[x.name for x in DatasetType], default=DatasetType.FILENAME.name)
    parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    # Custom experimental hyper-parameters
    parser.add_argument('--lmbd_cp2_dense_loss', default=0.2, type=float)
    parser.add_argument('--same_foreground', action='store_true', help='Use the same foreground images for both bacgrounds')
    parser.add_argument('--cap_queue', action='store_true', help='Cap queue size to dataset size')
    parser.add_argument('--include_background', action='store_true', help='Include background aggregate pixels as negative pairs')
    parser.add_argument("--pretrain_type", type=str, choices=[x.name for x in PretrainType], default=PretrainType.CP2.name)

    # Distributed training
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-images', default=1281167, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='total batch size over all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--optim', default='sgd', help='optimizer')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--scalar-freq', default=100, type=int,
                        help='metrics writing frequency')
    parser.add_argument('--ckpt-freq', default=10, type=int,
                        help='checkpoint saving frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--output-stride', default=16, type=int,
                        help='output stride of encoder')
    # fmt: on

    args = parser.parse_args()
    # convert to enum
    args.directory_type = DatasetType[args.directory_type]
    args.pretrain_type = PretrainType[args.pretrain_type]

    return args


def cleanup():
    dist.destroy_process_group()


def setup(rank, args):
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank,
    )


def prepare_data(rank, num_workers, args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ]

    # simply use RandomErasing for Copy-Paste implementation:
    # erase a random block of background image and replace the erased positions by foreground
    augmentation_bg = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
        transforms.RandomErasing(p=1.0, scale=(0.5, 0.8), ratio=(0.8, 1.25), value=0.0),
    ]

    train_dataset = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        transform=loader.TwoCropsTransform(transforms.Compose(augmentation)),
        directory_type=args.directory_type,
        split_name="train",
    )
    train_dataset_bg = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        transform=transforms.Compose(augmentation_bg),
        directory_type=args.directory_type,
        split_name="train",
    )

    def get_dataloader(dataset, seed):
        sampler = DistributedSampler(
            dataset=train_dataset_bg,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size // args.world_size,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            sampler=sampler,
        )
        return dataloader, sampler

    train_loader, train_sampler = get_dataloader(train_dataset, 0)
    train_loader_bg0, train_sampler_bg0 = get_dataloader(train_dataset_bg, 1024)
    train_loader_bg1, train_sampler_bg1 = get_dataloader(train_dataset_bg, 2048)

    return (
        (train_loader, train_sampler),
        (train_loader_bg0, train_sampler_bg0),
        (train_loader_bg1, train_sampler_bg1),
    )


def setup_logger(rank, args):
    logger = logging.getLogger(__name__ + f"-rank{rank}")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d]-2s %(message)s"
    )

    # Rank 0 can output  to console
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

    # All logs should be sent to the files
    handler = logging.FileHandler(os.path.join(args.run_log_dir, f"log-rank{rank}.txt"))
    handler.setLevel(level=logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def main_worker(rank, args):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # If your model does not change and your input sizes remain the same
    # - then you may benefit from setting torch.backends.cudnn.benchmark = True.
    # However, if your model changes: for instance, if you have layers that
    # are only "activated" when certain conditions are met, or you have
    # layers inside a loop that can be iterated a different number of times,
    # then setting torch.backends.cudnn.benchmark = True might stall your execution.
    # Source: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    # cudnn.benchmark = True
    # Setting cudnn.deterministic as True will use the default algorithms, i.e.,
    # setting cudnn.benchmark as True will have no effect
    cudnn.deterministic = True

    # get configuration file
    cfg = Config.fromfile(args.config)

    # initialize wandb for the main process
    wandb_run = None
    if rank == 0:
        wandb_run = wandb.init(
            name=args.run_id,
            project=args.wandb_project,
            dir=args.run_log_dir,
            tags=["pretrain"],
        )
        # Add hyperparameters to config
        wandb_run.config.update({"hyper-parameters": vars(args)})
        wandb_run.config.update({"config_file": cfg})
        # define our custom x axis metric
        wandb_run.define_metric("step")
        # define which metrics will be plotted against it (e.g. all metrics
        # under 'train')
        wandb_run.define_metric("train/*", step_metric="step")
        wandb_run.define_metric("learning_rate", step_metric="step")

    # setup process groups
    setup(rank, args)
    # get logger
    logger = setup_logger(rank, args)

    #
    # Data
    #
    data_loaders = prepare_data(
        rank=rank,
        num_workers=args.num_workers_per_dataset,
        args=args,
    )
    (
        (train_loader, train_sampler),
        (train_loader_bg0, train_sampler_bg0),
        (train_loader_bg1, train_sampler_bg1),
    ) = data_loaders
    logger.info(f"Initialized data loaders ({rank = })")
    len_dataset = len(train_loader.dataset)
    logger.info(f"{len_dataset}")

    #
    # Model
    #
    # instantiate the model(it's your own model) and move it to the right device
    model = builder.CP2_MOCO(
        cfg,
        m=0.999 if args.pretrain_type == PretrainType.CP2 else 0.996,
        K=len_dataset if args.cap_queue else DEFAULT_QUEUE_SIZE,
        include_background=args.include_background,
        lmbd_cp2_dense_loss=args.lmbd_cp2_dense_loss,
        pretrain_type=args.pretrain_type,
        device=device,
        wandb_log=wandb_run,
        rank=rank,
    )
    model.to(device)
    logger.info(model)

    # Initialize the model pretrained ImageNet weights
    model.encoder_q.backbone.init_weights()
    model.encoder_k.backbone.init_weights()
    # import copy
    # # Initialize the model pretrained ImageNet weights
    # weights_before_q = copy.deepcopy(model.encoder_q.backbone)
    # model.encoder_q.backbone.init_weights()
    # weights_after_q = copy.deepcopy(model.encoder_q.backbone)
    # # print(f"{weights_before_q = }\n{weights_after_q = }")
    # weights_before_k = copy.deepcopy(model.encoder_k.backbone)
    # model.encoder_k.backbone.init_weights()
    # weights_after_k = copy.deepcopy(model.encoder_k.backbone)
    # # print(f"{weights_before_k = }\n{weights_after_k = }")
    # print(f"{torch.all(weights_before_q.layer4[2].bn3.bias == weights_after_q.layer4[2].bn3.bias) = }")

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )
    logger.info(f"Initialized the model ({rank = })")

    #
    # Optimizers
    #
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError("Only sgd and adamw optimizers are supported.")

    logger.info(f"Initialized optimizer ({rank = })")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("loading checkpoint '{}'".format(args.resume))
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    step = 0
    for epoch in range(args.start_epoch, args.epochs):
        # When using DistributedSampler, we have to specify the epoch
        train_sampler.set_epoch(epoch)
        train_sampler_bg0.set_epoch(epoch)
        train_sampler_bg1.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)

        if rank == 0:
            model.log({"epoch": epoch})
            model.log({"learning_rate": lr})

        # train for one epoch
        step = train(
            [train_loader, train_loader_bg0, train_loader_bg1],
            model,
            optimizer,
            epoch,
            args,
            step,
        )
        if epoch % args.ckpt_freq == args.ckpt_freq - 1:
            if rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        # 'arch': args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "pretrain_type": args.pretrain_type.name,
                    },
                    is_best=False,
                    filename=os.path.join(
                        args.log_dir,
                        args.run_id,
                        "checkpoint.ckpt",
                    ),
                )
    cleanup()


def train(
    train_loader_list,
    model,
    optimizer,
    epoch,
    args,
    step,
):
    train_loader, train_loader_bg0, train_loader_bg1 = train_loader_list
    model.train()

    for i, (images, bg0, bg1) in enumerate(
        zip(train_loader, train_loader_bg0, train_loader_bg1)
    ):
        # data_time.update(time.time() - end)
        idx_a = 0
        idx_b = idx_a if args.same_foreground else 1

        img_a = images[idx_a].to(model.device)
        img_b = images[idx_b].to(model.device)
        bg0 = bg0.to(model.device)
        bg1 = bg1.to(model.device)

        visualize = model.rank == 0 and epoch == 0 and i == 0
        loss = model(img_a, img_b, bg0, bg1, visualize)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    model.on_train_epoch_end()

    return step


def save_checkpoint(state, is_best, filename="checkpoint.ckpt"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "best_checkpoint.pth")


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info("    ".join(entries))
        if torch.distributed.get_rank() == 0:
            self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    # parse command line
    args = get_args()
    print(f"Args: f{vars(args) = }")

    # create logging dir
    args.run_log_dir = os.path.join(args.log_dir, args.run_id)
    os.mkdir(args.run_log_dir)
    print(f"{args.run_log_dir = }")

    print(f"{torch.cuda.device_count() = }")
    # assert args.world_size <= torch.cuda.device_count()

    # get the number of workers
    args.num_workers_per_dataset = args.num_workers // (args.world_size * 3)
    print(f"{args.num_workers_per_dataset = }")

    cfg = Config.fromfile(args.config)

    # spawn parallel process
    mp.spawn(main_worker, args=[args], nprocs=args.world_size)
